from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

TOKEN_BYTES = 2  # tokens are stored as uint16


class DataLoader:
    """Load fixed-length, next-token prediction batches from uint16 token shards.

    The data is one or more shard files, read one shard at a time, so only the current
    shard has to be in memory. Within a shard, the tokens are cut into non-overlapping
    chunks of block_size tokens (plus one following target token). Chunks never cross a
    shard boundary. Each pass over the data reads every chunk exactly once, and with
    several processes (DDP) each rank reads a disjoint share of every shard's chunks:
    rank r takes chunks r, r + world_size, r + 2 * world_size, ...

    With shuffle=True, every pass visits the shards in a random order, and within each
    shard shifts the chunk grid by a random offset in [0, block_size) and shuffles the
    chunk order. All of this is seeded from (seed, pass, shard) only, so every rank
    computes the same order and the shares stay disjoint. seed must therefore be the
    same on all ranks.

    With shuffle=False, shards and chunks are read in order starting at token 0, which
    is deterministic and suited to evaluation.
    """

    def __init__(
        self,
        file_paths: str | Path | Sequence[str | Path],
        batch_size: int,
        block_size: int,
        shuffle: bool = False,
        seed: int = 0,
        pin_memory: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(f"invalid rank {rank} for world_size {world_size}")

        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        self.file_paths = [Path(path) for path in file_paths]
        if not self.file_paths:
            raise ValueError("no token files given")
        for path in self.file_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Token file not found: {path}")

        self.batch_size = batch_size
        self.block_size = block_size
        self.shuffle = shuffle
        self.seed = seed
        self.pin_memory = pin_memory
        self.rank = rank
        self.world_size = world_size

        # Batch counts come from file sizes, so shards are only opened when read.
        # With shuffle the grid can start as late as block_size - 1, so count chunks for
        # that worst case. The count must not depend on the offset: every pass (and every
        # rank) needs the same number of batches, or DDP ranks would get out of step.
        self._max_offset = block_size - 1 if shuffle else 0
        self._shard_lengths = [path.stat().st_size // TOKEN_BYTES for path in self.file_paths]
        self._shard_num_batches = [self._num_batches(length) for length in self._shard_lengths]
        self.num_batches = sum(self._shard_num_batches)
        if self.num_batches == 0:
            required = batch_size * block_size * world_size + 1 + self._max_offset
            raise ValueError(
                f"no shard has enough tokens for one batch on each of {world_size} rank(s); "
                f"a shard needs at least {required} tokens"
            )

        self._offsets = np.arange(block_size + 1)
        self._pass_index = 0  # how many passes over the data have started
        self._shard_queue: list[int] = []  # shards still to read in the current pass
        self._tokens: np.ndarray | None = None  # memmap of the current shard
        self._chunk_starts: np.ndarray | None = None  # this rank's starts in the shard
        self._batch_index = 0  # next batch within the current shard

    def _num_batches(self, shard_length: int) -> int:
        # Every chunk needs block_size + 1 tokens starting at or after the offset.
        num_chunks = max(0, (shard_length - 1 - self._max_offset) // self.block_size)
        # Drop the remainder so every rank gets the same number of full batches.
        chunks_per_rank = num_chunks // self.world_size
        return chunks_per_rank // self.batch_size

    def __len__(self) -> int:
        """Number of batches this rank reads in one pass over all shards."""
        return self.num_batches

    def _start_pass(self) -> None:
        num_shards = len(self.file_paths)
        if self.shuffle:
            shard_order = np.random.default_rng([self.seed, self._pass_index]).permutation(
                num_shards
            )
        else:
            shard_order = np.arange(num_shards)
        # Skip shards too small to give every rank a batch.
        self._shard_queue = [int(s) for s in shard_order if self._shard_num_batches[s] > 0]
        self._pass_index += 1
        self._load_next_shard()

    def _load_next_shard(self) -> None:
        shard = self._shard_queue.pop(0)
        length = self._shard_lengths[shard]
        num_chunks = (length - 1 - self._max_offset) // self.block_size

        if self.shuffle:
            # Seeded by (seed, pass, shard) so all ranks compute the same offset and order.
            rng = np.random.default_rng([self.seed, self._pass_index, shard])
            offset = int(rng.integers(0, self.block_size))
            order = rng.permutation(num_chunks)
        else:
            offset = 0
            order = np.arange(num_chunks)

        # Interleave chunks across ranks, then keep this rank's share of full batches.
        num_used = self._shard_num_batches[shard] * self.batch_size
        my_chunks = order[self.rank :: self.world_size][:num_used]
        self._chunk_starts = offset + my_chunks * self.block_size
        self._tokens = np.memmap(self.file_paths[shard], dtype=np.uint16, mode="r")
        self._batch_index = 0

    def _pass_finished(self) -> bool:
        shard_finished = self._batch_index * self.batch_size >= len(self._chunk_starts)
        return shard_finished and not self._shard_queue

    def __iter__(self):
        self._start_pass()
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._chunk_starts is None:
            self._start_pass()
        if self._pass_finished():
            raise StopIteration
        if self._batch_index * self.batch_size >= len(self._chunk_starts):
            self._load_next_shard()

        first = self._batch_index * self.batch_size
        starts = self._chunk_starts[first : first + self.batch_size]
        # (batch_size, block_size + 1) window; x and y are its two overlapping views.
        window = self._tokens[starts[:, None] + self._offsets].astype(np.int64)
        self._batch_index += 1

        # Embedding indices and cross-entropy targets must be torch.long.
        x_tensor = torch.from_numpy(window[:, :-1].copy())
        y_tensor = torch.from_numpy(window[:, 1:].copy())
        if self.pin_memory:
            x_tensor = x_tensor.pin_memory()
            y_tensor = y_tensor.pin_memory()
        return x_tensor, y_tensor

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next batch, starting a new pass over the data when one ends."""
        if self._chunk_starts is None or self._pass_finished():
            self._start_pass()
        return next(self)
