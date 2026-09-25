from pathlib import Path

import numpy as np
import torch


class DataLoader:
    """Load fixed-length, next-token prediction batches from a uint16 file.

    The tokens are cut into non-overlapping chunks of block_size tokens (plus one
    following target token). Each pass over the data reads every chunk exactly once,
    and with several processes (DDP) each rank reads a disjoint share of the chunks:
    rank r takes chunks r, r + world_size, r + 2 * world_size, ...

    With shuffle=True, every pass shifts the chunk grid by a random offset in
    [0, block_size) and shuffles the chunk order, so chunk boundaries and order change
    between passes. The offset and order come from (seed, pass index) only, so every
    rank computes the same permutation and the shares stay disjoint. seed must therefore
    be the same on all ranks.

    With shuffle=False, chunks start at token 0 and are read in file order, which is
    deterministic and suited to evaluation.
    """

    def __init__(
        self,
        file_path: str | Path,
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

        self.file_path = Path(file_path)
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Token file not found: {self.file_path}")

        self.batch_size = batch_size
        self.block_size = block_size
        self.shuffle = shuffle
        self.seed = seed
        self.pin_memory = pin_memory
        self.rank = rank
        self.world_size = world_size
        self._tokens = np.memmap(self.file_path, dtype=np.uint16, mode="r")

        # Each chunk needs block_size input tokens and one following target. With
        # shuffle the grid can start as late as block_size - 1, so count chunks for that
        # worst case. The count must not depend on the offset: every pass (and every
        # rank) needs the same number of batches, or DDP ranks would get out of step.
        max_offset = block_size - 1 if shuffle else 0
        self.num_chunks = (len(self._tokens) - 1 - max_offset) // block_size

        # Drop the remainder so every rank gets the same number of full batches.
        self.chunks_per_rank = self.num_chunks // world_size
        self.num_batches = self.chunks_per_rank // batch_size
        if self.num_batches == 0:
            required = batch_size * block_size * world_size + 1 + max_offset
            raise ValueError(
                f"{self.file_path} has {len(self._tokens)} tokens; "
                f"at least {required} are required for one batch on each of "
                f"{world_size} rank(s)"
            )

        self._offsets = np.arange(block_size + 1)
        self._pass_index = 0  # how many passes over the data have started
        self._chunk_starts: np.ndarray | None = None  # this rank's starts for the pass
        self._batch_index = 0

    def __len__(self) -> int:
        """Number of batches this rank reads in one pass over the data."""
        return self.num_batches

    def _start_pass(self) -> None:
        if self.shuffle:
            # Seeded by (seed, pass) so all ranks compute the same offset and order.
            rng = np.random.default_rng([self.seed, self._pass_index])
            offset = int(rng.integers(0, self.block_size))
            order = rng.permutation(self.num_chunks)
        else:
            offset = 0
            order = np.arange(self.num_chunks)

        # Interleave chunks across ranks, then keep this rank's equal-sized share.
        my_chunks = order[self.rank :: self.world_size][: self.chunks_per_rank]
        self._chunk_starts = offset + my_chunks * self.block_size
        self._pass_index += 1
        self._batch_index = 0

    def __iter__(self):
        self._start_pass()
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._chunk_starts is None:
            self._start_pass()
        if self._batch_index >= self.num_batches:
            raise StopIteration

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
        if self._chunk_starts is None or self._batch_index >= self.num_batches:
            self._start_pass()
        return next(self)
