from pathlib import Path

import numpy as np
import torch


class DataLoader:
    """Load fixed-length, next-token prediction batches from a uint16 file.

    With shuffle=True, every sample starts at a uniformly random token offset, so
    each epoch sees different windows of the data. With shuffle=False, samples are
    consecutive non-overlapping chunks in file order, which is deterministic and
    suited to evaluation.
    """

    def __init__(
        self,
        file_path: str | Path,
        batch_size: int,
        block_size: int,
        shuffle: bool = False,
        seed: int | None = None,
        pin_memory: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.file_path = Path(file_path)
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Token file not found: {self.file_path}")

        self.batch_size = batch_size
        self.block_size = block_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self._rng = np.random.default_rng(seed)
        self._tokens = np.memmap(self.file_path, dtype=np.uint16, mode="r")

        # Each sample needs block_size input tokens and one following target.
        # An epoch covers roughly one pass over the tokens in either mode.
        self.num_samples = (len(self._tokens) - 1) // block_size
        self.num_batches = self.num_samples // batch_size
        if self.num_batches == 0:
            required = batch_size * block_size + 1
            raise ValueError(
                f"{self.file_path} has {len(self._tokens)} tokens; "
                f"at least {required} are required for one batch"
            )

        # Largest valid start so that start + block_size is still a valid target index.
        self._max_start = len(self._tokens) - block_size - 1
        self._offsets = np.arange(block_size + 1)
        self._batch_index = 0

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        self._batch_index = 0
        return self

    def _batch_starts(self) -> np.ndarray:
        if self.shuffle:
            return self._rng.integers(0, self._max_start + 1, size=self.batch_size)
        first = self._batch_index * self.batch_size
        return np.arange(first, first + self.batch_size) * self.block_size

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._batch_index >= self.num_batches:
            raise StopIteration

        starts = self._batch_starts()
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
        return next(self)
