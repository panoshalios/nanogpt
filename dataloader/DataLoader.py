from pathlib import Path

import numpy as np
import torch


class DataLoader:
    """Load fixed-length, next-token prediction batches from a uint16 file."""

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
        self.num_samples = (len(self._tokens) - 1) // block_size
        self.num_batches = self.num_samples // batch_size
        if self.num_batches == 0:
            required = batch_size * block_size + 1
            raise ValueError(
                f"{self.file_path} has {len(self._tokens)} tokens; "
                f"at least {required} are required for one batch"
            )

        self._sample_indices = np.arange(self.num_samples)
        self._batch_index = 0
        self._reset()

    def _reset(self) -> None:
        self._batch_index = 0
        if self.shuffle:
            self._rng.shuffle(self._sample_indices)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        self._reset()
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._batch_index >= self.num_batches:
            raise StopIteration

        first = self._batch_index * self.batch_size
        sample_indices = self._sample_indices[first : first + self.batch_size]
        starts = sample_indices * self.block_size

        x = np.stack([self._tokens[start : start + self.block_size] for start in starts])
        y = np.stack([self._tokens[start + 1 : start + self.block_size + 1] for start in starts])
        self._batch_index += 1

        # Embedding indices and cross-entropy targets must be torch.long.
        x_tensor = torch.from_numpy(x.astype(np.int64))
        y_tensor = torch.from_numpy(y.astype(np.int64))
        if self.pin_memory:
            x_tensor = x_tensor.pin_memory()
            y_tensor = y_tensor.pin_memory()
        return x_tensor, y_tensor

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return next(self)
