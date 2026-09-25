"""Device, distributed, logging, learning rate schedule and optimizer helpers for train.py."""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DistributedContext:
    enabled: bool  # True when launched with torchrun
    rank: int  # global process index, 0 .. world_size - 1
    local_rank: int  # GPU index on this machine
    world_size: int  # total number of processes (GPUs)
    device: torch.device

    @property
    def is_master(self) -> bool:
        # Rank 0 does logging (and later checkpointing) so it only happens once.
        return self.rank == 0


def setup_distributed() -> DistributedContext:
    # torchrun sets RANK, LOCAL_RANK and WORLD_SIZE for every process it launches.
    # Without them this is a normal single-process run.
    if "RANK" not in os.environ:
        return DistributedContext(
            enabled=False, rank=0, local_rank=0, world_size=1, device=get_device()
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        # Each process drives exactly one GPU and talks to the others over NCCL.
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        # CPU fallback (gloo) so the distributed code path can be tested without GPUs.
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend)
    return DistributedContext(
        enabled=True, rank=rank, local_rank=local_rank, world_size=world_size, device=device
    )


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled:
        dist.destroy_process_group()


class MetricsLogger:
    """Appends one JSON object per line (JSON Lines) to a log file.

    Load it with pandas.read_json(path, lines=True), or follow it live with tail -f.
    Pass path=None on non-master ranks to make every call a no-op, so each record is
    written once.
    """

    def __init__(self, path: Path | None):
        self.path = path
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Line buffered: every record reaches the OS as soon as it is written, so a
            # crash or a terminated instance loses nothing already logged.
            self._file = open(path, "a", buffering=1, encoding="utf-8")

    def log(self, **record) -> None:
        if self._file is not None:
            self._file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def supports_bfloat16(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.cuda.is_bf16_supported()
    if device.type == "mps":
        return torch.backends.mps.is_macos_or_newer(14, 0)
    return False


def supports_tf32(device: torch.device) -> bool:
    # TF32 tensor cores exist on Ampere (compute capability 8.0) and newer.
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8


def get_lr(step: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    # 1) Linear warmup from near zero up to max_lr.
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) After decay finishes, hold at min_lr.
    if step >= max_steps:
        return min_lr
    # 3) In between, cosine decay from max_lr down to min_lr.
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # goes 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)


def create_optimizer(
    model: torch.nn.Module,
    device: torch.device,
    lr: float,
    weight_decay: float,
    verbose: bool = True,
) -> torch.optim.AdamW:
    # GPT-3 style weight decay: only 2D parameters (Linear weights and embeddings) are
    # decayed. 1D parameters (biases and LayerNorm weights) are not. Decaying them only
    # pulls them toward zero without regularizing anything useful.
    # parameters() yields the tied embedding / output weight only once.
    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() >= 2]
    no_decay_params = [p for p in params if p.dim() < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    num_decay = sum(p.numel() for p in decay_params)
    num_no_decay = sum(p.numel() for p in no_decay_params)
    if verbose:
        print(f"Decayed params: {len(decay_params)} tensors, {num_decay:,} values")
        print(f"Non-decayed params: {len(no_decay_params)} tensors, {num_no_decay:,} values")

    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device.type == "cuda",
    )
