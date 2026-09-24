"""Device, learning rate schedule and optimizer helpers used by train.py."""

import math

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    model: torch.nn.Module, device: torch.device, lr: float, weight_decay: float
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
    print(f"Decayed params: {len(decay_params)} tensors, {num_decay:,} values")
    print(f"Non-decayed params: {len(no_decay_params)} tensors, {num_no_decay:,} values")

    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device.type == "cuda",
    )
