import math
import time
from pathlib import Path

import torch

from dataloader import DataLoader
from model.gpt2 import GPT2, GPT2Config

VOCAB_SIZE = 1024
TRAIN_DATA = Path(__file__).parent / "input" / "shakespeare" / "train.bin"

# GPT-3 Small trains with ~0.5M tokens per optimizer step (GPT-3 paper, Table 2.1).
# That does not fit on the GPU at once, so each step accumulates gradients over
# several micro-batches of MICRO_BATCH_SIZE sequences of block_size tokens.
TOTAL_BATCH_SIZE = 524_288  # 2**19 tokens per optimizer step
MICRO_BATCH_SIZE = 16  # sequences per forward/backward pass

# Training length in optimizer steps. 19_073 steps * 524_288 tokens ~= 10B tokens.
MAX_STEPS = 19_073

# GPT-3 learning rate schedule (GPT-3 paper, Appendix B): linear warmup, then cosine
# decay down to 10% of the peak. MAX_LR is the value used for GPT-3 Small (125M).
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 715  # GPT-3 warms up over 375M tokens: 375e6 / 524_288 ~= 715 steps
WEIGHT_DECAY = 0.1


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


def get_lr(step: int, max_steps: int) -> float:
    # 1) Linear warmup from near zero up to MAX_LR.
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    # 2) After decay finishes, hold at MIN_LR.
    if step >= max_steps:
        return MIN_LR
    # 3) In between, cosine decay from MAX_LR down to MIN_LR.
    decay_ratio = (step - WARMUP_STEPS) / (max_steps - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # goes 1 -> 0
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


def create_optimizer(model: torch.nn.Module, device: torch.device) -> torch.optim.AdamW:
    # GPT-3 style weight decay: only 2D parameters (Linear weights and embeddings) are
    # decayed. 1D parameters (biases and LayerNorm weights) are not. Decaying them only
    # pulls them toward zero without regularizing anything useful.
    # parameters() yields the tied embedding / output weight only once.
    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() >= 2]
    no_decay_params = [p for p in params if p.dim() < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    num_decay = sum(p.numel() for p in decay_params)
    num_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Decayed params: {len(decay_params)} tensors, {num_decay:,} values")
    print(f"Non-decayed params: {len(no_decay_params)} tensors, {num_no_decay:,} values")

    return torch.optim.AdamW(
        param_groups,
        lr=MAX_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device.type == "cuda",
    )


def main() -> None:
    device = get_device()
    use_bfloat16 = supports_bfloat16(device)
    use_tf32 = supports_tf32(device)
    # "high" runs float32 matmuls in TF32. Everything else stays in full float32.
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    config = GPT2Config(vocab_size=VOCAB_SIZE)
    model = GPT2(config).to(device)
    model = torch.compile(model)
    optimizer = create_optimizer(model, device)

    # Pinned memory enables non-blocking CPU-to-CUDA transfers. It is not used
    # for CPU or MPS because those backends do not benefit from CUDA pinning.
    pin_memory = device.type == "cuda"
    data_loader = DataLoader(
        TRAIN_DATA,
        batch_size=MICRO_BATCH_SIZE,
        block_size=config.block_size,
        shuffle=True,
        pin_memory=pin_memory,
    )

    tokens_per_micro_batch = MICRO_BATCH_SIZE * config.block_size
    assert TOTAL_BATCH_SIZE % tokens_per_micro_batch == 0, (
        "TOTAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE * block_size"
    )
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_micro_batch

    precision = "bfloat16 mixed precision" if use_bfloat16 else "float32"
    matmul = "tf32" if use_tf32 else "fp32"
    print(f"Using device: {device} ({precision}, {matmul} matmul)")
    print(
        f"Batch: {TOTAL_BATCH_SIZE:,} tokens/step = {grad_accum_steps} micro-batches "
        f"of {MICRO_BATCH_SIZE} x {config.block_size}"
    )

    model.train()
    for step in range(MAX_STEPS):
        synchronize(device)
        start_time = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)

        for _ in range(grad_accum_steps):
            x, y = data_loader.next_batch()
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                _, loss = model(x, y)

            # The loss is a mean over one micro-batch. backward() adds gradients
            # together, so divide by grad_accum_steps to get the mean over the
            # full batch, which is what one big batch would have produced.
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        # Clip gradient norm (of the full-batch gradient)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Set this step's learning rate before the optimizer uses it.
        lr = get_lr(step, MAX_STEPS)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        synchronize(device)

        elapsed = time.perf_counter() - start_time
        tokens_per_second = TOTAL_BATCH_SIZE / elapsed if elapsed > 0 else float("inf")
        print(
            f"Step {step + 1}/{MAX_STEPS}: loss={loss_accum.item():.4f}, lr={lr:.2e}, "
            f"norm={norm.item():.4f}, time={elapsed:.2f}s, "
            f"throughput={tokens_per_second:.2f} tokens/s"
        )


if __name__ == "__main__":
    main()
