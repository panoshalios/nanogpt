import math
import time
from pathlib import Path

import torch

from dataloader import DataLoader
from model.gpt2 import GPT2, GPT2Config

EPOCHS = 50
BATCH_SIZE = 16
VOCAB_SIZE = 1024
TRAIN_DATA = Path(__file__).parent / "input" / "shakespeare" / "train.bin"

# GPT-3 learning rate schedule (GPT-3 paper, Appendix B): linear warmup, then cosine
# decay down to 10% of the peak. MAX_LR is the value used for GPT-3 Small (125M).
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 50


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


def main() -> None:
    device = get_device()
    use_bfloat16 = supports_bfloat16(device)
    use_tf32 = supports_tf32(device)
    # "high" runs float32 matmuls in TF32. Everything else stays in full float32.
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    config = GPT2Config(vocab_size=VOCAB_SIZE)
    model = GPT2(config).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MAX_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device.type == "cuda",
        weight_decay=0.1,
    )

    # Pinned memory enables non-blocking CPU-to-CUDA transfers. It is not used
    # for CPU or MPS because those backends do not benefit from CUDA pinning.
    pin_memory = device.type == "cuda"
    data_loader = DataLoader(
        TRAIN_DATA,
        batch_size=BATCH_SIZE,
        block_size=config.block_size,
        shuffle=True,
        pin_memory=pin_memory,
    )

    precision = "bfloat16 mixed precision" if use_bfloat16 else "float32"
    matmul = "tf32" if use_tf32 else "fp32"
    print(f"Using device: {device} ({precision}, {matmul} matmul)")

    # The schedule is defined over optimizer steps, and the decay spans the whole run.
    max_steps = EPOCHS * len(data_loader)
    step = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = torch.zeros((), device=device)
        synchronize(device)
        start_time = time.perf_counter()

        for x, y in data_loader:
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                _, loss = model(x, y)

            loss.backward()

            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Set this step's learning rate before the optimizer uses it.
            lr = get_lr(step, max_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()
            step += 1
            total_loss += loss.detach()

        synchronize(device)

        elapsed = time.perf_counter() - start_time
        average_loss = (total_loss / len(data_loader)).item()
        num_tokens = len(data_loader) * BATCH_SIZE * config.block_size
        tokens_per_second = num_tokens / elapsed if elapsed > 0 else float("inf")
        print(
            f"Epoch {epoch + 1}: loss={average_loss:.4f}, lr={lr:.2e}, time={elapsed:.2f}s, "
            f"throughput={tokens_per_second:.2f} tokens/s"
        )


if __name__ == "__main__":
    main()
