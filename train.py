import time
from pathlib import Path

import torch

from dataloader import DataLoader
from model.gpt2 import GPT2, GPT2Config
from train_utils import (
    create_optimizer,
    get_device,
    get_lr,
    supports_bfloat16,
    supports_tf32,
    synchronize,
)

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


def main() -> None:
    device = get_device()
    use_bfloat16 = supports_bfloat16(device)
    use_tf32 = supports_tf32(device)
    # "high" runs float32 matmuls in TF32. Everything else stays in full float32.
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    config = GPT2Config(vocab_size=VOCAB_SIZE)
    model = GPT2(config).to(device)
    model = torch.compile(model)
    optimizer = create_optimizer(model, device, lr=MAX_LR, weight_decay=WEIGHT_DECAY)

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
        lr = get_lr(step, MAX_STEPS, WARMUP_STEPS, MAX_LR, MIN_LR)
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
