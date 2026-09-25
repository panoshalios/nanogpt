import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataloader import DataLoader
from model.gpt2 import GPT2, GPT2Config
from train_utils import (
    DistributedContext,
    cleanup_distributed,
    create_optimizer,
    get_lr,
    setup_distributed,
    supports_bfloat16,
    supports_tf32,
    synchronize,
)

SEED = 1337
# GPT-2's tokenizer (tiktoken) has 50,257 tokens. Rounding up to a multiple of 128 gives
# the embedding and output matmuls GPU-friendly shapes; the extra ids are never used.
VOCAB_SIZE = 50_304
# Shards written by input/fineweb_edu/prepare.py
DATA_DIR = Path(__file__).parent / "input" / "fineweb_edu"
TRAIN_SHARDS = sorted(DATA_DIR.glob("fineweb_edu_train_*.bin"))
VAL_SHARDS = sorted(DATA_DIR.glob("fineweb_edu_val_*.bin"))

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

# Validation: every EVAL_INTERVAL optimizer steps, measure the loss on the first
# EVAL_TOKENS tokens of the validation shard. The same tokens are used every time, so
# the numbers are comparable across the run.
EVAL_INTERVAL = 250
EVAL_TOKENS = 10_485_760  # 20 * 2**19, split across all ranks


def main() -> None:
    # Launch on N GPUs with: torchrun --standalone --nproc_per_node=N train.py
    # Plain `python train.py` runs on a single device as before.
    ddp = setup_distributed()
    try:
        train(ddp)
    finally:
        cleanup_distributed(ddp)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    num_batches: int,
    ddp: DistributedContext,
    use_bfloat16: bool,
    pin_memory: bool,
) -> float:
    """Mean validation loss over num_batches batches per rank, averaged across ranks."""
    device = ddp.device
    # eval() turns off dropout, so the loss is deterministic for fixed weights.
    model.eval()
    loss_accum = torch.zeros((), device=device)

    # iter() restarts the loader at the beginning of the (unshuffled) validation data.
    val_iter = iter(val_loader)
    for _ in range(num_batches):
        x, y = next(val_iter)
        x = x.to(device, non_blocking=pin_memory)
        y = y.to(device, non_blocking=pin_memory)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
            _, loss = model(x, y)
        loss_accum += loss.detach() / num_batches

    if ddp.enabled:
        # Each rank evaluated a different share of the validation tokens.
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    model.train()
    return loss_accum.item()


def train(ddp: DistributedContext) -> None:
    device = ddp.device
    use_bfloat16 = supports_bfloat16(device)
    use_tf32 = supports_tf32(device)
    # "high" runs float32 matmuls in TF32. Everything else stays in full float32.
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")

    # Same seed everywhere so every rank starts from identical weights (DDP also
    # broadcasts rank 0's weights when it wraps the model).
    torch.manual_seed(SEED)
    config = GPT2Config(vocab_size=VOCAB_SIZE)
    model = GPT2(config).to(device)
    model = torch.compile(model)
    if ddp.enabled:
        # DDP averages gradients across all ranks during backward().
        model = DDP(model, device_ids=[ddp.local_rank] if device.type == "cuda" else None)
    optimizer = create_optimizer(
        model, device, lr=MAX_LR, weight_decay=WEIGHT_DECAY, verbose=ddp.is_master
    )

    # Pinned memory enables non-blocking CPU-to-CUDA transfers. It is not used
    # for CPU or MPS because those backends do not benefit from CUDA pinning.
    pin_memory = device.type == "cuda"
    if not TRAIN_SHARDS or not VAL_SHARDS:
        raise FileNotFoundError(
            f"Missing training or validation shards in {DATA_DIR}. "
            "Run: python input/fineweb_edu/prepare.py"
        )
    data_loader = DataLoader(
        TRAIN_SHARDS,
        batch_size=MICRO_BATCH_SIZE,
        block_size=config.block_size,
        shuffle=True,
        seed=SEED,
        pin_memory=pin_memory,
        rank=ddp.rank,
        world_size=ddp.world_size,
    )
    # Unshuffled, so every evaluation reads the same tokens; still split by rank.
    val_loader = DataLoader(
        VAL_SHARDS,
        batch_size=MICRO_BATCH_SIZE,
        block_size=config.block_size,
        shuffle=False,
        pin_memory=pin_memory,
        rank=ddp.rank,
        world_size=ddp.world_size,
    )

    # Every rank processes grad_accum_steps micro-batches per step, so the global batch
    # is split across both accumulation and GPUs.
    tokens_per_micro_batch = MICRO_BATCH_SIZE * config.block_size
    tokens_per_accum_step = tokens_per_micro_batch * ddp.world_size
    assert TOTAL_BATCH_SIZE % tokens_per_accum_step == 0, (
        "TOTAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE * block_size * world_size"
    )
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_accum_step
    # Capped at one pass so no validation token is counted twice.
    eval_batches = min(EVAL_TOKENS // tokens_per_accum_step, len(val_loader))
    assert eval_batches > 0, "EVAL_TOKENS is smaller than one micro-batch on every rank"

    if ddp.is_master:
        precision = "bfloat16 mixed precision" if use_bfloat16 else "float32"
        matmul = "tf32" if use_tf32 else "fp32"
        print(f"Using device: {device} x {ddp.world_size} ({precision}, {matmul} matmul)")
        print(
            f"Batch: {TOTAL_BATCH_SIZE:,} tokens/step = {ddp.world_size} GPUs x "
            f"{grad_accum_steps} micro-batches of {MICRO_BATCH_SIZE} x {config.block_size}"
        )
        pass_tokens = len(data_loader) * tokens_per_micro_batch * ddp.world_size
        print(f"Data: {len(TRAIN_SHARDS)} shards, {pass_tokens:,} tokens per pass")
        val_tokens = eval_batches * tokens_per_accum_step
        print(f"Validation: {val_tokens:,} tokens every {EVAL_INTERVAL} steps")

    def run_validation(steps_done: int) -> None:
        val_loss = evaluate(model, val_loader, eval_batches, ddp, use_bfloat16, pin_memory)
        if ddp.is_master:
            print(f"Step {steps_done}/{MAX_STEPS}: val_loss={val_loss:.4f}")

    # Before any training: should be close to ln(VOCAB_SIZE) ~= 10.83.
    run_validation(0)

    model.train()
    for step in range(MAX_STEPS):
        synchronize(device)
        start_time = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)

        for micro_step in range(grad_accum_steps):
            x, y = data_loader.next_batch()
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)

            # DDP would all-reduce gradients on every backward(). Only the last
            # micro-batch needs it; before that, gradients just add up locally.
            is_last_micro_step = micro_step == grad_accum_steps - 1
            sync_context = (
                model.no_sync() if ddp.enabled and not is_last_micro_step else nullcontext()
            )

            with sync_context:
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

        if ddp.enabled:
            # loss_accum only covers this rank's micro-batches. Average it across
            # ranks so the logged loss is for the full global batch.
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Clip gradient norm (of the full-batch gradient). Gradients are already
        # identical on every rank after the all-reduce, so each rank clips the same way.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Set this step's learning rate before the optimizer uses it.
        lr = get_lr(step, MAX_STEPS, WARMUP_STEPS, MAX_LR, MIN_LR)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        synchronize(device)

        elapsed = time.perf_counter() - start_time
        tokens_per_second = TOTAL_BATCH_SIZE / elapsed if elapsed > 0 else float("inf")
        if ddp.is_master:
            print(
                f"Step {step + 1}/{MAX_STEPS}: loss={loss_accum.item():.4f}, lr={lr:.2e}, "
                f"norm={norm.item():.4f}, time={elapsed:.2f}s, "
                f"throughput={tokens_per_second:.2f} tokens/s"
            )

        steps_done = step + 1
        if steps_done % EVAL_INTERVAL == 0 or steps_done == MAX_STEPS:
            run_validation(steps_done)


if __name__ == "__main__":
    main()
