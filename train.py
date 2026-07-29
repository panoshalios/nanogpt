import time
from pathlib import Path

import torch

from dataloader import DataLoader
from model.gpt2 import GPT2, GPT2Config

EPOCHS = 10
BATCH_SIZE = 16
VOCAB_SIZE = 400
TRAIN_DATA = Path(__file__).parent / "input" / "shakespeare" / "train.bin"


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


def main() -> None:
    device = get_device()
    use_bfloat16 = supports_bfloat16(device)
    config = GPT2Config(vocab_size=VOCAB_SIZE)
    model = GPT2(config).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

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
    print(f"Using device: {device} ({precision})")
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
            optimizer.step()
            total_loss += loss.detach()

        synchronize(device)
        elapsed = time.perf_counter() - start_time
        average_loss = (total_loss / len(data_loader)).item()
        print(f"Epoch {epoch + 1}: loss={average_loss:.4f}, time={elapsed:.2f}s")


if __name__ == "__main__":
    main()
