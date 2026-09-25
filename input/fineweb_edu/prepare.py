# Downloads FineWeb-Edu (sample-10BT, ~10B tokens), tokenizes it with GPT-2's tokenizer
# and writes it to disk as shards of raw uint16 token IDs that DataLoader memory-maps.
#
# Run from the repository root:
#   python input/fineweb_edu/prepare.py [--out-dir DIR] [--num-proc N]
#
# Output (~20 GB for 10B tokens at 2 bytes per token):
#   fineweb_edu_val_000000.bin     first shard, held out for validation
#   fineweb_edu_train_000001.bin   training shards, SHARD_SIZE tokens each
#   ...
#
# The raw dataset download is cached by Hugging Face under $HF_HOME (default
# ~/.cache/huggingface). On a cloud instance, point HF_HOME at the large disk first.
import argparse
import multiprocessing as mp
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATASET = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
SHARD_SIZE = 100_000_000  # tokens per shard (200 MB on disk)

# Module level so every worker process builds its own encoder once.
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # <|endoftext|>, id 50256


def tokenize(text: str) -> np.ndarray:
    # EOT goes before every document, so the model sees where one document ends and
    # the next begins. encode_ordinary treats any "<|endoftext|>" inside the text as
    # plain text, so a web page cannot insert fake document boundaries.
    tokens = [EOT] + enc.encode_ordinary(text)
    # GPT-2 token ids are < 50,257, so they fit in uint16 (max 65,535).
    return np.array(tokens, dtype=np.uint16)


def shard_path(out_dir: Path, index: int) -> Path:
    split = "val" if index == 0 else "train"
    return out_dir / f"fineweb_edu_{split}_{index:06d}.bin"


def write_shard(path: Path, tokens: np.ndarray) -> None:
    # Write to a temporary file and rename it, so an interrupted run never leaves a
    # truncated shard that looks complete.
    tmp_path = path.with_suffix(".tmp")
    tokens.tofile(tmp_path)
    os.replace(tmp_path, path)


def write_shards(texts: Iterable[str], out_dir: Path, shard_size: int, num_proc: int) -> list[Path]:
    """Tokenize texts in parallel and write them as consecutive shards of shard_size tokens."""
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer = np.empty(shard_size, dtype=np.uint16)
    filled = 0
    index = 0
    paths: list[Path] = []
    progress = tqdm(total=shard_size, unit="tok", unit_scale=True, desc=f"Shard {index}")

    with mp.Pool(num_proc) as pool:
        # imap keeps document order, so the output is deterministic.
        for tokens in pool.imap(tokenize, texts, chunksize=16):
            while len(tokens) > 0:
                # Copy as much of this document as fits in the current shard.
                n = min(shard_size - filled, len(tokens))
                buffer[filled : filled + n] = tokens[:n]
                filled += n
                progress.update(n)
                tokens = tokens[n:]

                if filled == shard_size:
                    path = shard_path(out_dir, index)
                    write_shard(path, buffer)
                    paths.append(path)
                    if index == 0:
                        # Drop the rest of the document that filled the validation
                        # shard, so no document is split between val and train.
                        tokens = tokens[:0]
                    index += 1
                    filled = 0
                    progress.close()
                    progress = tqdm(
                        total=shard_size, unit="tok", unit_scale=True, desc=f"Shard {index}"
                    )

    # The last shard is usually partial.
    if filled > 0:
        path = shard_path(out_dir, index)
        write_shard(path, buffer[:filled])
        paths.append(path)
    progress.close()
    return paths


def summarize(paths: list[Path]) -> None:
    """Print token counts and a decoded sample so the output can be sanity checked."""
    total = 0
    max_token = 0
    for path in paths:
        tokens = np.memmap(path, dtype=np.uint16, mode="r")
        total += len(tokens)
        max_token = max(max_token, int(tokens.max()))
    print(f"Wrote {len(paths)} shards, {total:,} tokens, max token id {max_token}")
    assert max_token < enc.n_vocab, "token id outside GPT-2's vocabulary"

    first_train = next((p for p in paths if "_train_" in p.name), paths[0])
    sample = np.memmap(first_train, dtype=np.uint16, mode="r")[:64]
    print(f"Start of {first_train.name}: {enc.decode(sample.tolist())!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize FineWeb-Edu into uint16 shards.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--num-proc", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()

    dataset = load_dataset(DATASET, name=DATASET_CONFIG, split="train")
    # Only the text column is sent to the workers; the metadata is not needed.
    texts = (row["text"] for row in dataset.select_columns(["text"]))
    paths = write_shards(texts, args.out_dir, args.shard_size, args.num_proc)
    summarize(paths)


if __name__ == "__main__":
    main()
