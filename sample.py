# Generates text from a training checkpoint.
#
#   python sample.py runs/<run>                  latest checkpoint in the run folder
#   python sample.py runs/<run>/checkpoint_019073.pt --prompt "The meaning of life is"
import argparse
from pathlib import Path

import tiktoken
import torch

from model.gpt2 import GPT2, GPT2Config
from train_utils import find_checkpoint, get_device, supports_bfloat16


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text from a GPT-2 checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="checkpoint file, or run folder (latest)")
    parser.add_argument("--prompt", default="Hello, I'm a language model,")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    # Below 1.0 makes the output more predictable, above 1.0 more random.
    parser.add_argument("--temperature", type=float, default=1.0)
    # Sample only among the top_k most likely tokens; cuts off the unlikely tail.
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    checkpoint_file = find_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_file, map_location=device)

    # The checkpoint stores the config it was trained with, so the model is rebuilt at
    # exactly the right shapes before loading the weights.
    model = GPT2(GPT2Config(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()  # turns off dropout
    val_loss = checkpoint.get("val_loss")
    val_text = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
    print(f"Loaded {checkpoint_file} (step {checkpoint['step']}{val_text}) on {device}")

    enc = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(args.prompt)
    # One row per sample; every row starts with the same prompt.
    idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    idx = idx.unsqueeze(0).repeat(args.num_samples, 1)

    torch.manual_seed(args.seed)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=supports_bfloat16(device)
    ):
        out = model.generate(
            idx,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            num_valid_tokens=enc.n_vocab,  # never sample the padding ids
        )

    for i, row in enumerate(out.tolist()):
        new_tokens = row[len(prompt_tokens) :]
        # <|endoftext|> starts an unrelated document in the training data, so stop there.
        if enc.eot_token in new_tokens:
            new_tokens = new_tokens[: new_tokens.index(enc.eot_token)]
        print(f"\n--- sample {i + 1} ---")
        print(args.prompt + enc.decode(new_tokens))


if __name__ == "__main__":
    main()
