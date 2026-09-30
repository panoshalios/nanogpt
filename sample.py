# Generates text from trained weights: the published model on the Hugging Face Hub, a
# local export folder, or a training checkpoint.
#
#   python sample.py panoshalios/nanogpt-124m-fineweb-edu   weights from the Hub
#   python sample.py runs/<run>/export                      folder written by export.py
#   python sample.py runs/<run>                             latest checkpoint of a run
#   python sample.py runs/<run>/checkpoint_019073.pt --prompt "The meaning of life is"
import argparse
import json
from pathlib import Path

import tiktoken
import torch
from safetensors.torch import load_model as load_safetensors

from model.gpt2 import GPT2, GPT2Config
from train_utils import find_checkpoint, get_device, supports_bfloat16


def load_model(checkpoint_file: Path, device: torch.device) -> tuple[GPT2, dict]:
    """Rebuild the model from a checkpoint; returns the model and the checkpoint dict."""
    checkpoint = torch.load(checkpoint_file, map_location=device)
    # The checkpoint stores the config it was trained with, so the model is rebuilt at
    # exactly the right shapes before loading the weights.
    model = GPT2(GPT2Config(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()  # turns off dropout
    return model, checkpoint


def load_weights(source: str, device: torch.device) -> tuple[GPT2, str]:
    """Load a model from a Hub repo id, an export folder, a checkpoint, or a run folder.

    Returns the model and a short description of what was loaded.
    """
    path = Path(source)
    if path.exists() and not (path / "model.safetensors").is_file():
        # A training checkpoint, or a run folder (use its latest checkpoint).
        checkpoint_file = find_checkpoint(path)
        model, checkpoint = load_model(checkpoint_file, device)
        val_loss = checkpoint.get("val_loss")
        val_text = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
        return model, f"{checkpoint_file} (step {checkpoint['step']}{val_text})"

    if path.exists():
        # A folder written by export.py: model.safetensors + config.json.
        config_file, weights_file = path / "config.json", path / "model.safetensors"
    else:
        # Otherwise a Hugging Face repo id like "user/name". Files are downloaded once
        # and cached in ~/.cache/huggingface.
        from huggingface_hub import hf_hub_download

        config_file = hf_hub_download(source, "config.json")
        weights_file = hf_hub_download(source, "model.safetensors")

    config = json.loads(Path(config_file).read_text())
    model = GPT2(GPT2Config(**config))
    # load_model (not load_file) re-ties the output layer to the token embedding, which
    # export.py stored only once.
    load_safetensors(model, str(weights_file))
    model.to(device).eval()  # eval() turns off dropout
    return model, source


def generate_text(
    model: GPT2,
    prompt: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = 50,
    seed: int = 42,
) -> list[str]:
    """Continue prompt num_samples times; each result includes the prompt."""
    device = next(model.parameters()).device
    enc = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(prompt)
    # One row per sample; every row starts with the same prompt.
    idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    idx = idx.unsqueeze(0).repeat(num_samples, 1)

    torch.manual_seed(seed)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=supports_bfloat16(device)
    ):
        out = model.generate(
            idx,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            num_valid_tokens=enc.n_vocab,  # never sample the padding ids
        )

    texts = []
    for row in out.tolist():
        new_tokens = row[len(prompt_tokens) :]
        # <|endoftext|> starts an unrelated document in the training data, so stop there.
        if enc.eot_token in new_tokens:
            new_tokens = new_tokens[: new_tokens.index(enc.eot_token)]
        texts.append(prompt + enc.decode(new_tokens))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text from trained GPT-2 weights.")
    parser.add_argument(
        "source",
        help="Hugging Face repo id, export folder, checkpoint file, or run folder (latest)",
    )
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
    model, description = load_weights(args.source, device)
    print(f"Loaded {description} on {device}")

    texts = generate_text(
        model,
        args.prompt,
        args.num_samples,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    for i, text in enumerate(texts):
        print(f"\n--- sample {i + 1} ---")
        print(text)


if __name__ == "__main__":
    main()
