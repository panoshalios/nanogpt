# Packages a trained checkpoint for the Hugging Face Hub: weights, config, loss curve,
# training log and a model card. Optionally uploads the folder.
#
#   python export.py runs/<run>                          write runs/<run>/export/ to review
#   python export.py runs/<run> --repo <user>/<name>     ...and upload it (private)
#
# Uploading needs a Hugging Face token with write access: run `hf auth login` first.
import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to a file; no display needed on a server
import matplotlib.pyplot as plt  # noqa: E402
from safetensors.torch import save_model  # noqa: E402

from sample import generate_text, load_model  # noqa: E402
from train_utils import find_checkpoint, get_device  # noqa: E402

DEFAULT_PROMPTS = [
    "Hello, I'm a language model,",
    "The most important discovery in the history of science was",
    "Photosynthesis is the process by which",
]


def read_log(log_path: Path, up_to_step: int) -> tuple[dict | None, dict, dict]:
    """Return (run config, {step: train_loss}, {step: val_loss}) from a log.jsonl.

    After a resume, steps between the checkpoint and the interruption appear twice; later
    records overwrite earlier ones, which keeps the values from the run that continued.
    """
    config, train, val = None, {}, {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "config" in record and config is None:
            config = record["config"]
        step = record.get("step")
        if step is None or step > up_to_step:
            continue
        if "train_loss" in record:
            train[step] = record["train_loss"]
        if "val_loss" in record:
            val[step] = record["val_loss"]
    return config, dict(sorted(train.items())), dict(sorted(val.items()))


def plot_losses(train: dict, val: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(list(train), list(train.values()), lw=0.8, alpha=0.7, label="train loss")
    ax.plot(list(val), list(val.values()), marker="o", ms=3, label="validation loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def model_card(
    repo: str | None,
    github_url: str | None,
    step: int,
    num_params: int,
    model_config: dict,
    run_config: dict | None,
    val: dict,
    samples: list[str],
    has_plot: bool,
) -> str:
    title = repo.split("/")[-1] if repo else f"GPT-2 {num_params / 1e6:.0f}M trained from scratch"
    code = f"[the training code]({github_url})" if github_url else "the training code"
    repo_id = repo or "<user>/<repo>"

    lines = [
        "---",
        "language: en",
        "datasets:",
        "- HuggingFaceFW/fineweb-edu",
        "tags:",
        "- gpt2",
        "- pytorch",
        "- from-scratch",
        "---",
        "",
        f"# {title}",
        "",
        f"A {num_params / 1e6:.0f}M parameter GPT-2 style language model, implemented and "
        f"trained from scratch. See {code}.",
        "",
        "## Results",
        "",
        "| | |",
        "|---|---|",
        f"| Optimizer steps | {step:,} |",
    ]
    if run_config is not None:
        tokens = step * run_config["total_batch_size"]
        tokens_text = f"{tokens / 1e9:.2f}B" if tokens >= 1e9 else f"{tokens:,}"
        lines.append(f"| Tokens trained on | {tokens_text} |")
    if val:
        final_step, final_loss = list(val.items())[-1]
        lines.append(f"| Validation loss (step {final_step:,}) | {final_loss:.4f} |")
    if has_plot:
        lines += ["", "![Training and validation loss](loss_curve.png)"]

    lines += ["", "## Samples", "", "Top-k 50, temperature 1.0, not cherry-picked.", ""]
    for text in samples:
        lines += ["```", text.strip(), "```", ""]

    lines += [
        "## Model",
        "",
        "| | |",
        "|---|---|",
        f"| Parameters | {num_params:,} |",
        f"| Layers | {model_config['n_layer']} |",
        f"| Heads | {model_config['n_head']} |",
        f"| Embedding size | {model_config['n_embd']} |",
        f"| Context length | {model_config['block_size']} |",
        f"| Vocabulary | {model_config['vocab_size']:,} (GPT-2's 50,257, padded) |",
        "",
        "Pre-norm transformer blocks, learned position embeddings, GELU MLP, flash "
        "attention, output layer tied to the token embedding.",
    ]

    if run_config is not None:
        lines += [
            "",
            "## Training",
            "",
            "| | |",
            "|---|---|",
            "| Data | FineWeb-Edu (sample-10BT), GPT-2 tokenizer (tiktoken) |",
            f"| Batch size | {run_config['total_batch_size']:,} tokens per step |",
            f"| Learning rate | {run_config['max_lr']} peak, linear warmup over "
            f"{run_config['warmup_steps']} steps, cosine decay to {run_config['min_lr']:.1e} |",
            f"| Optimizer | AdamW (0.9, 0.95), weight decay {run_config['weight_decay']} on "
            "2D weights, gradient clipping at 1.0 |",
            f"| Hardware | {run_config['world_size']} GPU(s), bfloat16 mixed precision, DDP |",
        ]

    lines += [
        "",
        "## Usage",
        "",
        f"The weights load into the `GPT2` class from {code}; they are not in the "
        "Hugging Face `transformers` format. Text is tokenized with tiktoken's `gpt2` "
        "encoding.",
        "",
        "```python",
        "import json",
        "from huggingface_hub import hf_hub_download",
        "from safetensors.torch import load_model",
        "from model.gpt2 import GPT2, GPT2Config",
        "",
        f'repo = "{repo_id}"',
        'config = json.load(open(hf_hub_download(repo, "config.json")))',
        "model = GPT2(GPT2Config(**config))",
        'load_model(model, hf_hub_download(repo, "model.safetensors"))',
        "```",
        "",
        "`log.jsonl` has the full training log: one JSON record per step.",
        "",
        "## Limitations",
        "",
        "A small base model trained for learning purposes. It continues text; it does "
        "not follow instructions, and it produces fluent but often wrong or made-up "
        "statements.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint for the Hugging Face Hub.")
    parser.add_argument("checkpoint", type=Path, help="checkpoint file, or run folder (latest)")
    parser.add_argument("--out-dir", type=Path, help="default: <run folder>/export")
    parser.add_argument("--repo", help="Hugging Face repo id <user>/<name>; uploads if given")
    parser.add_argument("--public", action="store_true", help="create the repo as public")
    parser.add_argument("--github-url", help="link to the training code in the model card")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    checkpoint_file = find_checkpoint(args.checkpoint)
    run_dir = checkpoint_file.parent
    out_dir = args.out_dir or run_dir / "export"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, checkpoint = load_model(checkpoint_file, device)
    step = checkpoint["step"]
    print(f"Loaded {checkpoint_file} (step {step})")

    # Weights only, as safetensors: plain tensor data, unlike a pickled .pt file.
    # save_model (not save_file) handles the output layer sharing its weight with the
    # token embedding: the shared tensor is stored once and re-tied by load_model.
    save_model(model, str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text(json.dumps(checkpoint["model_config"], indent=2))

    run_config, train, val = None, {}, {}
    log_path = run_dir / "log.jsonl"
    if log_path.is_file():
        run_config, train, val = read_log(log_path, up_to_step=step)
        shutil.copy(log_path, out_dir / "log.jsonl")
    has_plot = bool(train)
    if has_plot:
        plot_losses(train, val, out_dir / "loss_curve.png")

    samples = [
        generate_text(model, prompt, num_samples=1, max_new_tokens=args.max_new_tokens)[0]
        for prompt in DEFAULT_PROMPTS
    ]
    card = model_card(
        args.repo,
        args.github_url,
        step,
        model.num_parameters(),
        checkpoint["model_config"],
        run_config,
        val,
        samples,
        has_plot,
    )
    (out_dir / "README.md").write_text(card, encoding="utf-8")

    print(f"Wrote {out_dir}:")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.name:20s} {path.stat().st_size / 1e6:8.1f} MB")
    print("Review README.md; add a license to its front matter if you want one.")

    if args.repo:
        from huggingface_hub import HfApi

        api = HfApi()
        # private only applies when the repo is created; an existing repo keeps its
        # visibility (change it in the repo settings on huggingface.co).
        api.create_repo(args.repo, private=not args.public, exist_ok=True)
        api.upload_folder(repo_id=args.repo, folder_path=out_dir)
        print(f"Uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
