# NanoGPT Learning Project

A GPT-2 (124M) implementation based on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) and his [GPT-2 reproduction](https://github.com/karpathy/build-nanogpt), trained from scratch on 10B tokens of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

This repository is a hands-on learning project for brushing up on AI concepts and strengthening my understanding of foundational model fundamentals. The model pieces are written from scratch — the byte pair encoding tokenizer, layer norm, attention, the data loader, and the training loop — so that each piece of the stack is something I've had to reason through rather than import.

**Trained weights:** [panoshalios/nanogpt-124m-fineweb-edu](https://huggingface.co/panoshalios/nanogpt-124m-fineweb-edu) on the Hugging Face Hub.

## Results

| | |
|---|---|
| Parameters | 124,475,904 |
| Training data | FineWeb-Edu sample-10BT, ~10B tokens (19,073 steps × 524,288 tokens) |
| Hardware | 8× A100 80GB, ~1.46M tokens/s, ~2 hours |
| Final validation loss | **3.0541** |

For reference, Karpathy's build-nanogpt run reaches about 3.07 on the same data, and OpenAI's GPT-2 124M checkpoint scores about 3.29 on FineWeb-Edu validation data (it was trained on different data).

![Training and validation loss](https://huggingface.co/panoshalios/nanogpt-124m-fineweb-edu/resolve/main/loss_curve.png)

A sample (top-k 50, temperature 1.0, not cherry-picked):

```
The most important discovery in the history of science was the discovery of the subatomic
particles, called quarks. When scientists understood how subatomic particles act in space,
they were able to explain the existence of quarks, a subatomic particle with a very strange
configuration.
```

Fluent and on topic, and — like any model this size — confidently wrong on details.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy tiktoken datasets tqdm wandb safetensors huggingface_hub matplotlib
```

## Run the trained model

`sample.py` downloads the weights from the Hub (cached after the first run) and generates text. It runs on CUDA, Apple Silicon (MPS), or CPU:

```bash
python sample.py panoshalios/nanogpt-124m-fineweb-edu --prompt "The meaning of life is"
```

Options: `--num-samples`, `--max-new-tokens`, `--temperature`, `--top-k`, `--seed`. The same script also takes a local export folder, a training checkpoint, or a run folder instead of a repo id.

From Python:

```python
import torch
from sample import generate_text, load_weights

model, _ = load_weights("panoshalios/nanogpt-124m-fineweb-edu", torch.device("cpu"))
print(generate_text(model, "The meaning of life is", num_samples=1, max_new_tokens=50)[0])
```

The weights are a `model.safetensors` for the `GPT2` class in this repository, not the Hugging Face `transformers` format. Text is tokenized with tiktoken's `gpt2` encoding.

## What's implemented

- **GPT-2 model** (`model/gpt2.py`) — learned token and position embeddings, pre-norm transformer blocks, a 4x MLP with GELU, weight tying between the token embedding and the output head (no output bias, as in GPT-2), GPT-2's scaled initialization of the residual projections, and a `generate` method with temperature and top-k sampling.
- **Attention** (`layers/attention.py`) — four variants to compare: `SimpleSingleHeadAttention`, a `MultiHeadAttention` that concatenates independent heads, a `CausalSelfAttention` with the batched QKV projection and explicit masked softmax, and `FlashCausalSelfAttention`, the same layer with the attention step replaced by PyTorch's fused `scaled_dot_product_attention`. The model uses the flash version.
- **Layer norm** (`layers/layer_norm.py`) — thin module with an optional bias.
- **Byte pair encoding tokenizer** (`tokenizer/gpt2_tokenizer.py`) — learns merges from raw UTF-8 bytes, with save/load to JSON. Two encoding paths exist on purpose: `tokenize` applies merges in learned order, `tokenize_2` repeatedly picks the earliest learned merge present in the sequence. It is too slow for 10B tokens, so the FineWeb-Edu run uses tiktoken's GPT-2 encoding.
- **Data preparation** (`input/fineweb_edu/prepare.py`) — downloads FineWeb-Edu, tokenizes it in parallel with an end-of-text token before every document, and writes 100M-token `uint16` shards (the first one held out for validation).
- **Data loader** (`dataloader/DataLoader.py`) — memory-maps the shards one at a time and cuts them into non-overlapping sequences. Every pass visits the shards in a new order with a new chunk offset and shuffle; each GPU reads a disjoint share, and its position can be saved and restored for resuming.
- **Training loop** (`train.py`, `train_utils.py`) — distributed data parallel over any number of GPUs with gradient accumulation to a fixed 524,288-token batch, bfloat16 autocast, TF32, `torch.compile`, fused AdamW with weight decay on 2D weights only, gradient clipping, and GPT-3's warmup + cosine learning rate schedule. It evaluates validation loss on fixed held-out tokens, logs to a JSON Lines file (and optionally Weights & Biases), and saves checkpoints it can resume from exactly.
- **Export** (`export.py`) — turns a checkpoint into `model.safetensors` + `config.json` with a generated model card and loss curve, and uploads it to the Hugging Face Hub.

## Layout

```
model/       GPT-2 model and config
layers/      attention and layer norm
tokenizer/   byte pair encoding tokenizer
dataloader/  sharded, DDP-aware batching over uint16 token files
input/       datasets and their prepare scripts (FineWeb-Edu, Shakespeare)
train.py     training loop; train_utils.py has its helpers
sample.py    text generation from the Hub, an export folder, or a checkpoint
export.py    checkpoint -> safetensors + model card, upload to the Hub
runs/        one folder per training run: log.jsonl and checkpoints (gitignored)
output/      trained tokenizers
```

## Training it yourself

**1. Prepare the data** (~20 GB of shards, plus the Hugging Face download cache):

```bash
python input/fineweb_edu/prepare.py
```

**2. Train.** On one GPU, or on N GPUs with `torchrun`:

```bash
python train.py
```

```bash
torchrun --standalone --nproc_per_node=8 train.py --wandb
```

Each run writes to `runs/<start time>/`: `log.jsonl` with every training step and validation result, plus a checkpoint every 1,000 steps. `--wandb` also logs to Weights & Biases (run `wandb login` first). Resume an interrupted run, on the same number of GPUs, with:

```bash
torchrun --standalone --nproc_per_node=8 train.py --resume runs/<run>
```

Hyperparameters are constants at the top of `train.py`. `MICRO_BATCH_SIZE` only trades memory for speed — gradient accumulation keeps the batch at `TOTAL_BATCH_SIZE` — so set it to what fits: 64 on 80 GB GPUs (used for the run above); try 32 on 40 GB and drop to 16 if it runs out of memory.

**3. Sample and export:**

```bash
python sample.py runs/<run>
```

```bash
python export.py runs/<run> --repo <user>/<name> --github-url https://github.com/<user>/nanogpt
```

`export.py` without `--repo` only writes `runs/<run>/export/` for review; uploading needs `hf auth login` with a write token.

## Shakespeare tokenizer experiment

The from-scratch tokenizer is trained and tested on [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), expected at `input/shakespeare/input.txt` (text files under `input/` are gitignored). Train a 1,024-token vocabulary, which writes `output/shakespeare_tokenizer.json`:

```bash
python tokenizer_train.py
```

Then tokenize the text into `input/shakespeare/train.bin` and `val.bin` with a 90/10 split (run it as a module from the repository root, so the `tokenizer` package is importable):

```bash
python -m input.shakespeare.prepare
```

`train.py` is set up for the FineWeb-Edu shards; training on these files means pointing `TRAIN_SHARDS` / `VAL_SHARDS` at them, setting `VOCAB_SIZE = 1024`, and shrinking the model and step counts.
