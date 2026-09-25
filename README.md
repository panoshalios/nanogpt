# NanoGPT Learning Project

A small GPT implementation based on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).

This repository is a hands-on learning project for brushing up on AI concepts and strengthening my understanding of foundational model fundamentals. Everything is written from scratch — the byte pair encoding tokenizer, layer norm, attention, and the training loop — so that each piece of the stack is something I've had to reason through rather than import.

## What's implemented

- **Byte pair encoding tokenizer** (`tokenizer/gpt2_tokenizer.py`) — learns merges from raw UTF-8 bytes, with save/load to JSON. Two encoding paths exist on purpose: `tokenize` applies merges in learned order, `tokenize_2` repeatedly picks the earliest learned merge present in the sequence.
- **Attention** (`layers/attention.py`) — three variants to compare: `SimpleSingleHeadAttention`, a `MultiHeadAttention` that concatenates independent heads, and a `CausalSelfAttention` with the batched QKV projection that real implementations use.
- **Layer norm** (`layers/layer_norm.py`) — thin module with an optional bias, so bias-free configs are possible.
- **GPT-2 model** (`model/gpt2.py`) — learned token and position embeddings, pre-norm transformer blocks, a 4x MLP with GELU, weight tying between the token embedding and the output head, and a `generate` method with temperature and top-k sampling.
- **Data loader** (`dataloader/DataLoader.py`) — memory-maps a `uint16` token file and yields fixed-length `(input, target)` batches with optional shuffling and pinned memory.
- **Training loop** (`train.py`) — device selection across CUDA, MPS, and CPU, with bfloat16 autocast when the backend supports it.

## Layout

```
model/       GPT-2 model and config
layers/      attention and layer norm
tokenizer/   byte pair encoding tokenizer
dataloader/  batching over a binary token file
input/       datasets and their prepare scripts
output/      trained tokenizers
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy tiktoken datasets tqdm
```

## Running it

The Shakespeare dataset is expected at `input/shakespeare/input.txt` (the plain text of [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)). Text files under `input/` are gitignored, so download it first.

Train the tokenizer, which writes `output/shakespeare_tokenizer.json`:

```bash
python tokenizer_train.py
```

Tokenize the dataset into `train.bin` and `val.bin` with a 90/10 split:

```bash
python input/shakespeare/prepare.py
```

Train the model:

```bash
python train.py
```

Hyperparameters are constants at the top of `train.py` and fields on `GPT2Config`. Note that `VOCAB_SIZE` in `train.py` must match the `max_vocab_size` the tokenizer was trained with, currently 400.
