from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import FlashCausalSelfAttention
from layers.layer_norm import LayerNorm


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        expand_dim = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, expand_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(expand_dim, config.n_embd, bias=config.bias)
        # Writes into the residual stream; GPT2._init_weights scales its init down.
        self.c_proj.is_residual_projection = True
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = LayerNorm(config.n_embd, config.bias)
        # Can swap out with MultiHeadAttention or CausalSelfAttention from layers.attention
        self.heads = FlashCausalSelfAttention(config)
        self.layer_norm2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.heads(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)],
        )
        self.layer_norm = LayerNorm(config.n_embd, config.bias)
        # No bias, as in GPT-2: the output layer is tied to the token embedding, which has
        # no bias either. This also keeps the model convertible to Hugging Face's GPT-2.
        self.linear_layer = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # The the final linear layer shares weights with the token embedding layer
        # https://paperswithcode.com/method/weight-tying
        self.linear_layer.weight = self.token_embedding.weight

        # initialize all weights
        self.apply(self._init_weights)

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Return the number of unique model parameters."""
        parameters = (
            (parameter for parameter in self.parameters() if parameter.requires_grad)
            if trainable_only
            else self.parameters()
        )
        return sum(parameter.numel() for parameter in parameters)

    def model_size_bytes(self, include_buffers: bool = True) -> int:
        """Return the memory required to store the model tensors in their current dtypes."""
        size = sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())
        if include_buffers:
            size += sum(buffer.numel() * buffer.element_size() for buffer in self.buffers())
        return size

    def forward(self, idx: torch.Tensor, targets=None):
        device = idx.device
        b, t = idx.shape
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        token_embeddings = self.token_embedding(idx)  # (b, t, n_embd)
        position_embeddings = self.position_embedding(pos)  # (t, n_embd)
        token_and_position_embeddings = token_embeddings + position_embeddings

        y = self.dropout(token_and_position_embeddings)
        y = self.transformer_blocks(y)
        y = self.layer_norm(y)

        logits = self.linear_layer(y)

        if targets is None:
            return logits, None

        # Flatten (batch, seq) into one classification example per token:
        # logits  (b, t, vocab_size) -> (b * t, vocab_size)
        # targets (b, t)             -> (b * t,)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, num_valid_tokens=None):
        # num_valid_tokens: only sample ids below this. The vocabulary is padded past the
        # tokenizer's size (e.g. 50,304 vs 50,257) for faster matmuls; the padding ids
        # have no text, so they must never be sampled.

        for _ in range(max_new_tokens):
            # crop length to block size
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            # Get the logits for next token
            logits, _ = self(idx_cond)
            # Get the last position in sequence. Scale by temperature
            logits = logits[:, -1, :] / temperature
            if num_valid_tokens is not None:
                logits[:, num_valid_tokens:] = -float("inf")

            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "is_residual_projection", False):
                # GPT-2 scaled init: every block adds two branches (attention and MLP) onto
                # the residual stream, so its variance grows with 2 * n_layer added terms.
                # Scaling these layers' std by 1/sqrt(2 * n_layer) keeps the stream's scale
                # at initialization the same no matter how deep the model is.
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
