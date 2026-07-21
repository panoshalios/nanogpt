import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from model.gpt2 import GPT2Config


class SimpleSingleHeadAttention(nn.Module):
    def __init__(self, config: "GPT2Config"):
        super().__init__()
        self.head_size = config.n_embd // config.n_head
        self.key_att = nn.Linear(config.n_embd, self.head_size, bias=config.bias)
        self.query_att = nn.Linear(config.n_embd, self.head_size, bias=config.bias)
        self.value_att = nn.Linear(config.n_embd, self.head_size, bias=config.bias)
        self.proj = nn.Linear(self.head_size, self.head_size, bias=config.bias)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        key = self.key_att(x)
        query = self.query_att(x)
        value = self.value_att(x)
        attn = (query @ key.transpose(-2, -1)) * (1 / math.sqrt(self.head_size))  # (B, T, T)
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ value
        y = self.resid_dropout(self.proj(y))
        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, config: "GPT2Config"):
        super().__init__()
        self.heads = nn.ModuleList(
            [SimpleSingleHeadAttention(config) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        y = torch.cat([head(x) for head in self.heads], dim=-1)
        y = self.proj(y)
        return y


class CausalSelfAttention(nn.Module):
    def __init__(self, config: "GPT2Config"):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value
        # Where did the 3 * n_embd come from?
        # It's because we have three different linear layers for the key, query, and value
        # The key, query, and value are all projected into the same dimension
        # So we need three different linear layers to project them into the same dimension
        # The key, query, and value are all projected into the same dimension
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # TODO replace with flash attention
        # What does .tril do?
        # It creates a lower triangular matrix with ones on the diagonal

        # What does .view do? How does it reshape the tensor and why do we need to do this?
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()

        query_key_value = self.c_attn(x)
        query, key, value = query_key_value.split(self.n_embd, dim=2)
        head_size = C // self.n_head
        q = query.view(B, T, self.n_head, head_size).transpose(
            1, 2
        )  # before transpose: (B, T, n_head, head_size), after transpose: (B, n_head, T, head_size)
        k = key.view(B, T, self.n_head, head_size).transpose(
            1, 2
        )  # before transpose: (B, T, n_head, head_size), after transpose: (B, n_head, T, head_size)
        v = value.view(B, T, self.n_head, head_size).transpose(
            1, 2
        )  # before transpose: (B, T, n_head, head_size), after transpose: (B, n_head, T, head_size)

        attn = (q @ k.transpose(-2, -1)) * (
            1 / math.sqrt(head_size)
        )  # (B, n_head, T, head_size) @ (B, n_head, head_size, T) = (B, n_head, T, T)
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # (B, n_head, T, T)
        attn = F.softmax(attn, dim=-1)  # (B, n_head, T, T)

        # output of attention
        y = attn @ v

        # transpose back to (B, T, n_head, head_size). Contiguous is used to make the tensor contiguous in memory.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))

        return y
