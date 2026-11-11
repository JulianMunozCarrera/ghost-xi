"""
The ghost model, written in PyTorch. This version is used for training.

It is a small transformer. In plain words:

  1. Every token (the ball, each attacker, each defender) starts as 21
     numbers and gets turned into 64 numbers.
  2. Three times in a row, every token "looks at" all the other tokens
     and updates itself. This is called attention. It is how a hidden
     defender finds out where the ball went and what his teammates did.
  3. At the end, each token gives 2 numbers: how far it moved in x and y.
     We only use that answer for the hidden defenders.

Tokens have no fixed order, so it does not matter which player sits in
which slot. Empty slots are ignored.

After training, the weights are saved to a file and ghostxi/model_numpy.py
runs the same math without PyTorch.
"""

import math

import torch
import torch.nn as nn

from .tokens import FEATURES

HEADS = 4


class Block(nn.Module):
    """One round of "look at everyone else, then think about it"."""

    def __init__(self, width):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.mix = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.think1 = nn.Linear(width, 2 * width)
        self.think2 = nn.Linear(2 * width, width)

    def forward(self, x, present):
        batch, count, width = x.shape
        per_head = width // HEADS

        def split(t):   # (batch, tokens, width) -> (batch, heads, tokens, per_head)
            return t.reshape(batch, count, HEADS, per_head).transpose(1, 2)

        h = self.norm1(x)
        q, k, v = split(self.query(h)), split(self.key(h)), split(self.value(h))
        # scores[b, head, i, j] = how much token i wants to look at token j
        scores = q @ k.transpose(-1, -2) / math.sqrt(per_head)
        scores = scores.masked_fill(~present[:, None, None, :], -1e9)   # nobody looks at an empty slot
        weights = torch.softmax(scores, dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(batch, count, width)
        x = x + self.mix(attended)

        h = self.norm2(x)
        return x + self.think2(torch.relu(self.think1(h)))


class GhostNet(nn.Module):
    def __init__(self, width=64, layers=3):
        super().__init__()
        self.read = nn.Linear(FEATURES, width)
        self.layers = nn.ModuleList([Block(width) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.answer = nn.Linear(width, 2)

    def forward(self, tokens, present):
        """
        tokens:  (batch, 23, 21)
        present: (batch, 23) True where the slot holds a real token
        Returns (batch, 23, 2): how far each token moved, in units of 10 m.
        """
        x = self.read(tokens)
        for layer in self.layers:
            x = layer(x, present)
        return self.answer(self.norm(x))
