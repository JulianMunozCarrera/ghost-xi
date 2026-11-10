"""
The same ghost model, written in plain NumPy.

Training needs PyTorch. Running the trained model does not. This file
loads the saved weights and does the same math step by step, so the rest
of the project (and anyone who clones it) can make ghosts with only
NumPy installed.
"""

import os

import numpy as np

HEADS = 4

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "models", "ghost.npz")


def _layer_norm(x, weight, bias):
    """Shift and scale each token so its numbers have mean 0 and spread 1."""
    mean = x.mean(axis=-1, keepdims=True)
    spread = np.sqrt(((x - mean) ** 2).mean(axis=-1, keepdims=True) + 1e-5)
    return (x - mean) / spread * weight + bias


class GhostModel:
    def __init__(self, path=DEFAULT_WEIGHTS):
        self.w = dict(np.load(path))
        self.layers = 1 + max(int(name.split(".")[1]) for name in self.w if name.startswith("layers."))

    def _linear(self, x, name):
        return x @ self.w[name + ".weight"].T + self.w[name + ".bias"]

    def run(self, tokens, present):
        """
        tokens:  (23, 21) or (batch, 23, 21)
        present: (23,) or (batch, 23) True/False
        Returns how far each token moved, in units of 10 m. Same leading shape, last size 2.
        """
        single = tokens.ndim == 2
        x = tokens[None] if single else tokens
        present = present[None] if single else present
        batch, count, _ = x.shape

        x = self._linear(x.astype(np.float32), "read")
        width = x.shape[-1]
        per_head = width // HEADS

        for i in range(self.layers):
            name = f"layers.{i}."
            h = _layer_norm(x, self.w[name + "norm1.weight"], self.w[name + "norm1.bias"])
            q = self._linear(h, name + "query").reshape(batch, count, HEADS, per_head).transpose(0, 2, 1, 3)
            k = self._linear(h, name + "key").reshape(batch, count, HEADS, per_head).transpose(0, 2, 1, 3)
            v = self._linear(h, name + "value").reshape(batch, count, HEADS, per_head).transpose(0, 2, 1, 3)
            scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(per_head)
            scores = np.where(present[:, None, None, :], scores, -1e9)
            scores = scores - scores.max(axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights = weights / weights.sum(axis=-1, keepdims=True)
            attended = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, count, width)
            x = x + self._linear(attended, name + "mix")

            h = _layer_norm(x, self.w[name + "norm2.weight"], self.w[name + "norm2.bias"])
            x = x + self._linear(np.maximum(self._linear(h, name + "think1"), 0), name + "think2")

        x = _layer_norm(x, self.w["norm.weight"], self.w["norm.bias"])
        out = self._linear(x, "answer")
        return out[0] if single else out
