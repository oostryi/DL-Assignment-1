"""Shared single-tensor input encoding for tabular neural networks."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def embedding_dim(cardinality: int) -> int:
    return min(32, max(2, round(1.6 * cardinality**0.56)))


class TabularEmbedding(nn.Module):
    """Decode one [numeric | categorical codes] matrix and embed categories."""

    def __init__(self, num_numeric: int, cardinalities: Sequence[int]):
        super().__init__()
        self.num_numeric = num_numeric
        self.cardinalities = list(cardinalities)
        dims = [embedding_dim(n) for n in cardinalities]
        self.embeddings = nn.ModuleList(
            nn.Embedding(n, width, padding_idx=0) for n, width in zip(cardinalities, dims)
        )
        self.output_dim = num_numeric + sum(dims)

    def forward(self, x: Tensor) -> Tensor:
        expected = self.num_numeric + len(self.embeddings)
        if x.ndim != 2 or x.shape[1] != expected:
            raise ValueError(f"Expected [batch, {expected}] input, got {tuple(x.shape)}")
        numeric = x[:, : self.num_numeric]
        categorical = x[:, self.num_numeric :].long()
        embedded = [embedding(categorical[:, i]) for i, embedding in enumerate(self.embeddings)]
        return torch.cat([numeric, *embedded], dim=1)
