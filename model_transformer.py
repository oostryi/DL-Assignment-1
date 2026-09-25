from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn


class FeatureTokenizer(nn.Module):

    def __init__(
        self,
        num_numeric: int,
        cardinalities: Sequence[int],
        token_dim: int,
        numeric_embedding: str = "linear",
        numeric_bin_edges: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        super().__init__()
        self.num_numeric = num_numeric
        if numeric_embedding not in {"linear", "ple"}:
            raise ValueError(f"Unknown numerical embedding: {numeric_embedding}")
        self.numeric_embedding = numeric_embedding
        if numeric_embedding == "ple":
            if numeric_bin_edges is None:
                raise ValueError("PLE needs fold-local bin edges")
            edges, mask = numeric_bin_edges
            if edges.ndim != 2 or mask.shape != (edges.shape[0], edges.shape[1] - 1):
                raise ValueError("Incompatible PLE edges and mask")
            if edges.shape[0] > num_numeric:
                raise ValueError("More PLE columns than numerical inputs")
            self.num_piecewise = edges.shape[0]
            self.register_buffer("bin_left", torch.as_tensor(edges[:, :-1].copy()))
            self.register_buffer("bin_right", torch.as_tensor(edges[:, 1:].copy()))
            self.register_buffer("bin_mask", torch.as_tensor(mask.copy()))
            self.piecewise_weight = nn.Parameter(torch.randn(self.num_piecewise, mask.shape[1], token_dim) * 0.02)
            linear_features = num_numeric - self.num_piecewise
        else:
            self.num_piecewise = 0
            linear_features = num_numeric
        self.numeric_weight = nn.Parameter(torch.randn(linear_features, token_dim) * 0.02)
        self.numeric_bias = nn.Parameter(torch.zeros(num_numeric, token_dim))
        self.embeddings = nn.ModuleList(nn.Embedding(n, token_dim, padding_idx=0) for n in cardinalities)

    def forward(self, x: Tensor) -> Tensor:
        expected = self.num_numeric + len(self.embeddings)
        if x.ndim != 2 or x.shape[1] != expected:
            raise ValueError(f"Expected [batch, {expected}] input, got {tuple(x.shape)}")
        if self.numeric_embedding == "ple":
            values = x[:, : self.num_piecewise].unsqueeze(-1)
            widths = (self.bin_right - self.bin_left).clamp_min(1e-6)
            encoding = ((values - self.bin_left) / widths).clamp(0.0, 1.0) * self.bin_mask
            piecewise = torch.einsum("bft,ftd->bfd", encoding, self.piecewise_weight)
            piecewise = piecewise + self.numeric_bias[: self.num_piecewise]
            remainder = x[:, self.num_piecewise : self.num_numeric].unsqueeze(-1)
            linear = remainder * self.numeric_weight + self.numeric_bias[self.num_piecewise :]
            numeric = torch.cat([piecewise, linear], dim=1)
        else:
            numeric = x[:, : self.num_numeric].unsqueeze(-1) * self.numeric_weight + self.numeric_bias
        categories = x[:, self.num_numeric :].long()
        categorical = torch.stack([emb(categories[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        return torch.cat([numeric, categorical], dim=1)


class TabularTransformer(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cardinalities: Sequence[int],
        hidden_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.2,
        numeric_embedding: str = "linear",
        numeric_bin_edges: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        super().__init__()
        if hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by 4")
        self.tokenizer = FeatureTokenizer(
            num_numeric, cardinalities, hidden_dim, numeric_embedding, numeric_bin_edges
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(len(x), -1, -1)
        states = self.transformer(torch.cat([cls, tokens], dim=1))
        return self.head(states[:, 0]).squeeze(1)
