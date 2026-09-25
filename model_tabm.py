from collections.abc import Sequence
import warnings

import numpy as np
import torch
from torch import Tensor, nn


class TabMClassifier(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cardinalities: Sequence[int],
        hidden_dim: int,
        depth: int,
        dropout: float,
        k: int,
        numeric_embedding: str,
        numeric_bin_edges: tuple[np.ndarray, np.ndarray] | None,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        from tabm import TabM

        num_embeddings = None
        if numeric_embedding == "ple":
            if numeric_bin_edges is None:
                raise ValueError("PLE requires fold-local numerical bins")
            from rtdl_num_embeddings import PiecewiseLinearEmbeddings

            edges, mask = numeric_bin_edges
            bins = []
            for feature_edges, feature_mask in zip(edges, mask):
                count = int(feature_mask.sum())
                bins.append(torch.as_tensor(feature_edges[: count + 1].copy()))

            bins.extend(
                torch.tensor([-0.5, 0.5, 1.5])
                for _ in range(num_numeric - len(edges))
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=r"The .* feature has just two bin edges",
                    category=UserWarning,
                )
                num_embeddings = PiecewiseLinearEmbeddings(
                    bins, d_embedding=embedding_dim, activation=False, version="B"
                )
        elif numeric_embedding != "linear":
            raise ValueError(f"Unknown numerical embedding: {numeric_embedding}")

        self.num_numeric = num_numeric
        self.num_categorical = len(cardinalities)
        self.tabm = TabM.make(
            n_num_features=num_numeric,
            cat_cardinalities=list(cardinalities),
            num_embeddings=num_embeddings,
            d_out=1,
            n_blocks=depth,
            d_block=hidden_dim,
            dropout=dropout,
            k=k,
        )

    def forward(self, x: Tensor) -> Tensor:
        expected = self.num_numeric + self.num_categorical
        if x.ndim != 2 or x.shape[1] != expected:
            raise ValueError(f"Expected [batch, {expected}] input, got {tuple(x.shape)}")
        x_num = x[:, : self.num_numeric] if self.num_numeric else None
        x_cat = x[:, self.num_numeric :].long() if self.num_categorical else None
        return self.tabm(x_num, x_cat).squeeze(-1)
