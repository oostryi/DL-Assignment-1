from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from model_inputs import TabularEmbedding


TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"


@dataclass
class Config:
    seed: int = 42
    folds: int = 5
    epochs: int = 30
    patience: int = 5
    batch_size: int = 2048
    hidden_dim: int = 192
    depth: int = 3
    cross_layers: int = 2
    dropout: float = 0.20
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    model: str = "deep_cross"
    numeric_missing_indicators: bool = False
    numeric_embedding: str = "linear"
    numeric_bins: int = 16
    tabm_k: int = 32
    tabm_embedding_dim: int = 16
    gradient_clip: float = 5.0
    num_workers: int = 0
    max_train_rows: int | None = None
    train_path: str = "data/clean/train_eng.parquet"
    test_path: str = "data/clean/test_eng.parquet"
    output_dir: str = "artifacts/deep_learning_single"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TabularPreprocessor:
    """Fold-local preprocessing to prevent validation leakage.

    Numeric values are median-imputed, clipped using training-set quantiles, and
    standardised.  Categorical values receive a fold-local vocabulary with index 0
    reserved for missing/unknown values.
    """

    def __init__(
        self,
        categorical_columns: Sequence[str],
        numeric_columns: Sequence[str],
        add_numeric_missing_indicators: bool = False,
        numeric_bins: int = 0,
    ):
        self.categorical_columns = list(categorical_columns)
        self.numeric_columns = list(numeric_columns)
        self.add_numeric_missing_indicators = add_numeric_missing_indicators
        self.numeric_bins = numeric_bins
        self.numeric_bin_edges: np.ndarray | None = None
        self.numeric_bin_mask: np.ndarray | None = None
        self.missing_indicator_columns: list[str] = []
        self.medians: pd.Series | None = None
        self.means: pd.Series | None = None
        self.stds: pd.Series | None = None
        self.lower: pd.Series | None = None
        self.upper: pd.Series | None = None
        self.vocabularies: dict[str, dict[str, int]] = {}

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        numeric = frame[self.numeric_columns].replace([np.inf, -np.inf], np.nan).astype("float64")
        self.missing_indicator_columns = (
            numeric.columns[numeric.isna().any()].tolist()
            if self.add_numeric_missing_indicators else []
        )
        self.medians = numeric.median().fillna(0.0)
        numeric = numeric.fillna(self.medians)
        self.lower = numeric.quantile(0.005)
        self.upper = numeric.quantile(0.995)
        clipped = numeric.clip(self.lower, self.upper, axis=1)
        self.means = clipped.mean()
        self.stds = clipped.std().replace(0.0, 1.0).fillna(1.0)

        if self.numeric_bins:
            if self.numeric_bins < 2:
                raise ValueError("numeric_bins must be at least 2")
            scaled = ((clipped - self.means) / self.stds).to_numpy(dtype=np.float32)
            edges = np.zeros((len(self.numeric_columns), self.numeric_bins + 1), dtype=np.float32)
            mask = np.zeros((len(self.numeric_columns), self.numeric_bins), dtype=np.float32)
            quantiles = np.linspace(0.0, 1.0, self.numeric_bins + 1)
            for i in range(len(self.numeric_columns)):
                unique_edges = np.unique(np.quantile(scaled[:, i], quantiles)).astype(np.float32)
                if len(unique_edges) == 1:
                    unique_edges = np.array([unique_edges[0] - 1.0, unique_edges[0] + 1.0], dtype=np.float32)
                count = len(unique_edges) - 1
                edges[i, : count + 1] = unique_edges
                edges[i, count + 1 :] = unique_edges[-1]
                mask[i, :count] = 1.0
            self.numeric_bin_edges = edges
            self.numeric_bin_mask = mask

        for column in self.categorical_columns:
            values = frame[column].astype("string").fillna("__MISSING__")
            unique = sorted(values.unique().tolist())
            self.vocabularies[column] = {value: i + 1 for i, value in enumerate(unique)}
        return self

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.vocabularies[c]) + 1 for c in self.categorical_columns]

    @property
    def num_numeric_features(self) -> int:
        return len(self.numeric_columns) + len(self.missing_indicator_columns)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if any(x is None for x in (self.medians, self.means, self.stds, self.lower, self.upper)):
            raise RuntimeError("TabularPreprocessor.fit must be called before transform")
        numeric = frame[self.numeric_columns].replace([np.inf, -np.inf], np.nan).astype("float64")
        missing_flags = numeric[self.missing_indicator_columns].isna().to_numpy(dtype=np.float32)
        numeric = numeric.fillna(self.medians).clip(self.lower, self.upper, axis=1)
        numeric = ((numeric - self.means) / self.stds).to_numpy(dtype=np.float32)
        numeric = np.concatenate([numeric, missing_flags], axis=1)

        categorical = np.zeros((len(frame), len(self.categorical_columns)), dtype=np.int64)
        for i, column in enumerate(self.categorical_columns):
            values = frame[column].astype("string").fillna("__MISSING__")
            categorical[:, i] = values.map(self.vocabularies[column]).fillna(0).to_numpy(np.int64)
        return np.concatenate([numeric, categorical.astype(np.float32)], axis=1)


class TabularDataset(Dataset):
    def __init__(self, features: np.ndarray, target: np.ndarray | None = None):
        self.features = torch.from_numpy(features)
        self.target = None if target is None else torch.from_numpy(target.astype(np.float32))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        if self.target is None:
            return self.features[index]
        return self.features[index], self.target[index]


class FeatureGate(nn.Module):
    """Trainable per-feature soft gate implemented explicitly with nn.Parameter."""

    def __init__(self, features: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(features))
        self.scale = nn.Parameter(torch.ones(features))

    def forward(self, x: Tensor) -> Tensor:
        # Multiplying by 2 makes an initial sigmoid gate equal to an identity map.
        return x * (2.0 * torch.sigmoid(self.logits)) * self.scale


class CrossNetwork(nn.Module):
    """Low-rank explicit feature crossing: x_(l+1)=x0*(xl.w)+b+xl."""

    def __init__(self, features: int, layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(layers, features))
        self.biases = nn.Parameter(torch.zeros(layers, features))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x0: Tensor) -> Tensor:
        x = x0
        for weight, bias in zip(self.weights, self.biases):
            interaction = torch.sum(x * weight, dim=1, keepdim=True)
            x = x0 * interaction + bias + x
        return x


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class CreditRiskNet(nn.Module):
    """Single-input, single-output residual deep-and-cross network."""

    def __init__(
        self,
        num_numeric: int,
        cardinalities: Sequence[int],
        hidden_dim: int = 192,
        depth: int = 3,
        cross_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = TabularEmbedding(num_numeric, cardinalities)
        self.input_gate = FeatureGate(self.encoder.output_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dim),
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross = CrossNetwork(hidden_dim, cross_layers)
        self.deep = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(depth)])
        self.final = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.encoder.embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.fusion(self.input_gate(self.encoder(x)))
        cross = self.cross(x0)
        deep = self.deep(x0)
        return self.final(torch.cat([cross, deep], dim=1)).squeeze(1)


def build_model(
    config: Config,
    num_numeric: int,
    cardinalities: Sequence[int],
    numeric_bin_edges: tuple[np.ndarray, np.ndarray] | None = None,
) -> nn.Module:
    if config.model == "deep_cross":
        return CreditRiskNet(num_numeric, cardinalities, config.hidden_dim, config.depth, config.cross_layers, config.dropout)
    if config.model == "transformer":
        from model_transformer import TabularTransformer

        return TabularTransformer(
            num_numeric, cardinalities, config.hidden_dim, config.depth, config.dropout,
            numeric_embedding=config.numeric_embedding, numeric_bin_edges=numeric_bin_edges,
        )
    if config.model == "tabm":
        from model_tabm import TabMClassifier

        return TabMClassifier(
            num_numeric, cardinalities, config.hidden_dim, config.depth,
            config.dropout, config.tabm_k, config.numeric_embedding,
            numeric_bin_edges, config.tabm_embedding_dim,
        )
    raise ValueError(f"Unknown model: {config.model}")


class DecoupledAdamW(Optimizer):
    """Adam with bias correction and decoupled weight decay, built from Optimizer."""

    def __init__(self, params: Iterable, lr: float = 1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr <= 0 or eps <= 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1 or weight_decay < 0:
            raise ValueError("Invalid optimizer hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("DecoupledAdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                correction1 = 1 - beta1 ** state["step"]
                correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(correction2) / correction1
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group["eps"]), value=-step_size)
        return loss


def make_optimizer(model: nn.Module, config: Config) -> Optimizer:
    if config.model == "tabm":
        return torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim == 1 or name.endswith("bias") else decay).append(parameter)
    return DecoupledAdamW(
        [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=config.learning_rate,
    )


def best_f1(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    # Weighted BCE shifts probability calibration, so search the full useful range.
    thresholds = np.linspace(0.01, 0.99, 197)
    scores = np.array([f1_score(y_true, probabilities >= t) for t in thresholds])
    index = int(scores.argmax())
    return float(scores[index]), float(thresholds[index])


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> dict:
    model.eval()
    losses, targets, probabilities = [], [], []
    with torch.no_grad():
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            logits = model(features)
            loss = criterion(
                logits,
                target.unsqueeze(1).expand_as(logits) if logits.ndim == 2 else target,
            )
            losses.append(loss.item() * len(target))
            targets.append(target.cpu().numpy())
            batch_probabilities = torch.sigmoid(logits)
            if batch_probabilities.ndim == 2:
                batch_probabilities = batch_probabilities.mean(dim=1)
            probabilities.append(batch_probabilities.cpu().numpy())
    y_true = np.concatenate(targets)
    preds = np.concatenate(probabilities)
    f1, threshold = best_f1(y_true, preds)
    return {
        "loss": float(sum(losses) / len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, preds)),
        "pr_auc": float(average_precision_score(y_true, preds)),
        "f1": f1,
        "threshold": threshold,
        "predictions": preds,
    }


def gradient_statistics(model: nn.Module) -> list[dict]:
    result = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and parameter.ndim > 1:
            gradient = parameter.grad.detach().abs()
            result.append({"layer": name, "mean_abs_gradient": gradient.mean().item(), "max_abs_gradient": gradient.max().item()})
    return result


def train_fold(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    categorical_columns: Sequence[str],
    numeric_columns: Sequence[str],
    config: Config,
    device: torch.device,
    fold: int,
):
    preprocessor = TabularPreprocessor(
        categorical_columns, numeric_columns, config.numeric_missing_indicators,
        numeric_bins=config.numeric_bins if config.numeric_embedding == "ple" else 0,
    ).fit(train_frame)
    x_train = preprocessor.transform(train_frame)
    x_val = preprocessor.transform(validation_frame)
    train_dataset = TabularDataset(x_train, train_frame[TARGET].to_numpy())
    val_dataset = TabularDataset(x_val, validation_frame[TARGET].to_numpy())
    generator = torch.Generator().manual_seed(config.seed + fold)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=config.num_workers)

    bin_edges = None
    if config.numeric_embedding == "ple":
        bin_edges = (preprocessor.numeric_bin_edges, preprocessor.numeric_bin_mask)
    model = build_model(config, preprocessor.num_numeric_features, preprocessor.cardinalities, bin_edges).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6)
    positive = float(train_frame[TARGET].sum())
    negative = float(len(train_frame) - positive)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / max(positive, 1.0), device=device))

    history, best_state, best_auc, stale = [], None, -np.inf, 0
    saved_gradients: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for batch_index, (features, target) in enumerate(train_loader):
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(
                logits,
                target.unsqueeze(1).expand_as(logits) if logits.ndim == 2 else target,
            )
            loss.backward()
            if batch_index == 0:
                saved_gradients = gradient_statistics(model)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            running_loss += loss.item() * len(target)
            seen += len(target)

        metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step(metrics["roc_auc"])
        row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "val_loss": metrics["loss"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "f1": metrics["f1"],
            "f1_threshold": metrics["threshold"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"fold={fold} epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} ROC-AUC={row['roc_auc']:.4f} PR-AUC={row['pr_auc']:.4f}"
        )
        if metrics["roc_auc"] > best_auc + 1e-5:
            best_auc, stale = metrics["roc_auc"], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, val_loader, device, criterion)
    return model, preprocessor, history, saved_gradients, final_metrics


def run_tuning(
    train: pd.DataFrame,
    categorical: Sequence[str],
    numeric: Sequence[str],
    config: Config,
    device: torch.device,
    trials: int,
    tuning_rows: int,
    output_dir: Path,
) -> Config:
    """Small deterministic holdout search; the final CV remains the unbiased estimate."""
    candidates = [
        {"hidden_dim": 128, "depth": 2, "dropout": 0.10, "learning_rate": 1.0e-3},
        {"hidden_dim": 192, "depth": 3, "dropout": 0.20, "learning_rate": 8.0e-4},
        {"hidden_dim": 256, "depth": 3, "dropout": 0.30, "learning_rate": 5.0e-4},
    ][:trials]
    subset = train
    if tuning_rows < len(train):
        subset, _ = train_test_split(train, train_size=tuning_rows, stratify=train[TARGET], random_state=config.seed + 99)
    tuning_train, tuning_val = train_test_split(
        subset, test_size=0.2, stratify=subset[TARGET], random_state=config.seed + 100
    )
    rows = []
    best_score, best_values = -np.inf, None
    for trial, values in enumerate(candidates, start=1):
        trial_config = copy.deepcopy(config)
        for key, value in values.items():
            setattr(trial_config, key, value)
        trial_config.epochs = min(config.epochs, 3)
        trial_config.patience = 2
        _, _, history, _, metrics = train_fold(
            tuning_train, tuning_val, categorical, numeric, trial_config, device, fold=trial
        )
        row = {"trial": trial, **values, "epochs_run": len(history), "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"]}
        rows.append(row)
        if metrics["roc_auc"] > best_score:
            best_score, best_values = metrics["roc_auc"], values
    pd.DataFrame(rows).to_csv(output_dir / "tuning_log.csv", index=False)
    selected = copy.deepcopy(config)
    if best_values:
        for key, value in best_values.items():
            setattr(selected, key, value)
    (output_dir / "selected_hyperparameters.json").write_text(
        json.dumps({"selection_metric": "ROC-AUC", "best_tuning_roc_auc": best_score, **(best_values or {})}, indent=2),
        encoding="utf-8",
    )
    return selected


@torch.no_grad()
def predict(model: nn.Module, preprocessor: TabularPreprocessor, frame: pd.DataFrame, config: Config, device: torch.device) -> np.ndarray:
    features = preprocessor.transform(frame)
    loader = DataLoader(TabularDataset(features), batch_size=config.batch_size * 2, shuffle=False)
    model.eval()
    output = []
    for x in loader:
        logits = model(x.to(device))
        batch_probabilities = torch.sigmoid(logits)
        if batch_probabilities.ndim == 2:
            batch_probabilities = batch_probabilities.mean(dim=1)
        output.append(batch_probabilities.cpu().numpy())
    return np.concatenate(output)


def plot_training(history: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for fold, part in history.groupby("fold"):
        axes[0].plot(part["epoch"], part["train_loss"], "--", alpha=0.75, label=f"train F{fold}")
        axes[0].plot(part["epoch"], part["val_loss"], alpha=0.9, label=f"val F{fold}")
        axes[1].plot(part["epoch"], part["roc_auc"], label=f"F{fold}")
        axes[2].plot(part["epoch"], part["pr_auc"], label=f"F{fold}")
    axes[0].set_title("Loss curves")
    axes[1].set_title("Validation ROC-AUC")
    axes[2].set_title("Validation PR-AUC")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def plot_gradients(gradients: pd.DataFrame, output_dir: Path) -> None:
    summary = gradients.groupby("layer", as_index=False)[["mean_abs_gradient", "max_abs_gradient"]].mean()
    summary = summary.sort_values("mean_abs_gradient")
    fig, axis = plt.subplots(figsize=(10, max(5, len(summary) * 0.24)))
    y = np.arange(len(summary))
    axis.barh(y, summary["max_abs_gradient"], alpha=0.45, label="max |gradient|")
    axis.barh(y, summary["mean_abs_gradient"], alpha=0.9, label="mean |gradient|")
    axis.set_yticks(y, summary["layer"], fontsize=7)
    axis.set_xscale("log")
    axis.set_title("Gradient flow after the first batch of each fold")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gradient_flow.png", dpi=180)
    plt.close(fig)


def load_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(config.train_path)
    test = pd.read_parquet(config.test_path)
    if config.max_train_rows and config.max_train_rows < len(train):
        train, _ = train_test_split(
            train, train_size=config.max_train_rows, stratify=train[TARGET], random_state=config.seed
        )
        train = train.reset_index(drop=True)
    return train, test


def run(config: Config, tune_trials: int = 0, tuning_rows: int = 60000) -> dict:
    if config.numeric_embedding == "ple" and config.model not in {"transformer", "tabm"}:
        raise ValueError("PLE numeric embeddings are supported only by Transformer and TabM")
    if config.numeric_embedding == "ple" and config.numeric_bins < 2:
        raise ValueError("PLE needs at least two numerical bins")
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test = load_data(config)
    categorical = train.drop(columns=[TARGET]).select_dtypes(include=["category", "object", "string"]).columns.tolist()
    numeric = [c for c in train.columns if c not in categorical + [TARGET, ID_COLUMN]]
    splitter = StratifiedKFold(n_splits=max(2, config.folds), shuffle=True, random_state=config.seed)
    splits = list(splitter.split(train, train[TARGET]))[: config.folds]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} rows={len(train)} numeric={len(numeric)} categorical={len(categorical)}")
    if tune_trials:
        config = run_tuning(train, categorical, numeric, config, device, tune_trials, tuning_rows, output_dir)
        print(f"selected tuned configuration: {asdict(config)}")

    oof = np.full(len(train), np.nan, dtype=np.float32)
    test_predictions = np.zeros(len(test), dtype=np.float64)
    histories, gradients, fold_rows = [], [], []
    start = time.time()
    for fold, (train_index, validation_index) in enumerate(splits, start=1):
        model, preprocessor, history, gradient_rows, metrics = train_fold(
            train.iloc[train_index], train.iloc[validation_index], categorical, numeric, config, device, fold
        )
        oof[validation_index] = metrics.pop("predictions")
        test_predictions += predict(model, preprocessor, test, config, device) / len(splits)
        histories.extend(history)
        gradients.extend({"fold": fold, **row} for row in gradient_rows)
        fold_rows.append({
            "fold": fold,
            "numeric_missing_indicators": len(preprocessor.missing_indicator_columns),
            **metrics,
        })

    history_frame = pd.DataFrame(histories)
    gradient_frame = pd.DataFrame(gradients)
    fold_frame = pd.DataFrame(fold_rows)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    gradient_frame.to_csv(output_dir / "gradient_statistics.csv", index=False)
    fold_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    plot_training(history_frame, output_dir)
    plot_gradients(gradient_frame, output_dir)

    valid = ~np.isnan(oof)
    overall_f1, threshold = best_f1(train.loc[valid, TARGET].to_numpy(), oof[valid])
    summary = {
        "device": str(device),
        "model": config.model,
        "rows_used": int(len(train)),
        "folds_completed": int(len(splits)),
        "oof_coverage": float(valid.mean()),
        "local_roc_auc": float(roc_auc_score(train.loc[valid, TARGET], oof[valid])),
        "local_pr_auc": float(average_precision_score(train.loc[valid, TARGET], oof[valid])),
        "local_f1": overall_f1,
        "local_f1_threshold": threshold,
        "mean_fold_roc_auc": float(fold_frame["roc_auc"].mean()),
        "std_fold_roc_auc": float(fold_frame["roc_auc"].std(ddof=0)),
        "elapsed_minutes": (time.time() - start) / 60,
        "public_leaderboard_roc_auc": None,
        "private_leaderboard_roc_auc": None,
        "adequacy": (
            "Compare local ROC-AUC and PR-AUC against the LightGBM baseline (0.7639/0.2479). "
            "Leaderboard agreement cannot be assessed until Kaggle scores are recorded."
        ),
        "config": asdict(config),
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame({ID_COLUMN: test[ID_COLUMN].astype(int), TARGET: test_predictions}).to_csv(
        output_dir / "submission_dl.csv", index=False
    )
    pd.DataFrame({ID_COLUMN: train.loc[valid, ID_COLUMN].astype(int), TARGET: oof[valid]}).to_csv(
        output_dir / "oof_predictions.csv", index=False
    )
    leaderboard_path = output_dir / "leaderboard_results.csv"
    if not leaderboard_path.exists():
        pd.DataFrame(
            [{
                "model": config.model,
                "local_roc_auc": summary["local_roc_auc"],
                "local_pr_auc": summary["local_pr_auc"],
                "public_leaderboard_roc_auc": None,
                "private_leaderboard_roc_auc": None,
            }]
        ).to_csv(leaderboard_path, index=False)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> tuple[Config, int, int]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--model", choices=["deep_cross", "transformer", "tabm"], default="deep_cross")
    parser.add_argument("--numeric-missing-indicators", action="store_true")
    parser.add_argument("--numeric-embedding", choices=["linear", "ple"], default="linear")
    parser.add_argument("--numeric-bins", type=int, default=16)
    parser.add_argument("--tabm-k", type=int, default=32)
    parser.add_argument("--tabm-embedding-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--cross-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--tune-trials", type=int, choices=range(0, 4), default=0)
    parser.add_argument("--tuning-rows", type=int, default=60000)
    arguments = vars(parser.parse_args())
    tune_trials = arguments.pop("tune_trials")
    tuning_rows = arguments.pop("tuning_rows")
    if arguments["output_dir"] is None:
        arguments["output_dir"] = f"artifacts/{arguments['model']}_30_epochs"
    return Config(**arguments), tune_trials, tuning_rows


if __name__ == "__main__":
    parsed_config, parsed_trials, parsed_tuning_rows = parse_args()
    run(parsed_config, parsed_trials, parsed_tuning_rows)
