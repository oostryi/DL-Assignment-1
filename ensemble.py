from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 42


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def prediction_column(frame: pd.DataFrame) -> str:
    for column in ("prediction", TARGET):
        if column in frame.columns and column != ID:
            return column
    other = [column for column in frame.columns if column != ID]
    if len(other) == 1:
        return other[0]
    raise ValueError("Could not identify prediction column")


def check_prediction_file(frame: pd.DataFrame, name: str) -> None:
    if ID not in frame.columns:
        raise ValueError(f"{name}: missing {ID}")
    if frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError(f"{name}: {ID} must be unique and non-null")

    column = prediction_column(frame)
    values = frame[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: predictions contain NaN or Inf")
    if np.any((values < 0) | (values > 1)):
        raise ValueError(f"{name}: predictions must be in [0, 1]")


def load_predictions(path: str | Path, name: str) -> pd.DataFrame:
    frame = read_table(path)
    check_prediction_file(frame, name)
    column = prediction_column(frame)
    return frame[[ID, column]].rename(columns={column: name})


def merge_exact(base: pd.DataFrame, predictions: pd.DataFrame, name: str) -> pd.DataFrame:
    if set(base[ID]) != set(predictions[ID]):
        raise ValueError(f"{name}: SK_ID_CURR set does not match the base data")
    return base.merge(predictions, on=ID, how="left", validate="one_to_one", sort=False)


def prepare_data(
    train_path: str | Path,
    test_path: str | Path,
    lightgbm_oof: str | Path,
    lightgbm_test: str | Path,
    tabm_oof: str | Path,
    tabm_test: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = read_table(train_path)
    test = read_table(test_path)

    if ID not in train.columns or TARGET not in train.columns or ID not in test.columns:
        raise ValueError("Base train/test files do not contain required columns")
    if train[ID].isna().any() or test[ID].isna().any():
        raise ValueError("Base SK_ID_CURR values must be non-null")
    if train[ID].duplicated().any() or test[ID].duplicated().any():
        raise ValueError("Base SK_ID_CURR values must be unique")
    if train[TARGET].isna().any() or not set(pd.unique(train[TARGET])).issubset({0, 1, False, True}):
        raise ValueError("TARGET must contain only binary 0/1 values")

    train_meta = train[[ID, TARGET]].copy()
    test_meta = test[[ID]].copy()

    train_meta = merge_exact(train_meta, load_predictions(lightgbm_oof, "lightgbm"), "LightGBM OOF")
    train_meta = merge_exact(train_meta, load_predictions(tabm_oof, "tabm"), "TabM OOF")
    test_meta = merge_exact(test_meta, load_predictions(lightgbm_test, "lightgbm"), "LightGBM test")
    test_meta = merge_exact(test_meta, load_predictions(tabm_test, "tabm"), "TabM test")

    return train_meta, test_meta


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, pred)),
        "pr_auc": float(average_precision_score(y, pred)),
    }


def rank_blend(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_rank = pd.Series(first).rank(method="average", pct=True).to_numpy()
    second_rank = pd.Series(second).rank(method="average", pct=True).to_numpy()
    return (first_rank + second_rank) / 2.0


def weighted_blend_search(
    y: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    step: float = 0.05,
) -> tuple[pd.DataFrame, float]:
    if not 0 < step <= 1:
        raise ValueError("blend step must be in (0, 1]")

    rows = []
    weights = [float(weight) for weight in np.arange(0.0, 1.0, step)] + [1.0]
    for weight in weights:
        pred = weight * first + (1.0 - weight) * second
        rows.append({"lightgbm_weight": weight, "tabm_weight": 1.0 - weight, **metrics(y, pred)})

    results = pd.DataFrame(rows)
    best = results.sort_values(["roc_auc", "pr_auc"], ascending=False).iloc[0]
    return results, float(best["lightgbm_weight"])


def stacking_predictions(
    train_x: np.ndarray,
    test_x: np.ndarray,
    y: np.ndarray,
    folds: int = 5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if folds < 2:
        raise ValueError("meta folds must be at least 2")
    class_counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    if class_counts.min() < folds:
        raise ValueError("each target class must contain at least one row per meta fold")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_x, y), 1):
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        model.fit(train_x[train_idx], y[train_idx])
        pred = model.predict_proba(train_x[valid_idx])[:, 1]
        oof[valid_idx] = pred
        fold_rows.append({"fold": fold, **metrics(y[valid_idx], pred)})

    final_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    final_model.fit(train_x, y)
    test_pred = final_model.predict_proba(test_x)[:, 1]
    return oof, test_pred, pd.DataFrame(fold_rows)


def save_plot(metrics_frame: pd.DataFrame, output_dir: Path) -> None:
    ordered = metrics_frame.sort_values("roc_auc")
    values = ordered["roc_auc"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(values, ordered["model"])
    margin = max(0.0005, (values.max() - values.min()) * 0.15)
    ax.set_xlim(values.min() - margin, values.max() + margin)

    for _, row in ordered.iterrows():
        ax.text(
            row["roc_auc"] + margin * 0.08,
            row["model"],
            f'{row["roc_auc"]:.6f}',
            va="center",
        )

    ax.set_xlabel("OOF ROC-AUC")
    ax.set_title("OOF ROC-AUC comparison")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_meta, test_meta = prepare_data(
        args.train_path,
        args.test_path,
        args.lightgbm_oof,
        args.lightgbm_test,
        args.tabm_oof,
        args.tabm_test,
    )

    y = train_meta[TARGET].to_numpy(dtype=int)
    lgb_oof = train_meta["lightgbm"].to_numpy(dtype=float)
    tabm_oof = train_meta["tabm"].to_numpy(dtype=float)
    lgb_test = test_meta["lightgbm"].to_numpy(dtype=float)
    tabm_test = test_meta["tabm"].to_numpy(dtype=float)

    experiments: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}

    experiments["blend_50_50"] = (
        "blend",
        (lgb_oof + tabm_oof) / 2.0,
        (lgb_test + tabm_test) / 2.0,
    )

    blend_results, best_weight = weighted_blend_search(y, lgb_oof, tabm_oof, args.blend_step)
    weighted_oof = best_weight * lgb_oof + (1.0 - best_weight) * tabm_oof
    weighted_test = best_weight * lgb_test + (1.0 - best_weight) * tabm_test
    experiments["blend_weighted"] = ("blend", weighted_oof, weighted_test)

    experiments["rank_blend"] = (
        "blend",
        rank_blend(lgb_oof, tabm_oof),
        rank_blend(lgb_test, tabm_test),
    )

    train_x = train_meta[["lightgbm", "tabm"]].to_numpy(dtype=float)
    test_x = test_meta[["lightgbm", "tabm"]].to_numpy(dtype=float)
    stack_oof, stack_test, stack_folds = stacking_predictions(train_x, test_x, y, args.meta_folds)
    experiments["logistic_stack"] = ("two_stage", stack_oof, stack_test)

    rows = [
        {"model": "lightgbm", "kind": "base", **metrics(y, lgb_oof)},
        {"model": "tabm", "kind": "base", **metrics(y, tabm_oof)},
    ]
    for name, (kind, oof_pred, _) in experiments.items():
        rows.append({"model": name, "kind": kind, **metrics(y, oof_pred)})
    metrics_frame = pd.DataFrame(rows)

    ensemble_rows = metrics_frame[metrics_frame["kind"] != "base"]
    best_row = ensemble_rows.sort_values(["roc_auc", "pr_auc"], ascending=False).iloc[0]
    selected_name = str(best_row["model"])
    selected_test = experiments[selected_name][2]

    submission = pd.DataFrame({ID: test_meta[ID].astype(int), TARGET: selected_test})
    submission.to_csv(output_dir / "submission_ensemble.csv", index=False)
    metrics_frame.to_csv(output_dir / "base_and_ensemble_metrics.csv", index=False)
    blend_results.to_csv(output_dir / "blend_results.csv", index=False)
    stack_folds.to_csv(output_dir / "stack_fold_metrics.csv", index=False)

    save_plot(metrics_frame, output_dir)

    return {
        "selected_ensemble": selected_name,
        "selected_local_roc_auc": float(best_row["roc_auc"]),
        "selected_local_pr_auc": float(best_row["pr_auc"]),
        "best_lightgbm_weight": best_weight,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightGBM + TabM ensemble")
    parser.add_argument("--train-path", default="data/clean/train_eng.parquet")
    parser.add_argument("--test-path", default="data/clean/test_eng.parquet")
    parser.add_argument("--lightgbm-oof", default="artifacts/ml_lightgbm_predictions/oof_predictions.parquet")
    parser.add_argument("--lightgbm-test", default="artifacts/ml_lightgbm_predictions/test_predictions.parquet")
    parser.add_argument("--tabm-oof", default="artifacts/tabm_ensemble_predictions/oof_predictions.csv")
    parser.add_argument("--tabm-test", default="artifacts/tabm_ensemble_predictions/submission_dl.csv")
    parser.add_argument("--output-dir", default="artifacts/ensemble")
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument("--meta-folds", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(f"Selected ensemble: {result['selected_ensemble']}")
    print(f"OOF ROC-AUC: {result['selected_local_roc_auc']:.10f}")
    print(f"OOF PR-AUC:  {result['selected_local_pr_auc']:.10f}")
