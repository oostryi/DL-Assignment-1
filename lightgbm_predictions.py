from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ID = "SK_ID_CURR"
TARGET = "TARGET"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LightGBM OOF and test predictions")
    parser.add_argument("--train-path", default="data/clean/train_eng.parquet")
    parser.add_argument("--test-path", default="data/clean/test_eng.parquet")
    parser.add_argument("--output-dir", default="artifacts/ml_lightgbm_predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.train_path)
    test = pd.read_parquet(args.test_path)

    if ID not in train.columns or ID not in test.columns:
        raise ValueError(f"Train and test data must contain {ID}")
    if TARGET not in train.columns or TARGET in test.columns:
        raise ValueError("Unexpected TARGET columns in train/test data")
    if train[ID].isna().any() or test[ID].isna().any():
        raise ValueError(f"{ID} values must be non-null")
    if train[ID].duplicated().any() or test[ID].duplicated().any():
        raise ValueError(f"{ID} values must be unique")
    if train[TARGET].isna().any() or not set(pd.unique(train[TARGET])).issubset({0, 1, False, True}):
        raise ValueError("TARGET must contain only binary 0/1 values")
    if list(train.drop(columns=[TARGET]).columns) != list(test.columns):
        raise ValueError("Train and test feature columns do not match")

    # Same feature set as ml.ipynb. SK_ID_CURR is intentionally kept because
    # the original team baseline used X = train_eng.drop(columns=['TARGET']).
    y = train[TARGET]
    x = train.drop(columns=[TARGET])
    x_test = test[x.columns]

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(x), dtype=float)
    test_pred = np.zeros(len(x_test), dtype=float)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y), 1):
        model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            objective="binary",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            x.iloc[train_idx],
            y.iloc[train_idx],
            eval_set=[(x.iloc[valid_idx], y.iloc[valid_idx])],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        valid_pred = model.predict_proba(x.iloc[valid_idx])[:, 1]
        oof[valid_idx] = valid_pred
        test_pred += model.predict_proba(x_test)[:, 1] / 5.0

        fold_rows.append(
            {
                "fold": fold,
                "roc_auc": float(roc_auc_score(y.iloc[valid_idx], valid_pred)),
                "pr_auc": float(average_precision_score(y.iloc[valid_idx], valid_pred)),
                "best_iteration": int(model.best_iteration_),
            }
        )
        print(
            f"Fold {fold}: ROC-AUC={fold_rows[-1]['roc_auc']:.6f}, "
            f"PR-AUC={fold_rows[-1]['pr_auc']:.6f}"
        )

    if not np.isfinite(oof).all() or not np.isfinite(test_pred).all():
        raise RuntimeError("LightGBM predictions contain NaN or Inf")
    if np.any((oof < 0) | (oof > 1)) or np.any((test_pred < 0) | (test_pred > 1)):
        raise RuntimeError("LightGBM predictions must be in [0, 1]")

    local_roc = float(roc_auc_score(y, oof))
    local_pr = float(average_precision_score(y, oof))

    pd.DataFrame({ID: train[ID], "prediction": oof}).to_parquet(
        output_dir / "oof_predictions.parquet", index=False
    )
    pd.DataFrame({ID: test[ID], "prediction": test_pred}).to_parquet(
        output_dir / "test_predictions.parquet", index=False
    )
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)

    print(f"Pooled OOF ROC-AUC: {local_roc:.10f}")
    print(f"Pooled OOF PR-AUC:  {local_pr:.10f}")


if __name__ == "__main__":
    main()
