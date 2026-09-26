from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ensemble import (
    ID,
    TARGET,
    load_predictions,
    prepare_data,
    rank_blend,
    run,
    stacking_predictions,
    weighted_blend_search,
)


def write_csv(path: Path, frame: pd.DataFrame) -> str:
    frame.to_csv(path, index=False)
    return str(path)


def make_small_inputs(tmp_path: Path):
    train = pd.DataFrame({ID: [1, 2, 3, 4], TARGET: [0, 1, 0, 1]})
    test = pd.DataFrame({ID: [10, 11]})
    train_path = write_csv(tmp_path / "train.csv", train)
    test_path = write_csv(tmp_path / "test.csv", test)

    lgb_oof = write_csv(tmp_path / "lgb_oof.csv", pd.DataFrame({ID: [1, 2, 3, 4], TARGET: [0.1, 0.8, 0.2, 0.7]}))
    lgb_test = write_csv(tmp_path / "lgb_test.csv", pd.DataFrame({ID: [10, 11], TARGET: [0.3, 0.6]}))
    tabm_oof = write_csv(tmp_path / "tabm_oof.csv", pd.DataFrame({ID: [4, 3, 2, 1], TARGET: [0.75, 0.25, 0.85, 0.15]}))
    tabm_test = write_csv(tmp_path / "tabm_test.csv", pd.DataFrame({ID: [10, 11], TARGET: [0.4, 0.5]}))
    return train_path, test_path, lgb_oof, lgb_test, tabm_oof, tabm_test


def test_predictions_are_joined_by_id(tmp_path: Path):
    paths = make_small_inputs(tmp_path)
    train_meta, test_meta = prepare_data(*paths)
    assert train_meta["tabm"].tolist() == [0.15, 0.85, 0.25, 0.75]
    assert test_meta.shape == (2, 3)


def test_duplicate_prediction_ids_are_rejected(tmp_path: Path):
    path = write_csv(tmp_path / "bad.csv", pd.DataFrame({ID: [1, 1], TARGET: [0.2, 0.3]}))
    with pytest.raises(ValueError, match="unique"):
        load_predictions(path, "bad")


def test_invalid_prediction_range_is_rejected(tmp_path: Path):
    path = write_csv(tmp_path / "bad_range.csv", pd.DataFrame({ID: [1, 2], TARGET: [0.2, 1.2]}))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_predictions(path, "bad")


def test_id_set_mismatch_is_rejected(tmp_path: Path):
    paths = list(make_small_inputs(tmp_path))
    paths[4] = write_csv(tmp_path / "bad_tabm.csv", pd.DataFrame({ID: [1, 2, 3, 99], TARGET: [0.1, 0.2, 0.3, 0.4]}))
    with pytest.raises(ValueError, match="does not match"):
        prepare_data(*paths)


def test_rank_blend_returns_valid_scores():
    pred = rank_blend(np.array([0.1, 0.9, 0.3]), np.array([0.8, 0.2, 0.5]))
    assert pred.shape == (3,)
    assert np.all((pred >= 0) & (pred <= 1))


def test_weight_search_checks_full_range():
    y = np.array([0, 0, 1, 1, 0, 1])
    first = np.array([0.05, 0.1, 0.8, 0.9, 0.2, 0.7])
    second = np.array([0.4, 0.3, 0.6, 0.55, 0.45, 0.5])
    results, weight = weighted_blend_search(y, first, second, step=0.3)
    assert results["lightgbm_weight"].min() == 0.0
    assert results["lightgbm_weight"].max() == 1.0
    assert 0 <= weight <= 1


def test_stacking_has_full_oof_coverage():
    rng = np.random.default_rng(42)
    rows = 200
    signal = rng.normal(size=rows)
    y = (signal + rng.normal(size=rows) > 0).astype(int)
    x = np.column_stack([
        1 / (1 + np.exp(-(signal + rng.normal(scale=0.7, size=rows)))),
        1 / (1 + np.exp(-(0.8 * signal + rng.normal(scale=0.9, size=rows)))),
    ])
    oof, test_pred, folds = stacking_predictions(x, x[:20], y, folds=5)
    assert len(oof) == rows
    assert np.isfinite(oof).all()
    assert len(test_pred) == 20
    assert len(folds) == 5


def test_full_runner_creates_submission(tmp_path: Path):
    rng = np.random.default_rng(7)
    n_train, n_test = 160, 30
    train_ids = np.arange(1000, 1000 + n_train)
    test_ids = np.arange(9000, 9000 + n_test)
    signal = rng.normal(size=n_train)
    y = (signal + rng.normal(scale=0.9, size=n_train) > 0.4).astype(int)

    train_path = write_csv(tmp_path / "train.csv", pd.DataFrame({ID: train_ids, TARGET: y}))
    test_path = write_csv(tmp_path / "test.csv", pd.DataFrame({ID: test_ids}))

    lgb_train = 1 / (1 + np.exp(-(signal + rng.normal(scale=0.8, size=n_train))))
    tabm_train = 1 / (1 + np.exp(-(0.85 * signal + rng.normal(scale=0.9, size=n_train))))
    lgb_test_values = rng.uniform(0.05, 0.95, size=n_test)
    tabm_test_values = rng.uniform(0.05, 0.95, size=n_test)

    lgb_oof = write_csv(tmp_path / "lgb_oof.csv", pd.DataFrame({ID: train_ids, TARGET: lgb_train}))
    lgb_test = write_csv(tmp_path / "lgb_test.csv", pd.DataFrame({ID: test_ids, TARGET: lgb_test_values}))
    tabm_oof = write_csv(tmp_path / "tabm_oof.csv", pd.DataFrame({ID: train_ids, TARGET: tabm_train}))
    tabm_test = write_csv(tmp_path / "tabm_test.csv", pd.DataFrame({ID: test_ids, TARGET: tabm_test_values}))

    output = tmp_path / "ensemble"
    summary = run(Namespace(
        train_path=train_path,
        test_path=test_path,
        lightgbm_oof=lgb_oof,
        lightgbm_test=lgb_test,
        tabm_oof=tabm_oof,
        tabm_test=tabm_test,
        output_dir=str(output),
        blend_step=0.1,
        meta_folds=4,
    ))

    assert summary["selected_ensemble"] in {"blend_50_50", "blend_weighted", "rank_blend", "logistic_stack"}
    submission = pd.read_csv(output / "submission_ensemble.csv")
    assert submission.columns.tolist() == [ID, TARGET]
    assert len(submission) == n_test
    assert submission[ID].is_unique
    assert submission[TARGET].between(0, 1).all()
