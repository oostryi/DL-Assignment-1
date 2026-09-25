# Transformer ablation 1: numerical missingness indicators

Only one modification was tested: append a 0/1 indicator to the numerical
input for each numerical column containing missing values in the *training
part of that fold*. Flags are computed before median imputation. The input
remains a single matrix, and the model architecture, loss, optimizer, CV
splits, seed, learning rate, width, depth, batch size, scheduler, and stopping
rule are unchanged. There were 33 indicators in folds 1-4 and 32 in fold 5.
The default `--numeric-missing-indicators` setting is off, preserving prior
behavior.

Both runs used the RTX 5070 Ti GPU, all 307,511 training rows, stratified
5-fold CV with seed 42, a 30-epoch limit and patience 5. Predictions cover
every training row once.

| Metric | Original Transformer | + missing indicators | Difference |
|---|---:|---:|---:|
| OOF ROC-AUC | 0.750231 | 0.753258 | +0.003028 |
| OOF PR-AUC | 0.228207 | 0.232018 | +0.003811 |
| OOF best F1 | 0.296274 | 0.299197 | +0.002923 |
| Training time | 15.22 min | 19.73 min | +4.51 min |

| Fold | Original ROC-AUC | New ROC-AUC | Original PR-AUC | New PR-AUC |
|---|---:|---:|---:|---:|
| 1 | 0.748387 | 0.750473 | 0.228531 | 0.230389 |
| 2 | 0.758228 | 0.759061 | 0.237329 | 0.239215 |
| 3 | 0.750313 | 0.751950 | 0.227009 | 0.230774 |
| 4 | 0.756135 | 0.758450 | 0.239889 | 0.240158 |
| 5 | 0.746676 | 0.747198 | 0.226807 | 0.224980 |

The same OOF customers were aligned by `SK_ID_CURR` for subgroup checks:

| Subgroup | Rows | Original ROC-AUC | New ROC-AUC | Original PR-AUC | New PR-AUC |
|---|---:|---:|---:|---:|---:|
| `EXT_SOURCE_3` missing | 60,965 | 0.723143 | 0.727442 | 0.223740 | 0.228179 |
| `EXT_SOURCE_3` present | 246,546 | 0.756232 | 0.758723 | 0.229815 | 0.233041 |
| `EXT_SOURCE_1` missing | 173,378 | 0.748464 | 0.751468 | 0.238902 | 0.242598 |
| `EXT_SOURCE_1` present | 134,133 | 0.751903 | 0.754618 | 0.212526 | 0.215614 |

This is a useful but modest local gain. It helps both missing/present groups
for these two features, though not uniformly across every metric and fold.
It remains below the project's LightGBM OOF reference (ROC-AUC 0.7639,
PR-AUC 0.2479), so it is not the strongest standalone predictor yet.
Public/Private leaderboard results are unknown because these predictions have
not been submitted to Kaggle. The supplied `submission_dl.csv` is ready for
that external check; do not infer leaderboard gains from local CV alone.

Reproduce from the project root:

```powershell
.venv\Scripts\python.exe deep_learning.py --model transformer --numeric-missing-indicators --folds 5 --epochs 30 --patience 5 --batch-size 512 --hidden-dim 32 --depth 1 --output-dir artifacts/transformer_missing_indicators_gpu
```

`metrics_summary.json`, `fold_metrics.csv`, `training_history.csv`,
`training_curves.png`, `gradient_statistics.csv`, `gradient_flow.png`, and
`oof_predictions.csv` contain the full results. Validation checks confirmed
307,511 unique OOF IDs, 48,744 submission rows, finite scores and gradient
statistics, and five passing component tests.
