# Combined Transformer experiment

This run combines the requested changes: numerical missingness indicators,
quantile-based piecewise-linear embeddings (PLE) for the 68 original numerical
features, token width 64, and two Transformer encoder layers. The binary
missingness indicators retain linear tokens. PLE bin boundaries and all other
preprocessing statistics are fitted only on the training portion of each fold.
The implementation is an adapted, clamped quantile PLE with 16 bins per feature,
not a claim of exact reproduction of the published model.

All results below use the same 307,511 rows, stratified five folds (seed 42),
30-epoch limit, patience 5, batch size 512, weighted BCE, optimizer, and GPU.
The model checkpoint in each fold is selected by validation ROC-AUC. No Kaggle
Public/Private score is available.

| Configuration | OOF ROC-AUC | OOF PR-AUC | Time |
|---|---:|---:|---:|
| Original Transformer, 32 x 1, no masks | 0.75023 | 0.22821 | 15.22 min |
| 32 x 1 + missingness indicators | 0.75326 | 0.23202 | 19.73 min |
| 64 x 2 + indicators, linear numerics | 0.75407 | 0.23384 | 26.30 min |
| **64 x 2 + indicators + PLE** | **0.75938** | **0.23907** | 22.48 min |
| 128 x 2 + indicators + PLE (not selected) | 0.75920 | 0.23944 | 39.18 min |

The combined model gains **0.00915 ROC-AUC** and **0.01087 PR-AUC** over the
original Transformer. The PLE change alone, compared with the linear 64 x 2
model, gains **0.00531 ROC-AUC** and **0.00523 PR-AUC**. The shorter elapsed
time of PLE 64 x 2 relative to linear 64 x 2 reflects fewer early-stopped
epochs (85 versus 96 across all folds), not necessarily faster computation
per epoch. Width 128 does not improve the primary OOF ROC-AUC and costs much
more time, so it is not used as the main configuration.

| Fold | ROC-AUC | PR-AUC | Epochs run |
|---|---:|---:|---:|
| 1 | 0.757822 | 0.239366 | 16 |
| 2 | 0.765153 | 0.248089 | 17 |
| 3 | 0.757423 | 0.235251 | 17 |
| 4 | 0.763316 | 0.245881 | 16 |
| 5 | 0.756329 | 0.238221 | 19 |

Missing `EXT_SOURCE_3` subgroup ROC-AUC rises from 0.728835 under linear
64 x 2 to 0.737660 with PLE, and observed `EXT_SOURCE_3` from 0.759318 to
0.763700. The model remains below the project's LightGBM reference
(ROC-AUC 0.7639, PR-AUC 0.2479). These configurations were explored on the
same CV splits, so selecting the highest OOF number can introduce selection
bias. A separate holdout or leaderboard check is needed for external evidence.

Reproduce with `train_transformer_64_ple.ps1` from the project root. The
directory contains fold/aggregate metrics, OOF predictions, loss and metric
curves, gradient diagnostics, and a 48,744-row `submission_dl.csv`. Checks
confirmed 307,511 unique OOF IDs, finite predictions and gradient statistics,
and six passing component tests.

The design follows the feature-token Transformer approach in
[Gorishniy et al., 2021](https://arxiv.org/abs/2106.11959) and the
quantile-based PLE idea in
[Gorishniy et al., 2022](https://arxiv.org/abs/2203.05556).
