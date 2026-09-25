# Single-input deep learning model: 30-epoch run

The model receives one matrix with 68 standardized numerical features followed by
16 categorical codes. The categorical codes are embedded inside the model; a
trainable `FeatureGate` scales the resulting representation. A `CrossNetwork` and
three residual blocks process it in parallel. The model returns one binary logit
and is trained with weighted `BCEWithLogitsLoss` and the custom `DecoupledAdamW`
optimizer. Preprocessing is fitted within each training fold. The 30-epoch limit
is paired with early stopping after five epochs without ROC-AUC improvement.

All 307,511 training rows were used in stratified 5-fold cross-validation.

| Model | OOF ROC-AUC | OOF PR-AUC | Fold ROC-AUC std |
|---|---:|---:|---:|
| Single-input Deep & Cross (this run) | 0.74879 | 0.22431 | 0.00416 |
| Previous two-input Deep & Cross, 5 epochs | 0.74525 | 0.21850 | 0.00459 |
| LightGBM baseline | 0.7639 | 0.2479 | 0.0046 |

The new model improved ROC-AUC by 0.00354 and PR-AUC by 0.00580 compared with
the previous run. This is an observational comparison: both the architecture and
training schedule changed, so the gain cannot be attributed to either one alone.
It remains behind LightGBM by 0.01511 ROC-AUC. It is adequate as a stable neural
baseline, but not the strongest standalone model in the project.

The best fold checkpoints occurred between epochs 5 and 8; each fold stopped
between epochs 10 and 13. The complete epoch history is in `training_history.csv`.
The loss/metric plots and gradient statistics are saved alongside this report.
`submission_dl.csv` contains 48,744 Kaggle-format predictions. Public and Private
leaderboard scores require an actual submission and are left empty in
`leaderboard_results.csv`.

An additional architecture is implemented in `model_transformer.py`. Its initial
one-epoch subset run only verified the training and prediction path; full GPU
results are in `artifacts/GPU_COMPARISON.md`.
