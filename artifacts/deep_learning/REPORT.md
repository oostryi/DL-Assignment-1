# Previous Deep Learning Results (two-input, five-epoch model)

## Model and preprocessing

The final model is a two-branch deep-and-cross neural network. Sixteen categorical
features use learned embeddings, while 68 numerical features pass through a custom
trainable feature gate. The two branches are fused and processed by both a custom
cross network and a residual deep tower. A numerical-only auxiliary head adds an
extra loss signal. `FeatureGate` and `CrossNetwork` explicitly use `nn.Parameter`;
`DecoupledAdamW` is implemented directly from `torch.optim.Optimizer`.

All preprocessing is fitted separately inside each training fold. It includes median
imputation, 0.5/99.5 percentile clipping, standardization, unknown-category handling,
and learned embeddings. Regularization includes LayerNorm, dropout 0.2, weight decay
1e-4, gradient clipping at 5.0, ReduceLROnPlateau, and early stopping support.

## Parameter tuning

Two short holdout trials were run on a stratified 30,000-row subset. The selected
configuration was hidden width 192, residual depth 3, dropout 0.2, and learning rate
8e-4. Exact trial results are stored in `tuning_log.csv`; the chosen values are in
`selected_hyperparameters.json`.

## Local validation

The final run trained for five epochs on all 307,511 training rows with stratified
5-fold cross-validation.

| Metric | Deep learning | LightGBM baseline |
|---|---:|---:|
| OOF ROC-AUC | 0.74525 | 0.7639 |
| OOF PR-AUC | 0.21850 | 0.2479 |
| Mean fold ROC-AUC | 0.74612 | 0.7639 |
| Fold ROC-AUC std | 0.00459 | 0.0046 |

The neural network is adequate as a stable deep-learning baseline: it is far above
random ranking, PR-AUC is well above the positive-class prevalence (~0.0807), and
fold variance is low. It is not the best standalone model because it trails LightGBM
by about 0.0187 ROC-AUC and 0.0294 PR-AUC. The likely explanation is that gradient-
boosted trees have a stronger inductive bias for medium-sized heterogeneous tabular
data. More epochs, auxiliary relational data, or an ensemble are reasonable next
steps.

## Diagnostics and leaderboard

`training_curves.png` contains loss, ROC-AUC, and PR-AUC curves. `gradient_flow.png`
and `gradient_statistics.csv` document gradient health. `submission_dl.csv` contains
48,744 test predictions in Kaggle format. Public and Private leaderboard fields are
left blank in `leaderboard_results.csv`: they must be copied from a real Kaggle
submission and must not be inferred from local validation.
