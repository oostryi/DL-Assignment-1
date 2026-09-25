# Kaggle late-submission comparison

Competition: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk). Ten complete five-fold experiment predictions were submitted on 2026-09-25. Kaggle marked every submission `COMPLETE`. ROC-AUC is the leaderboard metric; PR-AUC is available only from local OOF validation.

| Experiment | Local OOF ROC-AUC | Local OOF PR-AUC | Public ROC-AUC | Private ROC-AUC | Kaggle ref |
|---|---:|---:|---:|---:|---:|
| Transformer 64×2 + PLE + missing flags, 30 epochs/patience 5 | 0.75938 | 0.23907 | **0.76267** | **0.75680** | 56550448 |
| TabM, dropout 0.1 | 0.76158 | 0.24342 | 0.76126 | 0.75641 | 56550445 |
| TabM, dropout 0.2 | **0.76240** | **0.24422** | 0.76118 | 0.75634 | 56550435 |
| Transformer 64×2 + PLE + missing flags, 100 epochs/patience 15 | 0.75917 | 0.23852 | 0.76266 | 0.75571 | 56550450 |
| Transformer 64×2, linear numeric tokens + missing flags | 0.75407 | 0.23384 | 0.74418 | 0.74315 | 56550451 |
| Transformer 32×1 + missing flags | 0.75326 | 0.23202 | 0.74156 | 0.73930 | 56550453 |
| Transformer 32×1 baseline | 0.75023 | 0.22821 | 0.74140 | 0.73928 | 56550454 |
| Deep & Cross, GPU | 0.74901 | 0.22487 | 0.74131 | 0.73684 | 56550455 |
| Deep & Cross, CPU | 0.74879 | 0.22431 | 0.74137 | 0.73631 | 56550457 |
| Earlier deep learning reference | 0.74525 | 0.21850 | 0.74159 | 0.73568 | 56550458 |

The best Private score is the 30-epoch PLE Transformer (0.75680). Its Private score is 0.00258 below local OOF ROC-AUC. TabM dropout 0.2 led local validation, but its Private score is 0.00046 below the Transformer and 0.00007 below TabM dropout 0.1. These tiny TabM differences do not establish a reliable dropout effect. The longer Transformer run did not improve Private ROC-AUC (0.75571 versus 0.75680).

The sharp improvement from linear numeric tokens to PLE on the leaderboard (Private 0.74315 to 0.75680) is larger than the local OOF improvement (0.75407 to 0.75938). This suggests the numerical representation may be especially helpful on the shifted test population; it is an inference, not a causal conclusion from a controlled test. The local train/test shift described in `DATA_INSIGHTS.md` also cautions against treating OOF ROC-AUC as a direct estimate of the leaderboard result.

The 128-wide Transformer was excluded because the user had previously requested not to use width 128. Smoke runs were excluded because they train on only a small subset and do not provide full five-fold validation. The local LightGBM reference is OOF ROC-AUC 0.7639 / PR-AUC 0.2479, but there is no corresponding leaderboard submission in this comparison. Kaggle does not provide PR-AUC for these submissions.

The TabM dropout-0.2 CSV was submitted again on 2026-09-25 (Kaggle ref 56550705). The resubmission completed with the same Public/Private ROC-AUC, 0.76118/0.75634.

The Transformer 64×2 + PLE 30-epoch CSV was also submitted again on 2026-09-25 (Kaggle ref 56550734). The resubmission completed with the same Public/Private ROC-AUC, 0.76267/0.75680.
