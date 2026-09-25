# TabM dropout 0.1 versus 0.2

Controlled change: TabM dropout increased from 0.1 to 0.2. Both runs used
the same seed 42, five stratified folds, 307,511 training rows, PLE and
missingness indicators, 32 ensemble members, two width-512 blocks, batch
size 512, AdamW (learning rate 0.002, weight decay 0.0003), and a 30-epoch
limit with patience 5. Checkpoints were selected by validation ROC-AUC.

| Metric | Dropout 0.1 | Dropout 0.2 | Difference |
|---|---:|---:|---:|
| Pooled OOF ROC-AUC | 0.761581 | 0.762404 | +0.000823 |
| Pooled OOF PR-AUC | 0.243421 | 0.244222 | +0.000802 |
| Mean fold ROC-AUC | 0.762823 | 0.762920 | +0.000098 |
| Training time | 15.70 min | 16.48 min | +0.78 min |

| Fold | ROC-AUC at 0.1 | ROC-AUC at 0.2 | PR-AUC at 0.1 | PR-AUC at 0.2 |
|---|---:|---:|---:|---:|
| 1 | 0.759844 | 0.759832 | 0.242237 | 0.242478 |
| 2 | 0.769163 | 0.769098 | 0.253788 | 0.253555 |
| 3 | 0.760738 | 0.760369 | 0.238109 | 0.237773 |
| 4 | 0.765940 | 0.766469 | 0.252608 | 0.251714 |
| 5 | 0.758430 | 0.758834 | 0.238822 | 0.238198 |

Conclusion: 0.2 modestly improves the pooled OOF metrics, but its foldwise
advantage is inconsistent. The gain may reflect fold-to-fold score scaling
or ordinary training variability; it is not enough to claim a robust
generalization improvement. Retain both runs and confirm on an independent
holdout or leaderboard before changing the primary TabM configuration.
Also, the training script seeds once before all folds. If earlier folds stop
after different numbers of epochs, later folds consume a different random
sequence. Thus this run is not a perfectly paired dropout-only comparison
of model initializations, despite matching CV splits and run seed.
Weighted BCE scores are not calibrated probabilities. Public/Private
leaderboard values are not available.

Reproduce from the project root with `train_tabm_dropout_02.ps1`. Full metrics, curves, gradient
diagnostics, OOF predictions and a 48,744-row submission are in this folder.
