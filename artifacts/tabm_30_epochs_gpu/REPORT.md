# TabM: full five-fold GPU run

TabM (Tabular model with Multiple predictions) is a parameter-efficient
ensemble. One network contains 32 related MLP predictors (`k=32`): they share
most backbone weights, but retain member-specific scaling and output weights.
Each member's binary cross-entropy contributes to the training loss; at
inference, the 32 **probabilities**, not logits, are averaged. This uses the
official `tabm==0.0.3` implementation behind `model_tabm.py`.

The model used two blocks of width 512, dropout 0.1, AdamW with learning rate
0.002 and weight decay 0.0003, weighted BCE, batch size 512, and a 30-epoch
limit with patience 5. Preprocessing was fitted separately on the training
portion of each stratified fold (seed 42): median imputation, clipping and
scaling, numeric missingness indicators, 16-bin PLE embeddings (official
version B), and categorical one-hot features. All 307,511 training rows have
one out-of-fold prediction; 48,744 test rows have a submission prediction.

| Model | OOF ROC-AUC | OOF PR-AUC | GPU time |
|---|---:|---:|---:|
| Transformer 64 x 2 + indicators + PLE | 0.759383 | 0.239072 | 22.48 min |
| **TabM** | **0.761581** | **0.243421** | **15.70 min** |
| LightGBM reference (prior run) | 0.7639 | 0.2479 | — |

TabM improves on the retained Transformer by **0.002198 ROC-AUC** and
**0.004349 PR-AUC**, while running about 6.8 minutes faster in this run.
The gap to LightGBM is still about 0.00232 ROC-AUC and 0.00448 PR-AUC.

| Fold | ROC-AUC | PR-AUC | Epochs run |
|---|---:|---:|---:|
| 1 | 0.759844 | 0.242237 | 15 |
| 2 | 0.769163 | 0.253788 | 14 |
| 3 | 0.760738 | 0.238109 | 14 |
| 4 | 0.765940 | 0.252608 | 15 |
| 5 | 0.758430 | 0.238822 | 16 |

TabM improves ROC-AUC on all five folds versus the Transformer. On rows with
missing `EXT_SOURCE_3`, it scores 0.740652 versus 0.737660 for the Transformer;
on observed rows, 0.765788 versus 0.763700. The original 307,511 OOF IDs
match in order; predictions and gradient statistics are finite.

This is a comparison of two **model pipelines**, not an isolated architecture
ablation: TabM uses the official PLE version B for all numerical inputs,
including missingness flags, and one-hot categories, whereas the Transformer
uses its own numerical tokenizer and learned categorical embeddings. It also
uses the official package's recommended AdamW settings. Hyperparameters were
not extensively tuned. Both models were explored on the same CV splits, so
model selection can make the best OOF result optimistic. Weighted BCE means
the raw scores should not be treated as calibrated default probabilities.
The train/test shift remains, and Public/Private Kaggle scores are unknown
until an actual submission is evaluated.

Reproduce with `train_tabm.ps1` from the project root after installing
`requirements-dl.txt`. Curves, gradient-flow plot, fold metrics, OOF
predictions, and `submission_dl.csv` are in this directory.

Source: [TabM paper](https://arxiv.org/abs/2410.24210) and
[official implementation](https://github.com/yandex-research/tabm).
