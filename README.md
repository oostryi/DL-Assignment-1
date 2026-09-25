# DL-Assignment-1

## 1. Exploratory Data Analysis (EDA)
*   **Data Insights:** when exploring distributions and missing values, it was important to alter each day feature (cutting off certain values, turning some days into years). Other problem observed in this stage with the day counters was that days_employed of unemployed clients demanded resolution due to having anomalies (like NaNs). During this stage it was also discovered that the dataset has many features that cause drifting and had to be trimmed in the future. We have also discovered some collinear features that were deleted manually.
*   **Proposed Target Metric:** the proposed metric of the competition is ROC AUC. ROC AUC in this case is not the best metric to capture the model's results, since the dataset possesses a large class imbalance. ROC AUC in this case may be overly optimistic due to relying on False Positive Rate which may be inaccurate when negatives outweigh positives by a large amount. 
*   **Alternative Metrics:** a metric I like to use for such imbalances is PR AUC. It's easy to benchmark the model against random choice with this metric (since random choice would have PR AUC of the percentage of the minority in the dataset).
*   **Complementary Metric:** as a complementary metric I would choose F1. While it doesn't provide such a detailed look into the model's results as observing precision and recall separately, it is a great choice of condensing these two metrics into on for automation (for example hyperparameter optimization).

## 2. Validation
*   **Validation Strategy:** I decided to use Stratified 5-Fold for Validation. 
*   **Motivation for the strategy:** K-Fold is a great choice for class imbalance since it lets the model be validated more times while not having to create artificial data for it. Stratified K-Fold allows to keep the initial distribution intact, assuring that the initial 92% negative and 8% positive instances have the same ratio in each of the folds. This way all 5 folds equally contribute to the model's validation, avoiding cases when a fold doesn't have positive instances so that the model that was taught to a certain distribution wouldn't be thrown off by an entirely different distribution at validation time.
*   **Adversarial Validation:** adversarial validation showed that the train and test splits are different in a way that is possible for a simple logistic regression to spot the difference (indicated by around 0.7 ROC AUC for prediction where the data came from: the training or the test set). Adversarial Validation is what helped discover drift features.

## 3. Deep Learning Model

The deep learning solution lives in `deep_learning.py`, with a report-oriented walkthrough in `dl.ipynb`. The current model accepts a single matrix `[numerical values | categorical codes]` and returns one logit:

* categorical variables are represented by learned embeddings;
* numeric values and embeddings are combined into a single shared representation;
* `FeatureGate` and `CrossNetwork` are custom layers built with `nn.Module` and explicit `nn.Parameter` tensors;
* a residual tower and cross network capture complementary feature interactions;
* `DecoupledAdamW` is implemented directly from `torch.optim.Optimizer` with first/second moments, bias correction, and decoupled weight decay.

A further single-input architecture is available: a feature-token Transformer in `model_transformer.py`. It shares the same preprocessing, optimizer, CV, metrics and output format via `--model transformer`. The Transformer is more computationally expensive on CPU; use a smaller `--hidden-dim` and `--batch-size` for it.

Preprocessing is fitted independently inside every fold to prevent leakage. Numerical values are median-imputed, clipped to training-fold quantiles, and standardized. Unknown categories use embedding index 0; missing categories receive their own learned category when observed in the training fold. Training uses stratified CV, weighted binary cross-entropy, LayerNorm, dropout, gradient clipping, weight decay, ReduceLROnPlateau, and early stopping.

Install and run:

```powershell
python -m pip install -r requirements-dl.txt
python deep_learning.py --model deep_cross --folds 5 --epochs 30 --patience 5 --batch-size 4096 --output-dir artifacts/deep_cross_30_epochs
python deep_learning.py --model transformer --hidden-dim 64 --depth 2 --batch-size 256 --folds 5 --epochs 30 --patience 5 --output-dir artifacts/transformer_30_epochs
```

Each run writes fold and aggregate metrics, loss/metric curves, gradient-flow visualization, OOF predictions, and `submission_dl.csv` in its chosen output directory. Ten complete DL experiments were submitted to Kaggle on 2026-09-25; see `artifacts/KAGGLE_LEADERBOARD_COMPARISON.md` for their Public/Private scores and local-validation comparison. The local LightGBM reference is ROC-AUC 0.7639 and PR-AUC 0.2479.

The previous two-input, 5-epoch run is preserved in `artifacts/deep_learning/` for reference. It achieved OOF ROC-AUC **0.74525** and PR-AUC **0.21850**. These numbers do not describe the new single-input architecture; see `artifacts/deep_cross_30_epochs/` for its results.

The completed single-input run used all 307,511 rows with a 30-epoch limit and early stopping. It achieved OOF ROC-AUC **0.74879**, PR-AUC **0.22431**, best F1 **0.29359** (threshold 0.68), and fold ROC-AUC standard deviation **0.00416**. The best fold checkpoints were at epochs 5-8. See `artifacts/deep_cross_30_epochs/REPORT.md` for the comparison and interpretation. `artifacts/transformer_smoke/` contains a short functional run only; it is not a full validation score.

### Full GPU runs

On the RTX 5070 Ti Laptop GPU, install the matching CUDA build of PyTorch in the project environment and check `torch.cuda.is_available()` before training:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-dl.txt
uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu130 'torch==2.14.0+cu130'
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`deep_learning.py` automatically selects CUDA when it is available. The
following initial architectures were trained with five folds, all 307,511
rows, and a 30-epoch limit with early stopping on this GPU:

| Model | OOF ROC-AUC | OOF PR-AUC |
|---|---:|---:|
| Transformer | **0.75023** | **0.22821** |
| Deep & Cross | 0.74901 | 0.22487 |

See `artifacts/GPU_COMPARISON.md` for the initial epoch counts, timings,
interpretation and artifact directories. The newer Transformer and TabM runs
are documented below. The earlier CPU results remain in their original folders.

See `artifacts/DATA_INSIGHTS.md` for data quality, distribution shift, model
diagnostics, and prioritized experiments to improve the retained models.

### One-change Transformer ablation: numerical missingness indicators

The first follow-up experiment adds binary indicators for numerical columns
with missing values in each training fold. The default remains off, so the
original Transformer run is unchanged. All other settings match the full GPU
Transformer run above:

```powershell
.venv\Scripts\python.exe deep_learning.py --model transformer --numeric-missing-indicators --folds 5 --epochs 30 --patience 5 --batch-size 512 --hidden-dim 32 --depth 1 --output-dir artifacts/transformer_missing_indicators_gpu
```

OOF ROC-AUC improved from **0.75023** to **0.75326** and OOF PR-AUC from
**0.22821** to **0.23202**. The indicators helped on all five folds by ROC-AUC,
but one fold's PR-AUC declined. See
`artifacts/transformer_missing_indicators_gpu/EXPERIMENT.md` for the controlled
comparison and subgroup results. Numeric embeddings, width/depth changes,
`pos_weight` ablation, calibration, and TabM have not been applied in this run.

### Combined Transformer: 64-wide, two layers, numerical PLE

The completed combined run uses numerical missingness indicators, fold-local
16-bin piecewise-linear embeddings for the original 68 numerical features,
token width 64, and two Transformer encoder layers. Binary missingness flags
keep their linear tokens. All 307,511 rows were evaluated in the same five
stratified folds on GPU, with a 30-epoch limit and early stopping.

```powershell
.\train_transformer_64_ple.ps1
```

OOF ROC-AUC is **0.75938**, OOF PR-AUC is **0.23907**. This is the preferred
combined Transformer configuration under the local ROC-AUC objective. The
width-128 variant was tested separately but is not used by this command.
See `artifacts/transformer_w64_d2_ple_gpu/REPORT.md` for comparisons, fold
results, and caveats. Its Kaggle Public/Private ROC-AUC scores are
**0.76267/0.75680**, the best Private result among the submitted DL runs.

### TabM candidate

TabM is a parameter-efficient ensemble of MLP-like predictors: 32 related
submodels share most weights, train together using a loss for each prediction,
and average their probabilities at inference. The project uses the official
`tabm` package through a single-matrix adapter in `model_tabm.py`. Its
fold-local preprocessing includes missingness indicators and 16-bin numerical
PLE embeddings (including binary indicators), with categorical one-hot inputs.

```powershell
.\train_tabm.ps1
```

The completed five-fold GPU run achieved **0.76158 OOF ROC-AUC** and
**0.24342 OOF PR-AUC** in 15.70 minutes, versus 0.75938 / 0.23907 for the
retained Transformer. It remains below the local LightGBM reference of
0.7639 / 0.2479. See `artifacts/tabm_30_epochs_gpu/REPORT.md` for the precise
comparison, fold scores, and limitations. Its Kaggle Public/Private ROC-AUC
scores are **0.76126/0.75641**.

### TabM dropout ablation

The same five-fold TabM configuration was rerun with dropout 0.2 instead of
0.1, changing no other setting. Pooled OOF ROC-AUC rose from 0.76158 to
0.76240 and PR-AUC from 0.24342 to 0.24422, but mean fold ROC-AUC changed
only from 0.76282 to 0.76292 and three folds were slightly lower. This is an
exploratory, small gain rather than firm evidence that 0.2 generalizes better.
The original `train_tabm.ps1` remains at 0.1; the 0.2 experiment and its
submission are in `artifacts/tabm_dropout_02_gpu/`. Reproduce that candidate
with `train_tabm_dropout_02.ps1`.

### Longer Transformer early stopping

The 64-wide, two-layer PLE Transformer was rerun with a 100-epoch maximum and
patience 15, changing no other training argument. Early stopping ended the
folds after 23-33 epochs. Its OOF ROC-AUC was **0.75917** and PR-AUC
**0.23852**, slightly below the prior 30-epoch/patience-5 run (0.75938 /
0.23907), while GPU time increased from 22.48 to 37.83 minutes. Keep the
shorter setting. Details and the reproduction command are in
`artifacts/transformer_w64_d2_ple_100ep_p15_gpu/EXPERIMENT.md`.
