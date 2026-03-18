# SSRL: Self-Supervised ECG Risk Prediction

This repository contains a comprehensive pipeline for self-supervised learning applied to ECG-based cardiovascular risk prediction, targeting resource-constrained clinical settings. Designed for publication at venues like Deep Learning Indaba.

## Features

- **Self-supervised pretraining** on PTB-XL with two strategies:
  - Masked signal reconstruction (primary method)
  - Contrastive learning (NT-Xent loss, alternative baseline)
- **Supervised deep learning baselines**:
  - CNN trained from scratch
- **Traditional ML baselines**:
  - Random Forest and XGBoost with handcrafted ECG features
- **Label-efficient fine-tuning** with simulated label scarcity (1%, 5%, 10%, 25%, 100%)
- **Cross-dataset transfer** evaluation to MIT-BIH Arrhythmia dataset
- **Robustness evaluation** under realistic noise and signal masking
- **Publication-ready visualizations** (ROC, label efficiency, calibration, robustness)

## 1. Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

The `-e` (editable) flag installs the package with all dependencies in development mode, resolving module import issues.

## 2. Dataset Expectations

Expected folder structure:

```
data/
  PTB-XL/
    ptbxl_database.csv
    scp_statements.csv
    records100/
      00000/
        *.hea, *.dat
      ...
    records500/
      ...
  MIT-BIH/
    files/mitdb/1.0.0/
      *.hea, *.dat, *.atr
```

**PTB-XL** (primary dataset):
- 21,799 ECGs from 18,869 patients
- 5 diagnostic superclasses: `NORM` (9,514), `MI` (5,469), `STTC` (5,235), `HYP` (2,649), `CD` (4,898)
- Pre-split into 10 folds; used as train (1-8), val (9), test (10)

**MIT-BIH** (secondary for transfer validation):
- 48 records with arrhythmia annotations
- Used as external robustness check

## 3. Quick Start: SSL Pretraining

### Masked Reconstruction (Recommended)

```powershell
python -m ssrl_ecg.train_ssl `
  --data-root data/PTB-XL `
  --epochs 20 `
  --batch-size 256 `
  --mask-ratio 0.3 `
  --seed 42 `
  --out checkpoints/ssl_masked.pt
```

### Contrastive Learning (Alternative Baseline)

```powershell
python -m ssrl_ecg.train_ssl_contrastive `
  --data-root data/PTB-XL `
  --epochs 20 `
  --batch-size 64 `
  --temperature 0.07 `
  --seed 42 `
  --out checkpoints/ssl_contrastive.pt
```

## 4. Supervised Baselines

### Deep Learning CNN (From Scratch)

```powershell
python -m ssrl_ecg.train_supervised `
  --data-root data/PTB-XL `
  --epochs 20 `
  --batch-size 128 `
  --label-fraction 0.1 `
  --seed 42 `
  --out checkpoints/supervised_10pct.pt
```

### Traditional ML (RF/XGBoost)

Requires `scikit-learn` and optionally `xgboost`:

```powershell
python -m ssrl_ecg.train_traditional_ml `
  --data-root data/PTB-XL `
  --label-fraction 0.1 `
  --model rf `
  --seed 42
```

Options: `--model rf` or `--model xgb`

## 5. SSL Fine-Tuning (Main Experiment)

### From Masked Pre-training

```powershell
python -m ssrl_ecg.train_finetune `
  --data-root data/PTB-XL `
  --ssl-checkpoint checkpoints/ssl_masked.pt `
  --epochs 20 `
  --batch-size 128 `
  --label-fraction 0.1 `
  --seed 42 `
  --out checkpoints/ssl_finetuned_10pct.pt
```

### Frozen Encoder Ablation

```powershell
python -m ssrl_ecg.train_finetune `
  --data-root data/PTB-XL `
  --ssl-checkpoint checkpoints/ssl_masked.pt `
  --epochs 20 `
  --batch-size 128 `
  --label-fraction 0.1 `
  --freeze-encoder `
  --seed 42 `
  --out checkpoints/ssl_finetuned_frozen_10pct.pt
```

## 6. Cross-Dataset Transfer (MIT-BIH)

Evaluate a PTB-XL trained model on MIT-BIH (binary normal vs. abnormal):

```powershell
python -m ssrl_ecg.transfer_mitbih `
  --mitbih-root data/MIT-BIH/files/mitdb/1.0.0 `
  --checkpoint checkpoints/ssl_finetuned_10pct.pt `
  --use-ssl
```

## 7. Robustness Evaluation

Evaluate performance under realistic degradation:

```powershell
# Clean test
python -m ssrl_ecg.evaluate `
  --data-root data/PTB-XL `
  --checkpoint checkpoints/ssl_finetuned_10pct.pt

# With additive Gaussian noise (std=0.1)
python -m ssrl_ecg.evaluate `
  --data-root data/PTB-XL `
  --checkpoint checkpoints/ssl_finetuned_10pct.pt `
  --noise-std 0.1

# With 20% signal masking
python -m ssrl_ecg.evaluate `
  --data-root data/PTB-XL `
  --checkpoint checkpoints/ssl_finetuned_10pct.pt `
  --mask-ratio 0.2
```

## 8. Dataset Analysis

Print summary statistics for both datasets:

```powershell
python -m ssrl_ecg.analyze_datasets `
  --ptbxl-root data/PTB-XL `
  --mitbih-root data/MIT-BIH/files/mitdb/1.0.0
```

## 9. Recommended Experiment Matrix for Paper

| Model | Label % | Focus |
|---|---|---|
| Supervised CNN | 1, 5, 10, 25, 100 | Baseline; shows label hunger |
| SSL + Fine-tune | 1, 5, 10, 25, 100 | Main result; shows label efficiency gain |
| Traditional ML (RF) | 1, 5, 10, 25, 100 | Sanity check; classical approach |

For each:
- Run 3–5 seeds (42, 52, 62, ...)
- Report mean ± std for AUROC, F1, sensitivity, specificity
- Use canonical PTB-XL fold split (train: folds 1–8, val: fold 9, test: fold 10)

## 10. Publication Figures

The `visualization` module provides publication-ready plotting utilities:

```python
from ssrl_ecg.visualization import (
    set_publication_style,
    plot_roc_curve,
    plot_label_efficiency,
    plot_robustness_comparison,
    plot_confusion_matrix,
    plot_signal_examples,
)

set_publication_style()

# Example: label efficiency plot
plot_label_efficiency(
    label_fractions=[0.01, 0.05, 0.1, 0.25, 1.0],
    supervised_auroc=[0.62, 0.70, 0.75, 0.80, 0.85],
    ssl_auroc=[0.70, 0.78, 0.82, 0.86, 0.88],
    output_path="figures/label_efficiency.png"
)
```

Generates:
- ROC curves (with AUC)
- Label efficiency comparison (AUROC vs. label fraction)
- Robustness under noise/masking
- Confusion matrices
- Example ECG signals (clean, noisy, masked)

## 11. Key Files and Modules

### Data Loading and Preprocessing
- [src/ssrl_ecg/data/ptbxl.py](src/ssrl_ecg/data/ptbxl.py) — PTB-XL dataset, splits, label construction
- [src/ssrl_ecg/data/mitbih.py](src/ssrl_ecg/data/mitbih.py) — MIT-BIH dataset loader

### Models
- [src/ssrl_ecg/models/cnn.py](src/ssrl_ecg/models/cnn.py) — 1D CNN encoder, SSL reconstruction, classifier

### Training Scripts
- [src/ssrl_ecg/train_ssl.py](src/ssrl_ecg/train_ssl.py) — Masked reconstruction pretraining
- [src/ssrl_ecg/train_ssl_contrastive.py](src/ssrl_ecg/train_ssl_contrastive.py) — Contrastive SSL (NT-Xent)
- [src/ssrl_ecg/train_supervised.py](src/ssrl_ecg/train_supervised.py) — Supervised CNN baseline
- [src/ssrl_ecg/train_traditional_ml.py](src/ssrl_ecg/train_traditional_ml.py) — RF/XGBoost with handcrafted features
- [src/ssrl_ecg/train_finetune.py](src/ssrl_ecg/train_finetune.py) — Fine-tuning from SSL or supervised encoders

### Evaluation
- [src/ssrl_ecg/evaluate.py](src/ssrl_ecg/evaluate.py) — Robustness evaluation (noise, masking)
- [src/ssrl_ecg/transfer_mitbih.py](src/ssrl_ecg/transfer_mitbih.py) — Cross-dataset transfer evaluation

### Utilities and Visualization
- [src/ssrl_ecg/utils.py](src/ssrl_ecg/utils.py) — Commons (seeding, device selection, metrics, augmentation)
- [src/ssrl_ecg/visualization.py](src/ssrl_ecg/visualization.py) — Publication-ready figures
- [src/ssrl_ecg/analyze_datasets.py](src/ssrl_ecg/analyze_datasets.py) — Dataset statistics

## 12. Configuration and Experiment Plans

See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for detailed protocol, result templates, and paper guidance.

## 13. Troubleshooting

**ModuleNotFoundError: No module named 'ssrl_ecg'**

Ensure you installed the package with:
```powershell
pip install -e .
```
(Not just `pip install -r requirements.txt`)

