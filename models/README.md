# AuraStream Model Artifacts Directory

This directory contains trained model pipelines, label mappings, and test evaluation reports.

## Model Files
* `baseline_humanitarian.json`: Pure-Python trained model parameters and vocabulary table for the 8-class humanitarian need task. Portable and immune to Windows Smart App Control.
* `baseline_humanitarian_metadata.json`: Model version, training parameters, duration, and test set performance.
* `baseline_informative.json`: Pure-Python trained model for 2-class informativeness classification.
* `baseline_informative_metadata.json`: Model metadata and test metrics for informativeness.
* `evaluation_humanitarian.json`: Held-out test set confusion matrix and detailed per-class metrics.
* `evaluation_informative.json`: Held-out test set evaluation for informativeness.
* `baseline_*.joblib`: Scikit-learn binary pipelines (available for Linux/container environments).

## Reproducing Model Artifacts
Run the following command from the repository root:
```powershell
python classification/train_baseline.py
```
Model files are tracked locally and gitignored from Git pushes to keep the repository lightweight.
