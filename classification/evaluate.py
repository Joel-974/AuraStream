"""
Evaluation Script for AuraStream Classification Models.
Computes confusion matrix, macro-F1, per-class recall/precision, and writes an auditable report.
100% Pure-Python.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.dataset import get_dataset_splits
from classification.config import MODELS_DIR
from classification.baseline_model import PureNaiveBayesClassifier
from classification.metrics import compute_classification_metrics


def evaluate_model(task: str = "humanitarian") -> dict:
    """
    Evaluate trained model on the held-out test split.
    """
    model_path = MODELS_DIR / f"baseline_{task}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_baseline.py first.")

    model = PureNaiveBayesClassifier.load(model_path)
    _, _, test_records = get_dataset_splits(task=task)

    X_test = [r["text"] for r in test_records]
    y_test = [r["label"] for r in test_records]

    y_pred = [model.predict_single(t)[0] for t in X_test]
    metrics = compute_classification_metrics(y_test, y_pred)

    result = {
        "task": task,
        "test_sample_count": len(y_test),
        "classes": metrics["classes"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_avg"]["f1-score"],
        "weighted_f1": metrics["weighted_avg"]["f1-score"],
        "confusion_matrix": metrics["confusion_matrix"],
        "per_class_metrics": metrics["per_class"],
    }

    report_path = MODELS_DIR / f"evaluation_{task}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n=======================================================")
    print(f"       EVALUATION REPORT: {task.upper()} TASK           ")
    print(f"=======================================================")
    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Macro-F1:     {metrics['macro_avg']['f1-score']:.4f}")
    print(f"Weighted-F1:  {metrics['weighted_avg']['f1-score']:.4f}")
    print(f"Report saved to: {report_path}")

    return result


def main():
    evaluate_model("humanitarian")
    evaluate_model("informative")


if __name__ == "__main__":
    main()
