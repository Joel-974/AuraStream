"""
Baseline Classification Model Training for AuraStream.
Trains Pure-Python TF-IDF + Multinomial Naive Bayes on CrisisMMD dataset.
Saves model artifacts, label mappings, and training metrics as clean, portable JSON.
"""

import json
import time
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.dataset import get_dataset_splits
from classification.config import MODELS_DIR, RANDOM_SEED
from classification.baseline_model import PureNaiveBayesClassifier
from classification.metrics import compute_classification_metrics


def train_baseline_pipeline(task: str = "humanitarian") -> dict:
    """
    Train TF-IDF + Naive Bayes baseline for disaster report classification.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"baseline_{task}.json"
    metadata_path = MODELS_DIR / f"baseline_{task}_metadata.json"

    print(f"\n=======================================================")
    print(f"   TRAINING BASELINE MODEL: {task.upper()} TASK         ")
    print(f"=======================================================")

    start_time = time.time()
    train_records, dev_records, test_records = get_dataset_splits(task=task)

    X_train = [r["text"] for r in train_records]
    y_train = [r["label"] for r in train_records]

    X_dev = [r["text"] for r in dev_records]
    y_dev = [r["label"] for r in dev_records]

    X_test = [r["text"] for r in test_records]
    y_test = [r["label"] for r in test_records]

    classes = sorted(list(set(y_train)))
    print(f"Dataset split: Train={len(X_train)}, Dev={len(X_dev)}, Test={len(X_test)}")
    print(f"Number of classes: {len(classes)}")

    model = PureNaiveBayesClassifier(alpha=1.0, balanced_priors=True)
    print("Fitting TF-IDF and Naive Bayes on training data...")
    model.fit(X_train, y_train, max_features=4000)
    training_duration = time.time() - start_time
    print(f"Training completed in {training_duration:.2f} seconds.")

    # Validation evaluation
    y_dev_pred = [model.predict_single(t)[0] for t in X_dev]
    dev_metrics = compute_classification_metrics(y_dev, y_dev_pred)

    # Test evaluation
    y_test_pred = [model.predict_single(t)[0] for t in X_test]
    test_metrics = compute_classification_metrics(y_test, y_test_pred)

    print(f"\n[Held-out Test Set Performance]")
    print(f"  * Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"  * Macro-F1:     {test_metrics['macro_avg']['f1-score']:.4f}")
    print(f"  * Weighted-F1:  {test_metrics['weighted_avg']['f1-score']:.4f}")

    print("\nDetailed Per-Class Performance:")
    print(f"{'Class':<40} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 84)
    for cls_name, vals in test_metrics["per_class"].items():
        print(f"{cls_name:<40} {vals['precision']:>10.4f} {vals['recall']:>10.4f} {vals['f1-score']:>10.4f} {vals['support']:>10}")

    # Save artifact
    model.save(model_path)
    print(f"\nSaved trained baseline artifact to: {model_path}")

    metadata = {
        "model_type": "TF-IDF + Balanced Naive Bayes",
        "task": task,
        "classes": classes,
        "random_seed": RANDOM_SEED,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "training_time_seconds": round(training_duration, 2),
        "metrics": {
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_avg"]["f1-score"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_avg"]["f1-score"],
            "test_weighted_f1": test_metrics["weighted_avg"]["f1-score"],
        },
        "per_class_metrics": test_metrics["per_class"],
        "model_file": str(model_path.name),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    return metadata


def main():
    train_baseline_pipeline(task="humanitarian")
    train_baseline_pipeline(task="informative")


if __name__ == "__main__":
    main()
