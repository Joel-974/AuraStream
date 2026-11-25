"""
Pure-Python Evaluation Metrics for Classification.
Computes Confusion Matrix, Accuracy, Precision, Recall, Macro-F1, and Weighted-F1.
Zero dependencies on external C-binaries.
"""

from collections import Counter
from typing import Dict, List, Tuple


def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """
    Compute full classification report and confusion matrix.
    """
    total = len(y_true)
    if total == 0:
        return {}

    classes = sorted(list(set(y_true).union(set(y_pred))))
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total

    # Confusion matrix
    class_indices = {cls_name: i for i, cls_name in enumerate(classes)}
    cm = [[0] * len(classes) for _ in range(len(classes))]
    for yt, yp in zip(y_true, y_pred):
        cm[class_indices[yt]][class_indices[yp]] += 1

    per_class = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_f1 = 0.0

    true_counts = Counter(y_true)

    for cls_name in classes:
        idx = class_indices[cls_name]
        tp = cm[idx][idx]
        fp = sum(cm[r][idx] for r in range(len(classes)) if r != idx)
        fn = sum(cm[idx][c] for c in range(len(classes)) if c != idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = true_counts.get(cls_name, 0)

        per_class[cls_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1-score": round(f1, 4),
            "support": support,
        }

        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        weighted_f1 += f1 * (support / total)

    num_classes = len(classes)
    macro_p /= num_classes
    macro_r /= num_classes
    macro_f1 /= num_classes

    return {
        "accuracy": round(accuracy, 4),
        "macro_avg": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1-score": round(macro_f1, 4),
        },
        "weighted_avg": {
            "f1-score": round(weighted_f1, 4),
        },
        "per_class": per_class,
        "classes": classes,
        "confusion_matrix": cm,
    }
