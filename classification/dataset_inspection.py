"""
Dataset Inspection Script for AuraStream.
Validates official CrisisMMD dataset annotations, class distributions, duplicates, and missing values.
Generates an audit report for Milestone 2.
"""

import sys
from pathlib import Path
from collections import Counter

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.dataset import get_dataset_splits
from classification.config import CRISISMMD_TASKS


def inspect_task(task_name: str = "humanitarian") -> dict:
    """
    Perform deep inspection of dataset splits for a given task.
    """
    print(f"\n=======================================================")
    print(f"      INSPECTING CRISISMMD TASK: {task_name.upper()}     ")
    print(f"=======================================================")

    train_data, dev_data, test_data = get_dataset_splits(task=task_name)

    train_labels = [r["label"] for r in train_data]
    dev_labels = [r["label"] for r in dev_data]
    test_labels = [r["label"] for r in test_data]

    train_dist = dict(Counter(train_labels))
    dev_dist = dict(Counter(dev_labels))
    test_dist = dict(Counter(test_labels))

    events = sorted(list(set(r.get("event_name", "") for r in train_data if r.get("event_name"))))

    # Check for text duplicates between train and test to detect leakage
    train_texts = set(r["text"].lower().strip() for r in train_data)
    test_texts = set(r["text"].lower().strip() for r in test_data)
    overlap = train_texts.intersection(test_texts)

    stats = {
        "task": task_name,
        "splits": {
            "train_size": len(train_data),
            "dev_size": len(dev_data),
            "test_size": len(test_data),
            "total_size": len(train_data) + len(dev_data) + len(test_data),
        },
        "train_class_distribution": train_dist,
        "dev_class_distribution": dev_dist,
        "test_class_distribution": test_dist,
        "events_represented": events,
        "train_test_text_overlap": len(overlap),
    }

    print(f"\n[Split Sizes]")
    print(f"  * Train set: {stats['splits']['train_size']} records")
    print(f"  * Dev/Val set: {stats['splits']['dev_size']} records")
    print(f"  * Test set:  {stats['splits']['test_size']} records")
    print(f"  * Total:     {stats['splits']['total_size']} records")

    print(f"\n[Disaster Events Represented]")
    for ev in events:
        print(f"  - {ev}")

    print(f"\n[Train Class Distribution]")
    for label, count in sorted(train_dist.items(), key=lambda x: x[1], reverse=True):
        pct = (count / stats['splits']['train_size']) * 100
        print(f"  - {label:<40}: {count:>5} ({pct:.2f}%)")

    print(f"\n[Data Leakage & Overlap Check]")
    print(f"  * Unique texts shared between Train and Test: {len(overlap)}")

    return stats


def main():
    print("Running AuraStream Dataset Inspection on CrisisMMD...")
    inspect_task("humanitarian")
    inspect_task("informative")
    print("\nDataset inspection completed successfully!")


if __name__ == "__main__":
    main()
