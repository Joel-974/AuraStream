"""
Dataset loading and preparation pipeline for AuraStream.
Fetches, caches, validates, and prepares CrisisMMD text annotations for model training.
Uses pure-Python data structures for high performance and compatibility across all platforms.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import requests

from classification.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CRISISMMD_BASE_URL,
    CRISISMMD_SPLITS,
    CRISISMMD_TASKS,
    RANDOM_SEED,
)


def ensure_directories():
    """Ensure data directories exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_split_if_missing(task: str = "humanitarian", split: str = "train", force_download: bool = False) -> Path:
    """
    Download and cache the official CrisisMMD annotation split JSON file.
    Does not download heavy image binaries; downloads text annotations and labels.
    """
    ensure_directories()
    task_dir = RAW_DATA_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)
    target_path = task_dir / f"{split}.json"

    if target_path.exists() and not force_download:
        return target_path

    url = f"{CRISISMMD_BASE_URL}/{task}/{split}.json"
    print(f"Downloading official CrisisMMD split [{task}/{split}] from {url}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # Validate that it is valid JSON
    data = response.json()
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} records to {target_path}")
    return target_path


def load_split_records(task: str = "humanitarian", split: str = "train") -> List[Dict]:
    """
    Load a dataset split into a standardized list of record dictionaries.
    Standardizes keys: 'text', 'label', 'event_name'.
    """
    file_path = download_split_if_missing(task=task, split=split)
    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    cleaned = []
    for r in records:
        text = r.get("tweet_text") or r.get("text") or ""
        label = r.get("label") or ""
        text = str(text).strip()
        label = str(label).strip()
        cleaned.append({
            "text": text,
            "label": label,
            "event_name": r.get("event_name", ""),
            "tweet_id": str(r.get("tweet_id", "")),
        })

    return cleaned


def get_dataset_splits(task: str = "humanitarian") -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load train, dev (validation), and test sets for a specific task.
    Filters out empty records and verifies label validity.
    """
    train_records = load_split_records(task=task, split="train")
    dev_records = load_split_records(task=task, split="dev")
    test_records = load_split_records(task=task, split="test")

    valid_labels = set(CRISISMMD_TASKS.get(task, []))

    filtered_splits = []
    for records in [train_records, dev_records, test_records]:
        clean = []
        for r in records:
            if not r["text"] or not r["label"]:
                continue
            if valid_labels and r["label"] not in valid_labels:
                continue
            clean.append(r)
        filtered_splits.append(clean)

    return filtered_splits[0], filtered_splits[1], filtered_splits[2]
