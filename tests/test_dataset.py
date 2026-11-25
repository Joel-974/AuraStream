"""
Unit tests for AuraStream dataset preparation module.
"""

import pytest
from classification.dataset import load_split_records, get_dataset_splits
from classification.config import CRISISMMD_TASKS


@pytest.fixture(scope="module")
def humanitarian_splits():
    """Load humanitarian dataset splits once per module."""
    return get_dataset_splits(task="humanitarian")


def test_humanitarian_train_split_loads():
    records = load_split_records(task="humanitarian", split="train")
    assert len(records) > 10000
    assert "text" in records[0]
    assert "label" in records[0]


def test_split_label_validity(humanitarian_splits):
    train_records, dev_records, test_records = humanitarian_splits
    valid_labels = set(CRISISMMD_TASKS["humanitarian"])

    for split_name, records in [
        ("train", train_records),
        ("dev", dev_records),
        ("test", test_records),
    ]:
        split_labels = set(r["label"] for r in records)
        assert split_labels.issubset(valid_labels), (
            f"Unexpected labels in {split_name} split: {split_labels - valid_labels}"
        )


def test_no_empty_texts_or_labels_in_splits(humanitarian_splits):
    train_records, dev_records, test_records = humanitarian_splits
    for records in (train_records, dev_records, test_records):
        for r in records:
            assert len(r["text"].strip()) > 0
            assert len(r["label"].strip()) > 0


def test_no_text_overlap_across_splits(humanitarian_splits):
    train_records, dev_records, test_records = humanitarian_splits
    train_texts = {r["text"].lower().strip() for r in train_records}
    dev_texts = {r["text"].lower().strip() for r in dev_records}
    test_texts = {r["text"].lower().strip() for r in test_records}

    assert len(train_texts & test_texts) == 0
    assert len(train_texts & dev_texts) == 0
    assert len(dev_texts & test_texts) == 0


def test_no_tweet_id_leakage_across_splits(humanitarian_splits):
    train_records, dev_records, test_records = humanitarian_splits
    train_ids = {r["tweet_id"] for r in train_records if r.get("tweet_id")}
    dev_ids = {r["tweet_id"] for r in dev_records if r.get("tweet_id")}
    test_ids = {r["tweet_id"] for r in test_records if r.get("tweet_id")}

    if train_ids and test_ids:
        assert len(train_ids & test_ids) == 0
    if train_ids and dev_ids:
        assert len(train_ids & dev_ids) == 0
    if dev_ids and test_ids:
        assert len(dev_ids & test_ids) == 0


def test_dataset_splitting_is_reproducible(humanitarian_splits):
    train_1, dev_1, test_1 = humanitarian_splits
    train_2, dev_2, test_2 = get_dataset_splits(task="humanitarian")

    assert len(train_1) == len(train_2)
    assert len(dev_1) == len(dev_2)
    assert len(test_1) == len(test_2)
    assert train_1[0] == train_2[0]
    assert dev_1[0] == dev_2[0]
    assert test_1[0] == test_2[0]
