"""
Fine-tuning script for Transformer-based disaster report classification in AuraStream.
Supports DistilBERT / RoBERTa fine-tuning with validation monitoring and held-out test evaluation.
"""

import argparse
import json
import time
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.dataset import get_dataset_splits
from classification.config import MODELS_DIR, RANDOM_SEED


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def train_transformer(
    task: str = "humanitarian",
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    max_train_samples: int = None,
    output_dir: Path = None,
):
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )

    output_dir = output_dir or (MODELS_DIR / "transformer" / task)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"   FINE-TUNING TRANSFORMER: {model_name} ({task.upper()}) ")
    print(f"=======================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution Device: {device.upper()}")
    if device == "cpu":
        print("Note: Running on CPU. For fast training, a small max_train_samples limit or GPU is recommended.")

    train_records, dev_records, test_records = get_dataset_splits(task=task)

    if max_train_samples and max_train_samples < len(train_records):
        print(f"Subsampling training set to {max_train_samples} samples for resource constraints...")
        import random
        random.seed(RANDOM_SEED)
        train_records = random.sample(train_records, max_train_samples)

    labels = sorted(list(set(r["label"] for r in train_records)))
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, records, label_map, tokenizer, max_length=128):
            texts = [r["text"] for r in records]
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=max_length,
            )
            self.labels = [label_map[r["label"]] for r in records]

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = TextDataset(train_records, label2id, tokenizer)
    dev_dataset = TextDataset(dev_records, label2id, tokenizer)
    test_dataset = TextDataset(test_records, label2id, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=RANDOM_SEED,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    start_time = time.time()
    print("Beginning fine-tuning...")
    trainer.train()
    training_duration = time.time() - start_time
    print(f"Fine-tuning completed in {training_duration:.2f} seconds.")

    # Evaluate on held-out test set
    print("\nEvaluating fine-tuned model on held-out test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Set Evaluation Results: {json.dumps(test_results, indent=2)}")

    # Save finalized model and tokenizer
    print(f"Saving fine-tuned model to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save metadata
    metadata = {
        "model_name": model_name,
        "task": task,
        "classes": labels,
        "label2id": label2id,
        "id2label": id2label,
        "epochs": epochs,
        "training_time_seconds": round(training_duration, 2),
        "test_results": test_results,
    }
    with open(output_dir / "transformer_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Transformer on CrisisMMD")
    parser.add_argument("--task", type=str, default="humanitarian", choices=["humanitarian", "informative"])
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_train_samples", type=int, default=None)
    args = parser.parse_args()

    train_transformer(
        task=args.task,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
    )


if __name__ == "__main__":
    main()
