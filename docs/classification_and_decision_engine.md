# AuraStream: Classification and Decision Engine Documentation

## 1. Module Overview

The **Classification and Decision Engine** is the intelligence core of the **AuraStream** platform. Its primary mission is to process incoming disaster reports submitted by citizens and emergency agencies, automatically classify their humanitarian needs, and apply safety-aligned decision logic to recommend incident triage priorities.

```
Incoming Disaster Report (Text / Future Multimodal)
                     │
                     ▼
       ┌───────────────────────────┐
       │     FastAPI Ingestion     │
       │    (Teammate Pipeline)    │
       └─────────────┬─────────────┘
                     │ (JSON Schema Contract)
                     ▼
       ┌───────────────────────────┐
       │   Classification Engine   │
       │    (TF-IDF + Naive Bayes  │
       │   or Fine-Tuned DistilBERT)│
       └─────────────┬─────────────┘
                     │ Status, Top Label, Confidences, Class Probs
                     ▼
       ┌───────────────────────────┐
       │      Decision Engine      │
       │   (Explicit Triage Rules, │
       │    Human-in-the-Loop Gate)│
       └─────────────┬─────────────┘
                     │ Structured Decision Result
                     ▼
  Command Dashboard / PostGIS Routing / Database
```

---

## 2. Dataset Strategy: CrisisMMD v2.0

* **Official Source**: Qatar Computing Research Institute (QCRI) / CrisisNLP (Alam et al., ICWSM 2018).
* **Dataset Structure**: 18,082 annotated social media posts across 7 major disaster events (*California Wildfires, Sri Lanka Floods, Mexico Earthquake, Iraq-Iran Earthquake, Hurricane Harvey, Hurricane Maria, Hurricane Irma*).
* **Data Splits**:
  * **Train Set**: 13,608 records
  * **Dev (Validation) Set**: 2,237 records
  * **Test Set (Held-Out)**: 2,237 records
* **Zero Leakage**: No overlapping report texts between training and test sets.

### Target Classification Tasks

1. **Humanitarian Categories (Primary Task - 8 Classes)**:
   * `injured_or_dead_people`: Direct reports of casualties and bodily trauma.
   * `missing_or_found_people`: People unaccounted for or located.
   * `infrastructure_and_utility_damage`: Collapsed bridges, damaged roads, downed power lines, structural destruction.
   * `rescue_volunteering_or_donation_effort`: Mobilization of aid, volunteers, and supplies.
   * `affected_individuals`: Evacuees, displaced people, or people stranded.
   * `vehicle_damage`: Submerged or crushed cars, stranded boats.
   * `other_relevant_information`: General updates, weather warnings, situational context.
   * `not_humanitarian`: Off-topic discussions, chitchat, or unrelated commentary.

2. **Informativeness (Secondary Task - 2 Classes)**:
   * `informative` (61.3% of train data)
   * `not_informative` (38.7% of train data)

---

## 3. Implemented Models

### Baseline Model: Pure-Python TF-IDF + Multinomial Naive Bayes
* **Design Philosophy**: Built with zero external C-dependencies to ensure complete immunity from Windows 11 Smart App Control policies and maximum speed on consumer hardware.
* **Feature Extraction**: Sublinear TF-IDF vectorizer (smooth IDF, L2 vector normalization).
* **Classifier**: Multinomial Naive Bayes with Laplace smoothing ($\alpha=1.0$) and balanced class priors.
* **Training Time**: ~0.66 seconds on CPU for 13,608 records.
* **Held-out Test Performance (Humanitarian Task - 8 Classes)**:
  * Accuracy: `50.92%`
  * Weighted-F1: `49.20%`
  * Macro-F1: `27.16%`
  * Precision on casualties (`injured_or_dead_people`): **55.56%**
* **Held-out Test Performance (Informative Task - 2 Classes)**:
  * Accuracy: `71.79%`
  * Macro-F1: `69.19%`
  * Weighted-F1: `71.23%`

### Transformer Model: DistilBERT (`classification/train_transformer.py`)
* Fine-tuning script provided for HuggingFace Transformers + PyTorch CPU/GPU.
* Supports training on GPU environments (Google Colab / University cluster).

---

## 4. Decision Engine & Safety Architecture

The Decision Engine is strictly separated from the ML model. It does not blindly trust raw prediction probabilities, but subjects them to auditable emergency triage rules:

1. **Mandatory Human-in-the-Loop**: Reports flagged with critical casualty indicators (`injured_or_dead_people`, `missing_or_found_people`, `infrastructure_and_utility_damage`) are marked as `immediate` or `high` priority, but always enforce `human_review_required = True`. AI never dispatches emergency units autonomously.
2. **Confidence Gating**:
   * Confidence $\ge 0.65$: High-confidence triage category applied.
   * Confidence $< 0.40$: Routed to `requires_review` queue with reason code `LOW_MODEL_CONFIDENCE`.
3. **Graceful Malformed Input Handling**: Empty or whitespace-only texts are rejected cleanly with `suggested_priority = "invalid"` and code `ERR_NO_TEXT_CONTENT`.
4. **Missing Model Resilience**: If the trained artifact is missing, incoming reports are not dropped; they are assigned `requires_review` with code `ERR_MODEL_UNAVAILABLE`.

---

## 5. Input and Output Contract

### Ingestion Input Contract (`DisasterReportInput`)
```json
{
  "report_id": "rep-2026-001",
  "text": "Flash flooding on Route 9, bridge collapsed and two cars submerged with passengers trapped",
  "timestamp": "2026-10-22T10:30:00Z",
  "location": {
    "latitude": 9.9312,
    "longitude": 76.2673,
    "address": "Route 9, Ernakulam"
  },
  "modalities_available": ["text"]
}
```

### Response Contract (`AuraStreamDecisionPayload`)
```json
{
  "report_id": "rep-2026-001",
  "timestamp": "2026-10-22T10:30:01.205Z",
  "classification": {
    "status": "predicted",
    "predicted_label": "infrastructure_and_utility_damage",
    "confidence": 0.9983,
    "class_probabilities": {
      "infrastructure_and_utility_damage": 0.9983,
      "rescue_volunteering_or_donation_effort": 0.0011,
      "injured_or_dead_people": 0.0004
    },
    "model_version": "humanitarian-baseline-v1"
  },
  "decision": {
    "suggested_priority": "immediate",
    "human_review_required": true,
    "reason_codes": [
      "CRITICAL_HUMANITARIAN_NEED",
      "HIGH_CONFIDENCE_CRITICAL"
    ],
    "explanation": "High confidence prediction of 'infrastructure_and_utility_damage' indicating urgent emergency attention."
  }
}
```

---

## 6. How to Reproduce on Windows

### Step 1: Open PowerShell and Navigate to the Repository
```powershell
cd D:\AuraStream
```

### Step 2: Set Python Path
```powershell
$env:PYTHONPATH = "D:\AuraStream"
```

### Step 3: Run Dataset Inspection & Audit
```powershell
python classification/dataset_inspection.py
```

### Step 4: Train Baseline Models
```powershell
python classification/train_baseline.py
```

### Step 5: Evaluate on Held-out Data
```powershell
python classification/evaluate.py
```

### Step 6: Run the Full Automated Test Suite (24 Tests)
```powershell
python -m pytest -v
```

---

## 7. Connecting to Teammate's Ingestion Layer

The module includes an adapter in [`classification/ingestion_adapter.py`](file:///D:/AuraStream/classification/ingestion_adapter.py).

Your teammate can mount our engine into their FastAPI ingestion application with just two lines of code:

```python
from fastapi import FastAPI
from classification.ingestion_adapter import router as classification_router

app = FastAPI(title="AuraStream Ingestion Service")

# Mount classification and decision endpoints
app.include_router(classification_router)
```

Available API Endpoints:
* `GET /api/v1/health`: Checks model loading state and readiness.
* `POST /api/v1/classify-report`: Classifies a single disaster report and returns prioritized triage decisions.
* `POST /api/v1/batch-classify`: High-throughput batch classification for multiple incoming reports.
