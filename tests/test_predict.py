"""
Unit tests for model inference engine, edge cases, and confidence validation.
"""

import math
import pytest
from pydantic import ValidationError

from classification.predict import DisasterClassifier
from classification.schemas import ClassificationResult, DisasterReportInput


def test_classifier_loads():
    classifier = DisasterClassifier(task="humanitarian")
    assert classifier.is_available is True
    assert classifier.pipeline is not None


def test_normal_inference():
    classifier = DisasterClassifier(task="humanitarian")
    report = DisasterReportInput(
        report_id="rep-1",
        text="Paramedics are treating multiple casualties at the building collapse site",
    )
    res = classifier.predict(report)
    assert res.status in ("predicted", "low_confidence")
    assert 0.0 <= res.confidence <= 1.0
    assert res.predicted_label is not None
    assert len(res.class_probabilities) == 8
    # Probabilities should sum closely to 1.0
    prob_sum = sum(res.class_probabilities.values())
    assert abs(prob_sum - 1.0) < 0.05


def test_empty_text_handling():
    classifier = DisasterClassifier(task="humanitarian")
    report = DisasterReportInput(report_id="rep-empty", text="")
    res = classifier.predict(report)
    assert res.status == "no_text"
    assert res.predicted_label is None
    assert res.confidence == 0.0


def test_whitespace_text_handling():
    classifier = DisasterClassifier(task="humanitarian")
    res = classifier.predict("   \n\t  ")
    assert res.status == "no_text"


def test_missing_model_fallback(tmp_path):
    fake_path = tmp_path / "non_existent_model.json"
    classifier = DisasterClassifier(task="humanitarian", model_path=fake_path)
    assert classifier.is_available is False
    res = classifier.predict("Urgent rescue needed")
    assert res.status == "model_unavailable"
    assert res.predicted_label is None


@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        {},
        {"report_id": "rep-no-text"},
        {"text": None},
        12345,
    ],
)
def test_invalid_prediction_inputs(invalid_input):
    classifier = DisasterClassifier(task="humanitarian")
    res = classifier.predict(invalid_input)
    assert res.status == "no_text"
    assert res.predicted_label is None
    assert res.confidence == 0.0


@pytest.mark.parametrize("valid_conf", [0.0, 0.5, 1.0])
def test_confidence_boundary_values(valid_conf):
    res = ClassificationResult(
        status="predicted",
        predicted_label="injured_or_dead_people",
        confidence=valid_conf,
        model_version="test-v1",
    )
    assert res.confidence == valid_conf


@pytest.mark.parametrize("invalid_conf", [-0.1, 1.5, float("nan"), None])
def test_confidence_out_of_range_and_invalid_rejected(invalid_conf):
    with pytest.raises(ValidationError):
        ClassificationResult(
            status="predicted",
            predicted_label="injured_or_dead_people",
            confidence=invalid_conf,
            model_version="test-v1",
        )
