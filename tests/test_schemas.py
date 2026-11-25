"""
Unit tests for AuraStream Pydantic schemas.
"""

import pytest
from datetime import datetime
from classification.schemas import (
    DisasterReportInput,
    LocationModel,
    ClassificationResult,
    DecisionResult,
    AuraStreamDecisionPayload,
)


def test_valid_disaster_report_input():
    report = DisasterReportInput(
        report_id="rep-101",
        text="Bridge on river 4 collapsed due to flash floods",
        location=LocationModel(latitude=28.6139, longitude=77.2090),
        modalities_available=["text"],
    )
    assert report.report_id == "rep-101"
    assert "Bridge" in report.text
    assert report.location.latitude == 28.6139
    assert isinstance(report.timestamp, datetime)


def test_empty_report_id_rejected():
    with pytest.raises(ValueError):
        DisasterReportInput(report_id="   ", text="Flooding in downtown")


def test_optional_location():
    report = DisasterReportInput(report_id="rep-102", text="Heavy rains continuing")
    assert report.location is None


def test_invalid_latitude():
    with pytest.raises(ValueError):
        DisasterReportInput(
            report_id="rep-103",
            text="Test",
            location=LocationModel(latitude=120.0, longitude=50.0),
        )


def test_classification_result_validation():
    res = ClassificationResult(
        status="predicted",
        predicted_label="injured_or_dead_people",
        confidence=0.88,
        class_probabilities={"injured_or_dead_people": 0.88, "not_humanitarian": 0.12},
        model_version="test-v1",
    )
    assert res.status == "predicted"
    assert res.confidence == 0.88


def test_decision_result_structure():
    dec = DecisionResult(
        suggested_priority="immediate",
        human_review_required=True,
        reason_codes=["CRITICAL_HUMANITARIAN_NEED"],
        explanation="Urgent medical dispatch review",
    )
    assert dec.suggested_priority == "immediate"
    assert dec.human_review_required is True
    assert "CRITICAL_HUMANITARIAN_NEED" in dec.reason_codes
