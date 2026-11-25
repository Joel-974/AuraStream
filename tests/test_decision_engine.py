"""
Unit tests for the Decision Engine logic and safety rules.
"""

import pytest
from classification.decision_engine import DecisionEngine
from classification.schemas import ClassificationResult


def test_critical_high_confidence():
    engine = DecisionEngine(high_threshold=0.65, low_threshold=0.40)
    classification = ClassificationResult(
        status="predicted",
        predicted_label="injured_or_dead_people",
        confidence=0.82,
        class_probabilities={"injured_or_dead_people": 0.82},
        model_version="test-v1",
    )
    decision = engine.evaluate(classification)
    assert decision.suggested_priority == "immediate"
    assert decision.human_review_required is True
    assert "CRITICAL_HUMANITARIAN_NEED" in decision.reason_codes
    assert "HIGH_CONFIDENCE_CRITICAL" in decision.reason_codes


def test_critical_moderate_confidence():
    engine = DecisionEngine(high_threshold=0.65, low_threshold=0.40)
    classification = ClassificationResult(
        status="predicted",
        predicted_label="injured_or_dead_people",
        confidence=0.55,
        class_probabilities={"injured_or_dead_people": 0.55},
        model_version="test-v1",
    )
    decision = engine.evaluate(classification)
    assert decision.suggested_priority == "high"
    assert decision.human_review_required is True
    assert "MODERATE_CONFIDENCE_CRITICAL" in decision.reason_codes


def test_low_confidence_triggers_review():
    engine = DecisionEngine(high_threshold=0.65, low_threshold=0.40)
    classification = ClassificationResult(
        status="predicted",
        predicted_label="infrastructure_and_utility_damage",
        confidence=0.32,
        class_probabilities={"infrastructure_and_utility_damage": 0.32},
        model_version="test-v1",
    )
    decision = engine.evaluate(classification)
    assert decision.suggested_priority == "requires_review"
    assert decision.human_review_required is True
    assert "LOW_MODEL_CONFIDENCE" in decision.reason_codes


def test_missing_text_triggers_invalid():
    engine = DecisionEngine()
    classification = ClassificationResult(
        status="no_text",
        predicted_label=None,
        confidence=0.0,
        model_version="test-v1",
    )
    decision = engine.evaluate(classification)
    assert decision.suggested_priority == "invalid"
    assert decision.human_review_required is True
    assert "ERR_NO_TEXT_CONTENT" in decision.reason_codes


def test_model_unavailable_triggers_review():
    engine = DecisionEngine()
    classification = ClassificationResult(
        status="model_unavailable",
        predicted_label=None,
        confidence=0.0,
        model_version="unavailable",
    )
    decision = engine.evaluate(classification)
    assert decision.suggested_priority == "requires_review"
    assert decision.human_review_required is True
    assert "ERR_MODEL_UNAVAILABLE" in decision.reason_codes


def test_error_status_triggers_review():
    engine = DecisionEngine()
    classification = ClassificationResult(
        status="error",
        predicted_label=None,
        confidence=0.0,
        model_version="test-v1",
    )
    decision = engine.evaluate(classification)
    assert decision.suggested_priority == "requires_review"
    assert decision.human_review_required is True
    assert "ERR_ERROR" in decision.reason_codes


def test_contradictory_predicted_none_label():
    engine = DecisionEngine(high_threshold=0.65, low_threshold=0.40)
    classification = ClassificationResult(
        status="predicted",
        predicted_label=None,
        confidence=0.80,
        model_version="test-v1",
    )
    decision = engine.evaluate(classification)
    assert decision.suggested_priority == "requires_review"
    assert decision.human_review_required is True
    assert "UNRECOGNIZED_CATEGORY" in decision.reason_codes


def test_unrecognized_category_triggers_review():
    engine = DecisionEngine(high_threshold=0.65, low_threshold=0.40)
    classification = ClassificationResult(
        status="predicted",
        predicted_label="unknown_category",
        confidence=0.85,
        model_version="test-v1",
    )
    decision = engine.evaluate(classification)
    assert decision.suggested_priority == "requires_review"
    assert decision.human_review_required is True
    assert "UNRECOGNIZED_CATEGORY" in decision.reason_codes


def test_secondary_category_high_and_moderate_confidence():
    engine = DecisionEngine(high_threshold=0.65, low_threshold=0.40)
    high_res = engine.evaluate(
        ClassificationResult(
            status="predicted",
            predicted_label="rescue_volunteering_or_donation_effort",
            confidence=0.75,
            model_version="test-v1",
        )
    )
    assert high_res.suggested_priority == "high"
    assert high_res.human_review_required is True
    assert "SECONDARY_RELIEF_COORDINATION" in high_res.reason_codes

    mod_res = engine.evaluate(
        ClassificationResult(
            status="predicted",
            predicted_label="rescue_volunteering_or_donation_effort",
            confidence=0.50,
            model_version="test-v1",
        )
    )
    assert mod_res.suggested_priority == "medium"
    assert mod_res.human_review_required is True


def test_other_relevant_information_priority():
    engine = DecisionEngine(high_threshold=0.65, low_threshold=0.40)
    high_res = engine.evaluate(
        ClassificationResult(
            status="predicted",
            predicted_label="other_relevant_information",
            confidence=0.70,
            model_version="test-v1",
        )
    )
    assert high_res.suggested_priority == "medium"
    assert high_res.human_review_required is True
    assert "SITUATIONAL_AWARENESS" in high_res.reason_codes

    mod_res = engine.evaluate(
        ClassificationResult(
            status="predicted",
            predicted_label="other_relevant_information",
            confidence=0.50,
            model_version="test-v1",
        )
    )
    assert mod_res.suggested_priority == "low"
    assert mod_res.human_review_required is True


def test_non_humanitarian_high_confidence():
    engine = DecisionEngine(high_threshold=0.65)
    classification = ClassificationResult(
        status="predicted",
        predicted_label="not_humanitarian",
        confidence=0.90,
        class_probabilities={"not_humanitarian": 0.90},
        model_version="test-v1",
    )
    decision = engine.evaluate(classification)
    assert decision.suggested_priority == "low"
    assert decision.human_review_required is False
    assert "NON_HUMANITARIAN_CONTENT" in decision.reason_codes


def test_borderline_non_humanitarian_triggers_review():
    engine = DecisionEngine(high_threshold=0.65, low_threshold=0.40)
    classification = ClassificationResult(
        status="predicted",
        predicted_label="not_humanitarian",
        confidence=0.55,
        class_probabilities={"not_humanitarian": 0.55},
        model_version="test-v1",
    )
    decision = engine.evaluate(classification)
    assert decision.suggested_priority == "requires_review"
    assert decision.human_review_required is True
    assert "BORDERLINE_NON_HUMANITARIAN" in decision.reason_codes


@pytest.mark.parametrize("status", ["no_text", "error", "model_unavailable", "low_confidence"])
def test_uncertain_or_invalid_never_high_or_immediate(status):
    engine = DecisionEngine(high_threshold=0.65, low_threshold=0.40)
    decision = engine.evaluate(
        ClassificationResult(
            status=status,
            predicted_label="injured_or_dead_people",
            confidence=0.90 if status == "low_confidence" else 0.0,
            model_version="test-v1",
        )
    )
    assert decision.suggested_priority not in ("immediate", "high")
    assert decision.human_review_required is True
