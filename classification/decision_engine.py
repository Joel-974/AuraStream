"""
Decision Engine for AuraStream.
Translates model predictions into actionable triage priorities with explicit rationale.
Enforces safety constraints, provisional thresholds, and human-in-the-loop review.
"""

from typing import List, Optional
from classification.schemas import (
    ClassificationResult,
    DecisionResult,
    DisasterReportInput,
    AuraStreamDecisionPayload,
)
from classification.config import (
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_LOW,
    CRITICAL_CATEGORIES,
    SECONDARY_RESPONSE_CATEGORIES,
)


class DecisionEngine:
    """
    Decoupled rule-based decision engine.
    Applies auditable triage policies on classification outputs.
    """

    def __init__(
        self,
        high_threshold: float = CONFIDENCE_THRESHOLD_HIGH,
        low_threshold: float = CONFIDENCE_THRESHOLD_LOW,
    ):
        # Provisional settings - easy to reconfigure
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def evaluate(
        self,
        classification: ClassificationResult,
        report: Optional[DisasterReportInput] = None,
    ) -> DecisionResult:
        """
        Evaluate classification output and report metadata to generate a structured decision.
        """
        reason_codes: List[str] = []

        # Rule 1: Handle invalid or missing text
        if classification.status == "no_text":
            return DecisionResult(
                suggested_priority="invalid",
                human_review_required=True,
                reason_codes=["ERR_NO_TEXT_CONTENT"],
                explanation="Report contains no textual information to evaluate.",
            )

        # Rule 2: Handle missing or failed model
        if classification.status in ("model_unavailable", "error"):
            return DecisionResult(
                suggested_priority="requires_review",
                human_review_required=True,
                reason_codes=[f"ERR_{classification.status.upper()}"],
                explanation="Model prediction unavailable; report routed directly to operator review.",
            )

        # Rule 3: Handle low-confidence predictions
        if classification.confidence < self.low_threshold or classification.status == "low_confidence":
            return DecisionResult(
                suggested_priority="requires_review",
                human_review_required=True,
                reason_codes=["LOW_MODEL_CONFIDENCE"],
                explanation=f"Top prediction confidence ({classification.confidence:.2f}) is below provisional threshold ({self.low_threshold:.2f}). Requires human verification.",
            )

        label = classification.predicted_label or "unknown"
        conf = classification.confidence

        # Rule 4: Critical human safety / casualty categories
        if label in CRITICAL_CATEGORIES:
            reason_codes.append("CRITICAL_HUMANITARIAN_NEED")
            if conf >= self.high_threshold:
                priority = "immediate"
                reason_codes.append("HIGH_CONFIDENCE_CRITICAL")
                explanation = f"High confidence prediction of '{label}' indicating urgent emergency attention."
            else:
                priority = "high"
                reason_codes.append("MODERATE_CONFIDENCE_CRITICAL")
                explanation = f"Moderate confidence prediction of '{label}'. Elevated priority assigned for operator confirmation."

            # Safety requirement: Urgent reports always require human confirmation before field dispatch
            return DecisionResult(
                suggested_priority=priority,
                human_review_required=True,
                reason_codes=reason_codes,
                explanation=explanation,
            )

        # Rule 5: Secondary response categories (donations, volunteers, affected individuals)
        if label in SECONDARY_RESPONSE_CATEGORIES:
            reason_codes.append("SECONDARY_RELIEF_COORDINATION")
            priority = "high" if conf >= self.high_threshold else "medium"
            return DecisionResult(
                suggested_priority=priority,
                human_review_required=True,
                reason_codes=reason_codes,
                explanation=f"Report classified as '{label}'. Routed for relief supply or volunteer coordination.",
            )

        # Rule 6: General situational awareness
        if label == "other_relevant_information":
            reason_codes.append("SITUATIONAL_AWARENESS")
            return DecisionResult(
                suggested_priority="medium" if conf >= self.high_threshold else "low",
                human_review_required=True,
                reason_codes=reason_codes,
                explanation="Provides general disaster context or situational awareness.",
            )

        # Rule 7: Non-humanitarian / off-topic content
        if label == "not_humanitarian":
            reason_codes.append("NON_HUMANITARIAN_CONTENT")
            if conf >= self.high_threshold:
                return DecisionResult(
                    suggested_priority="low",
                    human_review_required=False,
                    reason_codes=reason_codes,
                    explanation="Classified with high confidence as non-disaster or irrelevant chitchat.",
                )
            else:
                return DecisionResult(
                    suggested_priority="requires_review",
                    human_review_required=True,
                    reason_codes=["BORDERLINE_NON_HUMANITARIAN"],
                    explanation="Tentatively non-humanitarian but confidence is moderate; routed to review queue.",
                )

        # Default fallback
        return DecisionResult(
            suggested_priority="requires_review",
            human_review_required=True,
            reason_codes=["UNRECOGNIZED_CATEGORY"],
            explanation=f"Unrecognized prediction category '{label}'. Operator review mandated.",
        )


def process_report(
    classifier,
    engine: DecisionEngine,
    report: DisasterReportInput,
) -> AuraStreamDecisionPayload:
    """
    End-to-end processing pipeline from incoming report to structured decision payload.
    """
    classification = classifier.predict(report)
    decision = engine.evaluate(classification, report)
    return AuraStreamDecisionPayload(
        report_id=report.report_id,
        classification=classification,
        decision=decision,
    )
