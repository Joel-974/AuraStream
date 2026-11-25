"""
Ingestion Adapter for AuraStream.
FastAPI integration layer connecting the ingestion pipeline to the Classification & Decision Engine.
Can be mounted directly into the teammate's existing FastAPI application or run as a standalone service.
"""

from typing import List
from fastapi import APIRouter, HTTPException, status
from classification.schemas import (
    DisasterReportInput,
    AuraStreamDecisionPayload,
    ClassificationResult,
    DecisionResult,
)
from classification.predict import DisasterClassifier
from classification.decision_engine import DecisionEngine, process_report

router = APIRouter(prefix="/api/v1", tags=["Classification and Decision"])

# Global singleton classifier and decision engine instances
_classifier = DisasterClassifier(task="humanitarian")
_decision_engine = DecisionEngine()


def get_classifier() -> DisasterClassifier:
    """Dependency injector / accessor for classifier."""
    return _classifier


def get_decision_engine() -> DecisionEngine:
    """Dependency injector / accessor for decision engine."""
    return _decision_engine


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Check the health of the classification and decision engine service.
    Indicates whether a trained model is loaded or pending training.
    """
    classifier = get_classifier()
    return {
        "status": "healthy",
        "model_loaded": classifier.is_available,
        "task": classifier.task,
        "model_path": str(classifier.model_path),
        "note": "Ready for inference" if classifier.is_available else "Model artifact not found. Run training script.",
    }


@router.post(
    "/classify-report",
    response_model=AuraStreamDecisionPayload,
    status_code=status.HTTP_200_OK,
    summary="Classify a single disaster report and produce triage decision",
)
def classify_single_report(report: DisasterReportInput):
    """
    Process an incoming disaster report through the classification model
    and decision engine, returning prioritized recommendations.
    """
    try:
        classifier = get_classifier()
        engine = get_decision_engine()
        return process_report(classifier, engine, report)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification pipeline processing error: {str(exc)}",
        )


@router.post(
    "/batch-classify",
    response_model=List[AuraStreamDecisionPayload],
    status_code=status.HTTP_200_OK,
    summary="Batch classify multiple disaster reports",
)
def classify_batch_reports(reports: List[DisasterReportInput]):
    """
    Process multiple incoming disaster reports in a single request.
    """
    classifier = get_classifier()
    engine = get_decision_engine()
    results = []
    for report in reports:
        results.append(process_report(classifier, engine, report))
    return results
