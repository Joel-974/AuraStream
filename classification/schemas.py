"""
Pydantic schemas for AuraStream Classification and Decision Engine.
Defines strict data contracts for incoming disaster reports and outgoing decision payloads.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class LocationModel(BaseModel):
    """Optional geographic location metadata."""
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    address: Optional[str] = None


class DisasterReportInput(BaseModel):
    """
    Standard incoming disaster report representation.
    Compatible with FastAPI ingestion endpoints.
    """
    report_id: str = Field(..., description="Unique report identifier")
    text: Optional[str] = Field(None, description="Disaster report textual description")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Report timestamp")
    location: Optional[LocationModel] = None
    modalities_available: List[str] = Field(default_factory=lambda: ["text"], description="Available media modalities")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional ingestion metadata")

    @field_validator("report_id")
    @classmethod
    def validate_report_id(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("report_id cannot be empty")
        return cleaned


class ClassificationResult(BaseModel):
    """
    Structured model prediction output.
    Separated from downstream decision logic.
    """
    status: str = Field(
        ...,
        description="Prediction status: 'predicted', 'low_confidence', 'no_text', 'model_unavailable', or 'error'"
    )
    predicted_label: Optional[str] = Field(None, description="Highest probability class label")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score of top class")
    class_probabilities: Dict[str, float] = Field(default_factory=dict, description="Probability distribution over classes")
    model_version: str = Field("unknown", description="Model identifier and version tag")


class DecisionResult(BaseModel):
    """
    Actionable recommendation produced by the Decision Engine.
    Employs cautious triage rules and flags uncertain cases for human verification.
    """
    suggested_priority: str = Field(
        ...,
        description="Triage priority: 'immediate', 'high', 'medium', 'low', 'requires_review', or 'invalid'"
    )
    human_review_required: bool = Field(
        True,
        description="Whether a human operator must confirm the recommendation before action"
    )
    reason_codes: List[str] = Field(
        default_factory=list,
        description="Auditable machine-readable codes explaining the decision rationale"
    )
    explanation: str = Field(
        "",
        description="Human-readable explanation of why this decision was reached"
    )


class AuraStreamDecisionPayload(BaseModel):
    """
    Complete end-to-end output payload delivered back to storage, API, or command dashboard.
    """
    report_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    classification: ClassificationResult
    decision: DecisionResult
