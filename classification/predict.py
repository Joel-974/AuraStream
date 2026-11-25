"""
Inference engine for AuraStream disaster report classification.
Provides standalone prediction functions and safe input handling.
Powered by pure-Python baseline model.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.schemas import ClassificationResult, DisasterReportInput
from classification.config import MODELS_DIR
from classification.baseline_model import PureNaiveBayesClassifier


class DisasterClassifier:
    """
    Inference wrapper for trained disaster classification models.
    Supports on-demand loading, graceful missing model fallbacks, and probability ranking.
    """

    def __init__(self, task: str = "humanitarian", model_path: Optional[Path] = None):
        self.task = task
        self.model_path = model_path or (MODELS_DIR / f"baseline_{task}.json")
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load trained model parameters from disk if available."""
        if self.model_path.exists():
            try:
                self.model = PureNaiveBayesClassifier.load(self.model_path)
            except Exception as e:
                print(f"Warning: Failed to load model from {self.model_path}: {e}")
                self.model = None
        else:
            self.model = None

    @property
    def is_available(self) -> bool:
        """Check whether the model is trained and loaded."""
        return self.model is not None

    @property
    def pipeline(self):
        """Backwards-compatible alias for the underlying model."""
        return self.model

    def predict(self, input_data: Union[str, DisasterReportInput, Dict]) -> ClassificationResult:
        """
        Run inference on a raw text string, dictionary, or DisasterReportInput schema.
        Handles missing text, empty strings, and missing models cleanly.
        """
        # Extract text from varying input shapes
        text = None
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, DisasterReportInput):
            text = input_data.text
        elif isinstance(input_data, dict):
            text = input_data.get("text")

        # Handle missing or empty text
        if text is None or not str(text).strip():
            return ClassificationResult(
                status="no_text",
                predicted_label=None,
                confidence=0.0,
                class_probabilities={},
                model_version=f"{self.task}-baseline-v1" if self.is_available else "unavailable",
            )

        cleaned_text = str(text).strip()

        # Handle model not loaded or unavailable
        if not self.is_available:
            return ClassificationResult(
                status="model_unavailable",
                predicted_label=None,
                confidence=0.0,
                class_probabilities={},
                model_version="unavailable",
            )

        try:
            top_class, top_conf, prob_dict = self.model.predict_single(cleaned_text)

            status = "predicted"
            if top_conf < 0.35:
                status = "low_confidence"

            return ClassificationResult(
                status=status,
                predicted_label=top_class,
                confidence=top_conf,
                class_probabilities=prob_dict,
                model_version=f"{self.task}-baseline-v1",
            )
        except Exception as e:
            return ClassificationResult(
                status="error",
                predicted_label=None,
                confidence=0.0,
                class_probabilities={},
                model_version=f"{self.task}-baseline-v1",
            )
