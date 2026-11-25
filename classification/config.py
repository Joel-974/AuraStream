"""
Configuration settings for AuraStream Classification and Decision Engine.
Centralizes paths, provisional thresholds, labels, and reproducibility seeds.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Reproducibility
RANDOM_SEED = 42

# CrisisMMD Official Dataset URLs (Text and annotations splits from QCRI CrisisNLP)
CRISISMMD_BASE_URL = "https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/main"
CRISISMMD_SPLITS = ["train", "dev", "test"]
CRISISMMD_TASKS = {
    "humanitarian": [
        "affected_individuals",
        "infrastructure_and_utility_damage",
        "injured_or_dead_people",
        "missing_or_found_people",
        "not_humanitarian",
        "other_relevant_information",
        "rescue_volunteering_or_donation_effort",
        "vehicle_damage",
    ],
    "informative": [
        "informative",
        "not_informative",
    ],
    "damage": [
        "little_or_no_damage",
        "mild_damage",
        "severe_damage",
    ],
}

# Provisional Decision Engine Thresholds
# Note: These are provisional heuristic settings designed for cautious emergency triage,
# not scientifically validated clinical cutoffs. They must be tuned with emergency response stakeholders.
CONFIDENCE_THRESHOLD_HIGH = 0.65
CONFIDENCE_THRESHOLD_LOW = 0.40

# Critical humanitarian categories that trigger high or immediate triage attention
CRITICAL_CATEGORIES = {
    "injured_or_dead_people",
    "infrastructure_and_utility_damage",
    "missing_or_found_people",
}

SECONDARY_RESPONSE_CATEGORIES = {
    "rescue_volunteering_or_donation_effort",
    "affected_individuals",
    "vehicle_damage",
}
