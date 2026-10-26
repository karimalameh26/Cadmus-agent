"""cadmus: observable, ground-truth-gated self-improving document/image classification."""

from cadmus.config import AppConfig, load_config
from cadmus.contracts import EvidenceRecord, Prediction, PromotionDecision, PromptBundle
from cadmus.runtime import Framework

__version__ = "0.1.0"
__all__ = ["AppConfig", "EvidenceRecord", "Framework", "Prediction", "PromotionDecision", "PromptBundle",
           "load_config"]
