"""ag-agent: observable, ground-truth-gated self-improving document/image classification."""

from ag_agent.config import AppConfig, load_config
from ag_agent.contracts import EvidenceRecord, Prediction, PromotionDecision, PromptBundle
from ag_agent.runtime import Framework

__version__ = "0.1.0"
__all__ = ["AppConfig", "EvidenceRecord", "Framework", "Prediction", "PromotionDecision", "PromptBundle",
           "load_config"]
