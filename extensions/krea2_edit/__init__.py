"""Source-graph-compatible Krea 2 edit training extension."""
from .krea2_edit import Krea2EditModel

AI_TOOLKIT_MODELS = [Krea2EditModel]
__all__ = ["Krea2EditModel", "AI_TOOLKIT_MODELS"]
