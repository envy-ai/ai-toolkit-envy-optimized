"""Model-neutral import; legacy Qwen class/import remains compatible."""
from .QwenGuidanceDistillationTrainer import QwenGuidanceDistillationTrainer, guidance_distillation_target

GuidanceDistillationTrainer = QwenGuidanceDistillationTrainer
