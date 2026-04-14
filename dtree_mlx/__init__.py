"""MLX runtime for DFlash speculative decoding on Apple Silicon."""

from .adapters import LoadedTargetModel, adapter_for_model_type, load_target_model
from .api import DFlashGenerator, DFlashResult, Generator
from .dtree_runtime import dtree_generate
from .draft import DFlashDraftModel, load_draft_model
from .runtime import dflash_generate, longest_prefix_match, sample_tokens

__all__ = [
    "DFlashGenerator",
    "DFlashResult",
    "Generator",
    "DFlashDraftModel",
    "LoadedTargetModel",
    "adapter_for_model_type",
    "dflash_generate",
    "dtree_generate",
    "load_draft_model",
    "load_target_model",
    "longest_prefix_match",
    "sample_tokens",
]
