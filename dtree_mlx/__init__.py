"""MLX runtime for DFlash speculative decoding on Apple Silicon."""

from importlib import import_module


_EXPORTS = {
    "DFlashGenerator": ("api", "DFlashGenerator"),
    "DFlashResult": ("api", "DFlashResult"),
    "Generator": ("api", "Generator"),
    "DFlashDraftModel": ("draft", "DFlashDraftModel"),
    "LoadedTargetModel": ("adapters", "LoadedTargetModel"),
    "adapter_for_model_type": ("adapters", "adapter_for_model_type"),
    "dflash_generate": ("runtime", "dflash_generate"),
    "dtree_generate": ("dtree_runtime", "dtree_generate"),
    "load_draft_model": ("draft", "load_draft_model"),
    "load_target_model": ("adapters", "load_target_model"),
    "longest_prefix_match": ("runtime", "longest_prefix_match"),
    "sample_tokens": ("runtime", "sample_tokens"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
