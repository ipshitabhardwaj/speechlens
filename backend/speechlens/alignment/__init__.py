from __future__ import annotations

import threading

from speechlens.alignment.energy_fallback import EnergyFallbackAligner
from speechlens.alignment.protocol import Aligner
from speechlens.alignment.types import AlignmentResult, WordTiming

_aligner_instance: Aligner | None = None
_aligner_lock = threading.Lock()


def _ctc_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def get_aligner() -> Aligner:
    global _aligner_instance
    if _aligner_instance is not None:
        return _aligner_instance

    with _aligner_lock:
        if _aligner_instance is not None:
            return _aligner_instance

        if _ctc_available():
            from speechlens.alignment.ctc import CTCAligner
            _aligner_instance = CTCAligner()
        else:
            _aligner_instance = EnergyFallbackAligner()

        return _aligner_instance


def _reset_aligner() -> None:
    """Reset the singleton (for tests only)."""
    global _aligner_instance
    _aligner_instance = None


__all__ = [
    "Aligner",
    "AlignmentResult",
    "EnergyFallbackAligner",
    "WordTiming",
    "get_aligner",
    "_reset_aligner",
]
