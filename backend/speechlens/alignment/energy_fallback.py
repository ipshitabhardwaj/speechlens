from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from speechlens.alignment.types import AlignmentResult
from speechlens.features.extract import VADResult


class EnergyFallbackAligner:
    def align(
        self,
        samples: NDArray[np.float32],
        sample_rate: int,
        transcript: str,
        vad: VADResult,
    ) -> AlignmentResult:
        return AlignmentResult(
            method="energy_fallback",
            words=None,
            pauses=list(vad.pauses),
            unavailable=["word_timestamps: no forced alignment in this mode"],
        )
