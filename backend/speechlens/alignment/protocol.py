from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from speechlens.alignment.types import AlignmentResult
from speechlens.features.extract import VADResult


@runtime_checkable
class Aligner(Protocol):
    def align(
        self,
        samples: NDArray[np.float32],
        sample_rate: int,
        transcript: str,
        vad: VADResult,
    ) -> AlignmentResult: ...
