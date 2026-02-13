"""ASR module — Protocol + Qwen3 implementation.

Pure model inference, no microphone I/O.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ASRModule(Protocol):
    """Structural type for any ASR backend."""

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Convert a mono float32 audio array to text."""
        ...


class Qwen3ASR:
    """Qwen3-ASR-0.6B wrapper.

    Uses ``qwen_asr.Qwen3ASRModel`` with local weights.
    The model handles resampling internally, so *audio* can be at any rate.
    """

    def __init__(self, model_path: str = "models/Qwen3-ASR-0.6B", device: str = "auto"):
        import torch
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        results = self.model.transcribe(audio=(audio, sample_rate))
        return results[0].text
