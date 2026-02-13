"""TTS module — Protocol + Qwen3 implementation.

Pure model inference, no speaker I/O.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TTSModule(Protocol):
    """Structural type for any TTS backend."""

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Convert text to audio. Returns (mono float32 array, sample_rate)."""
        ...


class Qwen3TTS:
    """Qwen3-TTS-12Hz-0.6B-Base wrapper (voice-clone mode).

    Uses ``qwen_tts.Qwen3TTSModel`` with local weights.  The Base model
    always requires a reference audio clip for voice cloning.
    """

    def __init__(
        self,
        model_path: str = "models/Qwen3-TTS-12Hz-0.6B-Base",
        ref_audio_path: str = "assets/default_voice.wav",
        ref_text: str = "Hello, welcome to BankCo.",
        device: str = "auto",
    ):
        import torch
        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16,
        )
        # Pre-compute the voice-clone prompt so we don't re-extract
        # speaker embeddings on every call.
        self.voice_clone_prompt = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=ref_text,
        )

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language="English",
            voice_clone_prompt=self.voice_clone_prompt,
        )
        return wavs[0], sr
