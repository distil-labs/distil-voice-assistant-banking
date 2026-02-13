"""BankCo Voice Assistant — full speech pipeline (Phase 2).

Ties ASR, TextOrchestrator, and TTS together with push-to-talk mic/speaker I/O.

Usage:
    python voice_assistant.py \
        --slm-model model --slm-port 8000 \
        --asr-model models/Qwen3-ASR-0.6B \
        --tts-model models/Qwen3-TTS-12Hz-0.6B-Base \
        --ref-audio assets/default_voice.wav \
        --ref-text "Hello, welcome to BankCo." \
        --device cuda:0 --debug
"""

from __future__ import annotations

import argparse
import threading

import numpy as np
import sounddevice as sd

from asr import ASRModule, Qwen3ASR
from orchestrator import SLMClient, TextOrchestrator
from tts import Qwen3TTS, TTSModule

RECORD_SAMPLE_RATE = 16_000  # Hz, mono


class VoiceAssistant:
    """Push-to-talk voice loop: mic -> ASR -> orchestrator -> TTS -> speaker."""

    def __init__(self, asr: ASRModule, orchestrator: TextOrchestrator, tts: TTSModule):
        self.asr = asr
        self.orchestrator = orchestrator
        self.tts = tts

    def record_utterance(self) -> tuple[np.ndarray, int]:
        """Record from the default mic until Enter is pressed again.

        Uses a background ``sounddevice.InputStream`` so the main thread
        can simply block on ``input()``.
        """
        chunks: list[np.ndarray] = []
        stop_event = threading.Event()

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                print(f"  [audio] {status}")
            chunks.append(indata.copy())

        stream = sd.InputStream(
            samplerate=RECORD_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=_callback,
        )

        input("  Press Enter to START recording...")
        stream.start()
        input("  Recording... Press Enter to STOP.")
        stream.stop()
        stream.close()

        if not chunks:
            return np.zeros(0, dtype=np.float32), RECORD_SAMPLE_RATE

        audio = np.concatenate(chunks, axis=0).squeeze()  # (samples,)
        return audio, RECORD_SAMPLE_RATE

    @staticmethod
    def play_audio(audio: np.ndarray, sample_rate: int) -> None:
        """Play an audio array through the default speaker (blocking)."""
        sd.play(audio, samplerate=sample_rate)
        sd.wait()

    def run(self) -> None:
        """Main loop"""
        print("BankCo Voice Assistant (push-to-talk - say 'quit' or 'exit' to stop)\n")


        try:
            while True:
                # 1. Record
                audio, sr = self.record_utterance()
                if audio.size == 0:
                    print("  (no audio captured, try again)")
                    continue

                # 2. ASR
                transcript = self.asr.transcribe(audio, sr)
                print(f"  You: {transcript}")

                if not transcript.strip():
                    print("  (empty transcript, try again)")
                    continue

                # 3. Orchestrator
                response = self.orchestrator.process_utterance(transcript)
                if response is None:
                    print("Bot: Goodbye! Thanks for calling BankCo.")
                    break
                print(f"  Bot: {response}")

                # 4. TTS + playback
                tts_audio, tts_sr = self.tts.synthesize(response)
                self.play_audio(tts_audio, tts_sr)

        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye! Thanks for calling BankCo.")


def main() -> None:
    parser = argparse.ArgumentParser(description="BankCo voice assistant (Phase 2)")

    # SLM / orchestrator
    parser.add_argument("--slm-model", type=str, default="model", help="Model name served by the SLM backend")
    parser.add_argument("--slm-port", type=int, default=8000, help="Port of the OpenAI-compatible SLM server")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API key for SLM server")

    # ASR
    parser.add_argument("--asr-model", type=str, default="models/Qwen3-ASR-0.6B", help="Path to ASR model weights")

    # TTS
    parser.add_argument("--tts-model", type=str, default="models/Qwen3-TTS-12Hz-0.6B-Base", help="Path to TTS model weights")
    parser.add_argument("--ref-audio", type=str, default="assets/default_voice.wav", help="Reference audio for TTS voice cloning")
    parser.add_argument("--ref-text", type=str, default="The birch canoe slid on the smooth planks. Glue the sheet.", help="Transcript of the reference audio")

    # Shared
    parser.add_argument("--device", type=str, default="auto", help="Torch device for ASR/TTS models")
    parser.add_argument("--debug", action="store_true", help="Print raw SLM output each turn")
    args = parser.parse_args()

    # --- Build components ---
    print("Loading ASR model...")
    asr = Qwen3ASR(model_path=args.asr_model, device=args.device)

    print("Loading TTS model...")
    tts = Qwen3TTS(
        model_path=args.tts_model,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        device=args.device,
    )

    slm = SLMClient(model_name=args.slm_model, api_key=args.api_key, port=args.slm_port)
    orchestrator = TextOrchestrator(slm, debug=args.debug)

    # --- Run ---
    assistant = VoiceAssistant(asr=asr, orchestrator=orchestrator, tts=tts)
    assistant.run()


if __name__ == "__main__":
    main()
