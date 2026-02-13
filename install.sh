#!/usr/bin/env bash
set -e

# System dependency (macOS)
brew install sox

# Core dependencies
pip install transformers==4.57.6 openai torch torchaudio sounddevice soundfile numpy

# Transitive deps missing due to --no-deps on qwen packages
pip install accelerate librosa einops onnxruntime sox soynlp pytz qwen-omni-utils nagisa

# qwen-asr pins transformers==4.57.6, qwen-tts pins transformers==4.57.3
# Install with --no-deps to avoid the unresolvable conflict
pip install --no-deps qwen-asr qwen-tts
