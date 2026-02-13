# BankCo Voice Assistant - banking voice agent powered by a fine-tuned SLM

<p align="center">
  <img src="voice-assistant-banner.png" alt="BankCo Voice Assistant" width="600">
</p>

*A real-time voice banking assistant that replaces the cloud LLM with a fine-tuned 0.6B model, cutting pipeline latency from ~850ms to ~315ms.*

Enterprise voice assistants follow a cascaded pipeline: ASR (speech-to-text) -> LLM (intent & reasoning) -> TTS (text-to-speech). The LLM stage is consistently the bottleneck, consuming over 70% of processing time. But for voice assistants handling defined workflows with bounded intent taxonomies, the LLM performs a narrow set of tasks — intent classification, slot extraction, and dialogue state tracking. These are classification and extraction tasks, not open-ended generation.

We fine-tuned a Qwen3-0.6B model to handle multi-turn tool calling for a banking customer service workflow. The model classifies intents and extracts slots as structured function calls, running at **~40ms inference** vs 375-750ms for cloud LLMs. Combined with local ASR and TTS, the full pipeline runs under 400ms — well within the 500-800ms threshold for natural conversation.

| Component | Cloud LLM Pipeline | SLM Pipeline |
|---|---|---|
| ASR (Speech-to-Text) | 200-350ms | ~200ms |
| **LLM / SLM** | **375-750ms** | **~40ms** |
| TTS (Text-to-Speech) | 75-150ms | ~75ms |
| **Total** | **680-1300ms** | **~315ms** |

| Model | Parameters | Tool Call Accuracy (test set) |
|---|---|---|
| GPT-oss-120B (teacher) | 120B | 87.5% |
| **Qwen3-0.6B (tuned)** | **0.6B** | **90.4%** |
| Qwen3-0.6B (base) | 0.6B | 38.5% |

The tuned 0.6B model **exceeds the 120B teacher** on tool call accuracy while being **200x smaller** and running locally. The base Qwen3-0.6B only achieves 38.5% accuracy — fine-tuning is essential.


## Usage Examples

The assistant handles banking requests through voice (push-to-talk) or text. It identifies the intent, asks for any missing information, and executes the action.

### Multi-turn slot filling

```
You: I want to transfer some money
Bot: Could you provide the amount, which account to transfer from, and which account to transfer to?
You: 200 dollars from checking to savings
Bot: Done. Transferred $200.00 from checking to savings.
```

### Complete information in one turn

```
You: Cancel my credit card ending in 4532
Bot: Done. Your credit card ending in 4532 has been cancelled.
```

### Intent unclear then clarified

```
You: What about that thing I did the other day?
Bot: I didn't quite understand that. Could you tell me what you need? I can help you check
     your balance, transfer money, cancel or replace cards, pay bills, report fraud,
     or connect you to an agent.
You: ok lets cancel my card
Bot: Could you provide credit or debit, and the last 4 digits?
You: debit
Bot: Could you provide the last 4 digits?
You: 1234
Bot: Done. Your debit card ending in 1234 has been cancelled.
```

### Mid-conversation intent change

```
You: I need to cancel my card
Bot: Could you provide credit or debit, and the last 4 digits?
You: Actually, what's my checking balance first?
Bot: Your checking balance is $2,847.32.
```


## Getting Started

### Prerequisites

- Python 3.12+
- [Homebrew](https://brew.sh/) (macOS)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) with `llama-server` on your PATH

### 1. Download the models

Download the three models and place them in the `models/` directory:

| Model | Purpose | Download |
|---|---|---|
| Qwen3-ASR-0.6B | Speech-to-text | [Hugging Face](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |
| Qwen3-TTS-12Hz-0.6B-Base | Text-to-speech | [Hugging Face](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) |
| Qwen3-voice-assistant-slm-0.6B (GGUF) | Intent & slot extraction | [Hugging Face](https://huggingface.co/distil-labs/distil-qwen3-0.6b-voice-assistant-banking) |

```bash
# ASR
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir models/Qwen3-ASR-0.6B

# TTS
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/Qwen3-TTS-12Hz-0.6B-Base

# SLM (fine-tuned intent router)
huggingface-cli download distil-labs/distil-qwen3-0.6b-voice-assistant-banking --local-dir models/Qwen3-voice-assistant-slm-0.6B-gguf
```

The `assets/` directory ships with the repo and includes the default TTS reference voice clip.

### 2. Install dependencies

```bash
./install.sh
```

This installs all Python packages and the SoX system binary. The `qwen-asr` and `qwen-tts` packages have conflicting `transformers` pins (`==4.57.6` vs `==4.57.3`), so they are installed with `--no-deps`.

### 3. Start the SLM server

```bash
llama-server \
    --model models/Qwen3-voice-assistant-slm-0.6B-gguf/Qwen3-voice-assistant-slm-0.6B.gguf \
    --port 8000
```

### 4. Run the assistant

**Text-only mode** (no mic/speaker needed):

```bash
python orchestrator.py --port 8000
```

**Full voice mode** (push-to-talk):

```bash
python voice_assistant.py --slm-port 8000
```

Use `--device mps` on Apple Silicon (recommended) or `--device cuda:0` on NVIDIA GPUs. TTS is too slow on CPU.


## How the SLM Orchestrator Works

### The Problem

In multi-turn tool calling, the model must correctly classify intent and extract slots across a conversation — not just once, but every turn. A model with 80% single-turn accuracy drops to just 33% over a 5-turn conversation. Every percentage point matters.

| Single turn accuracy | 5-turn accuracy |
|---|---|
| 80% | 33% |
| 90% | 59% |
| 95% | 77% |
| 99% | 95% |

The base Qwen3-0.6B achieves only 38.5% on our banking test set — unusable for production. We needed fine-tuning.

### Architecture

The SLM acts as a function caller. Given a user utterance and conversation history, it outputs a structured tool call:

```
User: "I need to cancel my credit card ending in 1234"

SLM Output:
{"name": "cancel_card", "arguments": {"card_type": "credit", "card_last_four": "1234"}}
```

The deterministic orchestrator then checks for missing required slots, generates clarification questions or success responses using templates, and maintains conversation state. The SLM never generates user-facing text — all responses come from templates, ensuring predictable latency and brand consistency.

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Qwen3-ASR   │    │  Qwen3-0.6B   │    │   Qwen3-TTS   │
│               │───>│  (fine-tuned)  │───>│               │
│  ~200ms       │    │  ~40ms         │    │  ~75ms        │
└───────────────┘    └───────────────┘    └───────────────┘
       ▲                    │                    │
       │                    ▼                    ▼
┌──────┴──────┐    ┌───────────────┐    ┌──────┴──────┐
│  Microphone │    │ ORCHESTRATOR  │    │   Speaker   │
└─────────────┘    │ - Slot check  │    └─────────────┘
                   │ - Templates   │
                   │ - State mgmt  │
                   └───────────────┘
```

### Training Pipeline

**1. Seed Data:** We wrote 77 training conversations covering 14 banking functions, including ASR transcription artifacts (filler words, homophones, word splits) to handle real voice input. Examples:

```
Clean:    "Transfer 500 from my savings to checking"
With ASR: "Trans fur 500 from my savin to checkin"
With ASR: "Um I need to like transfer um five hundred dollars"
```

**2. Synthetic Expansion:** Using the [Distil Labs data synthesis pipeline](https://www.distillabs.ai/blog/small-expert-agents-from-10-examples/?utm_source=github&utm_medium=referral&utm_campaign=voice-assistant), we expanded the seed data into thousands of training examples with diverse phrasings and error patterns.

**3. Fine-tuning:** We trained Qwen3-0.6B using the Distil CLI on a multi-turn tool calling task, with a 120B teacher model for distillation.

```bash
curl -fsSL https://cli-assets.distillabs.ai/install.sh | sh
distil login

distil model create banking-voice-assistant
distil model upload-data <model-id> --data ./slm-finetuning/data
distil model run-training <model-id>
distil model download <model-id>
```

See [`slm-finetuning/`](slm-finetuning/) for the full training data and configuration.

### Supported Functions

The model handles 14 banking operations:

| Function | Required Slots | Description |
|---|---|---|
| `check_balance` | `account_type` | Check account balance |
| `transfer_money` | `amount`, `from_account`, `to_account` | Transfer between accounts |
| `pay_bill` | `payee`, `amount` | Pay a bill |
| `cancel_card` | `card_type`, `card_last_four` | Cancel a card |
| `replace_card` | `card_type`, `card_last_four` | Request replacement card |
| `activate_card` | `card_last_four` | Activate a new card |
| `reset_pin` | `card_type`, `card_last_four` | Reset card PIN |
| `report_fraud` | `card_type` | Report fraudulent transaction |
| `get_statement` | `account_type` | Request account statement |
| `speak_to_human` | — | Connect to a human agent |
| `greeting` | — | Conversation start |
| `goodbye` | — | Conversation end |
| `thank_you` | — | Express gratitude |
| `intent_unclear` | — | Cannot determine intent |


## ASR Module

The ASR module (`asr.py`) converts speech to text using [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B). It defines a `Protocol`-based interface, so **you can swap in any ASR backend** — a different local model (Whisper, Moonshine), a cloud API (Google Speech-to-Text, Deepgram, AssemblyAI), or a custom implementation. Any class with a matching `transcribe(audio, sample_rate) -> str` signature works as a drop-in replacement.

```python
class ASRModule(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...

# Use any backend:
asr = Qwen3ASR(model_path="models/Qwen3-ASR-0.6B", device="mps")       # local
asr = WhisperASR(model="large-v3")                                       # local alternative
asr = DeepgramASR(api_key="...")                                          # cloud
```


## TTS Module

The TTS module (`tts.py`) converts text to speech using [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) with voice cloning. It uses the same `Protocol`-based interface as ASR, so **you can swap in any TTS backend** — a different local model (Piper, Bark), a cloud API (ElevenLabs, Google TTS, Amazon Polly), or a custom implementation.

```python
class TTSModule(Protocol):
    def synthesize(self, text: str) -> tuple[np.ndarray, int]: ...

# Use any backend:
tts = Qwen3TTS(model_path="models/Qwen3-TTS-12Hz-0.6B-Base", device="mps")  # local
tts = ElevenLabsTTS(api_key="...", voice_id="...")                             # cloud
```

The default reference voice for cloning is provided at `assets/default_voice.wav`. To use a custom voice:

```bash
python voice_assistant.py \
    --ref-audio path/to/your_voice.wav \
    --ref-text "Exact transcript of the reference audio." \
    --slm-port 8000 --device mps
```


## Train Your Own Model

The workflow we used is generic across multi-turn tool calling tasks. You can train a model for your own voice assistant:

### 1. Define your functions

Specify the intent functions and their argument schemas. See `slm-finetuning/data/job_description.json` for the format.

### 2. Create seed examples

Write 50-100 example conversations covering your intents, including multi-turn slot filling. Add ASR artifacts if your pipeline uses speech input. See `slm-finetuning/data/train.jsonl`.

### 3. Train with Distil CLI

```bash
curl -fsSL https://cli-assets.distillabs.ai/install.sh | sh
distil login

distil model create my-voice-assistant
distil model upload-data <model-id> --data ./data
distil model run-training <model-id>
distil model download <model-id>
```

Check out the [Distil CLI Claude Code skill](https://github.com/distil-labs/distil-cli-skill) that can help you call the right training commands directly from Claude Code.


## FAQ

**Q: Why not just use GPT-4 / Claude for the intent routing?**

You can — but latency matters for voice. Cloud LLMs add 375-750ms per turn just for the "brain" stage. A local SLM runs in ~40ms, bringing total pipeline latency under 400ms for natural real-time conversation. It also runs offline with no API costs.

**Q: Why not use a small model without fine-tuning?**

The base Qwen3-0.6B achieves only 38.5% tool call accuracy on our test set — worse than a coin flip for most intents. Fine-tuning raises this to 90.4%, which is essential for reliable multi-turn conversations.

**Q: Can I use cloud ASR/TTS instead of local models?**

Yes! The ASR and TTS modules are Protocol-based — swap in any backend with a matching `transcribe()` or `synthesize()` signature. The key innovation is the SLM replacing the LLM; the ASR and TTS choices are independent.

**Q: What hardware do I need?**

The full pipeline (ASR + SLM + TTS) runs on a MacBook with Apple Silicon using MPS. The SLM server (llama.cpp) runs on CPU. On Linux, use CUDA for ASR/TTS. TTS inference is too slow on CPU.

**Q: The model makes an incorrect tool call. What can I do?**

Track the failure cases, add them to `slm-finetuning/data/train.jsonl`, and retrain. The model achieves 90.4% accuracy — for the remaining cases, the orchestrator's slot elicitation will still guide the user to a correct outcome.

**Q: Can you train a model for my company's specific voice workflows?**

Yes! Visit [distillabs.ai](https://www.distillabs.ai/?utm_source=github&utm_medium=referral&utm_campaign=voice-assistant) to discuss custom solutions trained on your specific intent taxonomies and dialogue patterns.


## Links

<p align="center">
  <a href="https://www.distillabs.ai/?utm_source=github&utm_medium=referral&utm_campaign=voice-assistant">
    <img src="https://github.com/distil-labs/badges/blob/main/badge-distillabs-home.svg?raw=true" alt="Distil Labs Homepage" />
  </a>
  <a href="https://github.com/distil-labs">
    <img src="https://github.com/distil-labs/badges/blob/main/badge-github.svg?raw=true" alt="GitHub" />
  </a>
  <a href="https://huggingface.co/distil-labs">
    <img src="https://github.com/distil-labs/badges/blob/main/badge-huggingface.svg?raw=true" alt="Hugging Face" />
  </a>
  <a href="https://www.linkedin.com/company/distil-labs/">
    <img src="https://github.com/distil-labs/badges/blob/main/badge-linkedin.svg?raw=true" alt="LinkedIn" />
  </a>
  <a href="https://distil-labs-community.slack.com/join/shared_invite/zt-36zqj87le-i3quWUn2bjErRq22xoE58g">
    <img src="https://github.com/distil-labs/badges/blob/main/badge-slack.svg?raw=true" alt="Slack" />
  </a>
  <a href="https://x.com/distil_labs">
    <img src="https://github.com/distil-labs/badges/blob/main/badge-twitter.svg?raw=true" alt="Twitter" />
  </a>
</p>
