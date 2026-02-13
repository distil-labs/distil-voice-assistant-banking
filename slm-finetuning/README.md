# Fine-tuning a Banking Voice Assistant with Distil CLI

Train a compact model that handles intent classification and slot extraction for banking customer service using the Distil Labs platform.

## Prerequisites

Install the Distil CLI:

```bash
curl -fsSL https://cli-assets.distillabs.ai/install.sh | sh
```

Authenticate:

```bash
distil login
```

## Training Data

The `data/` folder contains everything needed to train the model:

| File | Description |
|------|-------------|
| `job_description.json` | Task definition with 14 banking function schemas |
| `train.jsonl` | Training examples (77 multi-turn conversations) |
| `test.jsonl` | Evaluation examples (76 conversations) |
| `config.yaml` | Training configuration (Qwen3-0.6B base model) |

### Example Training Sample

**Input (multi-turn conversation):**
```json
[
  {"role": "user", "content": "I need to move some money"},
  {"role": "assistant", "content": null, "tool_calls": [{"function": {"name": "transfer_money", "arguments": "{}"}}]},
  {"role": "user", "content": "500 from savings to checking"}
]
```

**Output (function call):**
```json
{"name": "transfer_money", "parameters": {"amount": 500, "from_account": "savings", "to_account": "checking"}}
```

### Supported Functions

The model handles 14 banking operations:
- `check_balance` - Check account balances
- `get_statement` - Request account statements
- `transfer_money` - Transfer between accounts
- `pay_bill` - Pay bills to payees
- `cancel_card` / `replace_card` / `activate_card` - Card management
- `report_fraud` - Report fraudulent transactions
- `reset_pin` - Reset card PIN
- `speak_to_human` - Connect to customer service
- `greeting` / `goodbye` / `thank_you` / `intent_unclear` - Conversation control

## Training Steps

### 1. Create a Model

```bash
distil model create banking-voice-assistant
```

Save the returned `<model-id>` for subsequent commands.

### 2. Upload Training Data

```bash
distil model upload-data <model-id> --data ./data
```

### 3. Run Teacher Evaluation

Validate that a large model can solve the task before training:

```bash
distil model run-teacher-evaluation <model-id>
```

Check status:

```bash
distil model teacher-evaluation <model-id>
```

### 4. Train the Model

Start distillation to create your compact voice assistant model:

```bash
distil model run-training <model-id>
```

Monitor progress:

```bash
distil model training <model-id>
```

### 5. Download the Model

Once training completes, download the Ollama-ready package:

```bash
distil model download <model-id>
```

## Local Deployment

Run your trained model locally with [Ollama](https://ollama.com):

```bash
ollama create banking-assistant -f Modelfile
ollama run banking-assistant
```

## Model Configuration

The training uses:
- **Base model:** Qwen3-0.6B
- **Teacher model:** openai.gpt-oss-120b
- **Task type:** Multi-turn tool calling (closed book)

The dataset includes ASR transcription artifacts (filler words, homophones, word splits) to handle real voice input.

## Learn More

- [Distil Documentation](https://docs.distillabs.ai)
- [Input Preparation Guide](https://docs.distillabs.ai/how-to/input-preparation)
