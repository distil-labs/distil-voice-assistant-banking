# Fine-tuning a Banking Voice Assistant with Distil CLI

Train a compact model that handles intent classification and slot extraction for banking customer service using the Distil Labs platform.

This guide covers two approaches:
1. **Training from structured data** — manually prepared train/test splits in the `data/` folder
2. **Training from traces** — production-style LLM interaction logs in the `traces/` folder, automatically processed by the platform

## Prerequisites

Install the Distil CLI:

```bash
curl -fsSL https://cli-assets.distillabs.ai/install.sh | sh
```

Authenticate:

```bash
distil login
```

## Supported Functions

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

---

## Option A: Training from Structured Data

### Training Data

The `data/` folder contains everything needed to train the model:

| File | Description |
|------|-------------|
| `job_description.json` | Task definition with 14 banking function schemas |
| `train.jsonl` | Training examples (77 multi-turn conversations) |
| `test.jsonl` | Evaluation examples (76 conversations) |
| `config.yaml` | Training configuration (Qwen3-0.6B base model) |

#### Example Training Sample

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

### Steps

#### 1. Create a Model

```bash
distil model create banking-voice-assistant
```

Save the returned `<model-id>` for subsequent commands.

#### 2. Upload Training Data

```bash
distil model upload-data <model-id> --data ./data
```

#### 3. Run Teacher Evaluation

Validate that a large model can solve the task before training:

```bash
distil model run-teacher-evaluation <model-id>
```

Check status:

```bash
distil model teacher-evaluation <model-id>
```

#### 4. Train the Model

Start distillation to create your compact voice assistant model:

```bash
distil model run-training <model-id>
```

Monitor progress:

```bash
distil model training <model-id>
```

#### 5. Download the Model

Once training completes, download the Ollama-ready package:

```bash
distil model download <model-id>
```

---

## Option B: Training from Traces

Instead of manually preparing structured train/test splits, you can train directly from production traces — logs of real LLM interactions in OpenAI messages format. The platform automatically filters, relabels, and splits them into training and test data.

### Trace Data

The `traces/` folder contains:

| File | Description |
|------|-------------|
| `traces.jsonl` | Multi-turn conversations in OpenAI messages format (~5k examples) |
| `job_description.json` | Task definition with 14 banking function schemas |
| `config.yaml` | Training configuration with trace processing parameters |

#### Trace Format

Each line in `traces.jsonl` is a complete multi-turn conversation with tool calls:

```json
{
  "messages": [
    {"role": "user", "content": "Hello, good morning"},
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "greeting", "arguments": {}}}]},
    {"role": "user", "content": "What's my checking balance?"},
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "check_balance", "arguments": {"account_type": "checking"}}}]}
  ],
  "tools": [...]
}
```

#### Trace Processing Configuration

The `config.yaml` includes a `trace_processing` section that controls how traces are processed:

```yaml
base:
  task: multi-turn-tool-calling-closed-book
  student_model_name: Qwen3-0.6B
  teacher_model_name: zai.glm-5
trace_processing:
  convert_to_single_turn: false    # Keep multi-turn conversations intact
  num_traces_as_training_base: 40  # Number of traces used as seed training data
  num_traces_as_testing_base: 40   # Number of traces used as seed test data
  teacher_model_name: zai.glm-5
  relabelling_committee_models:
    - zai.glm-5
    - openai.gpt-oss-120b
```

Key parameters:
- **`convert_to_single_turn`**: Set to `false` to preserve multi-turn conversations. When `true`, conversations are split into independent single-turn examples.
- **`num_traces_as_training_base`** / **`num_traces_as_testing_base`**: Number of traces used as seed data. Remaining traces are used as unstructured data for augmentation.
- **`relabelling_committee_models`**: Committee of models that relabel trace examples. The teacher picks the best candidate.

### Steps

#### 1. Create a Model

```bash
distil model create banking-voice-assistant-tft
```

Save the returned `<model-id>` for subsequent commands.

#### 2. Upload Traces

Upload the traces and kick off automatic processing (filtering, relabelling, train/test splitting):

```bash
distil model upload-traces <model-id> --data ./traces
```

Monitor processing status:

```bash
distil model upload-status <model-id>
```

To reprocess previously uploaded traces with different parameters (e.g., change committee models or trace counts):

```bash
distil model reprocess-traces <model-id> --config ./traces/config.yaml
```

#### 3. Run Teacher Evaluation

```bash
distil model run-teacher-evaluation <model-id>
```

Check status:

```bash
distil model teacher-evaluation <model-id>
```

#### 4. Train the Model

```bash
distil model run-training <model-id>
```

Monitor progress:

```bash
distil model training <model-id>
```

#### 5. Download the Model

```bash
distil model download <model-id>
```

---

## Local Deployment

Run your trained model locally with [Ollama](https://ollama.com):

```bash
ollama create banking-assistant -f Modelfile
ollama run banking-assistant
```

## Learn More

- [Distil Documentation](https://docs.distillabs.ai)
- [Data Preparation Guide](https://docs.distillabs.ai/how-to/data-preparation)
