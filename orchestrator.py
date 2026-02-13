"""BankCo Voice Assistant — Text-based orchestrator (Stage 1).

Pairs a lightweight SLM client (OpenAI-compatible API) with a deterministic
orchestrator that handles slot elicitation, dialogue control, and simulated
backend execution.

Usage:
    python orchestrator.py --model model --port 11434 [--debug]
"""

import argparse
import json
import random

from openai import OpenAI

# ---------------------------------------------------------------------------
# Tools definition (14 banking functions)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_balance",
            "description": "Check the balance of a bank account",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_type": {
                        "type": "string",
                        "enum": ["checking", "savings", "credit"],
                        "description": "Type of account to check balance for",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_statement",
            "description": "Request an account statement to be sent to the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_type": {
                        "type": "string",
                        "enum": ["checking", "savings", "credit"],
                        "description": "Type of account to get statement for",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["last_month", "last_3_months", "last_year"],
                        "description": "Time period for the statement",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_money",
            "description": "Transfer money between the user's own bank accounts",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "Amount to transfer in dollars",
                    },
                    "from_account": {
                        "type": "string",
                        "enum": ["checking", "savings"],
                        "description": "Account to transfer money from",
                    },
                    "to_account": {
                        "type": "string",
                        "enum": ["checking", "savings"],
                        "description": "Account to transfer money to",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_card",
            "description": "Cancel and deactivate a bank card",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_type": {
                        "type": "string",
                        "enum": ["credit", "debit"],
                        "description": "Type of card to cancel",
                    },
                    "card_last_four": {
                        "type": "string",
                        "description": "Last 4 digits of the card number",
                    },
                    "reason": {
                        "type": "string",
                        "enum": ["lost", "stolen", "damaged", "other"],
                        "description": "Reason for cancelling the card",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_card",
            "description": "Request a replacement card to be sent to the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_type": {
                        "type": "string",
                        "enum": ["credit", "debit"],
                        "description": "Type of card to replace",
                    },
                    "card_last_four": {
                        "type": "string",
                        "description": "Last 4 digits of the card number",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activate_card",
            "description": "Activate a new card that was received in the mail",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_last_four": {
                        "type": "string",
                        "description": "Last 4 digits of the card number to activate",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_fraud",
            "description": "Report a fraudulent or unauthorized transaction on a card",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_type": {
                        "type": "string",
                        "enum": ["credit", "debit"],
                        "description": "Type of card with fraudulent activity",
                    },
                    "card_last_four": {
                        "type": "string",
                        "description": "Last 4 digits of the card number",
                    },
                    "transaction_amount": {
                        "type": "number",
                        "description": "Amount of the fraudulent transaction in dollars",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_pin",
            "description": "Reset the PIN for a bank card",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_type": {
                        "type": "string",
                        "enum": ["credit", "debit"],
                        "description": "Type of card to reset PIN for",
                    },
                    "card_last_four": {
                        "type": "string",
                        "description": "Last 4 digits of the card number",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pay_bill",
            "description": "Pay a bill to a company or service provider",
            "parameters": {
                "type": "object",
                "properties": {
                    "payee": {
                        "type": "string",
                        "description": "Name of the company or person to pay",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount to pay in dollars",
                    },
                    "from_account": {
                        "type": "string",
                        "enum": ["checking", "savings"],
                        "description": "Account to pay from",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speak_to_human",
            "description": "Connect the user to a human customer service agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "enum": ["general", "fraud", "loans", "technical"],
                        "description": "Department to connect to",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "intent_unclear",
            "description": "Use when the user's intent cannot be determined from their message",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "greeting",
            "description": "User is greeting or starting the conversation",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "goodbye",
            "description": "User is ending the conversation",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "thank_you",
            "description": "User is expressing gratitude or thanks",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Orchestrator constants (from design.md)
# ---------------------------------------------------------------------------
FUNCTION_REQUIRED_ARGS: dict[str, list[str]] = {
    "cancel_card": ["card_type", "card_last_four"],
    "replace_card": ["card_type", "card_last_four"],
    "activate_card": ["card_last_four"],
    "reset_pin": ["card_type", "card_last_four"],
    "transfer_money": ["amount", "from_account", "to_account"],
    "check_balance": ["account_type"],
    "pay_bill": ["payee", "amount"],
    "get_statement": ["account_type"],
    "report_fraud": ["card_type"],
    "speak_to_human": [],
    "greeting": [],
    "goodbye": [],
    "thank_you": [],
    "intent_unclear": [],
}

INDIVIDUAL_SLOT_PROMPTS: dict[str, dict[str, str]] = {
    "cancel_card": {
        "card_type": "credit or debit",
        "card_last_four": "the last 4 digits",
        "reason": "the reason for cancellation",
    },
    "replace_card": {
        "card_type": "credit or debit",
        "card_last_four": "the last 4 digits",
    },
    "activate_card": {
        "card_last_four": "the last 4 digits of the card",
    },
    "reset_pin": {
        "card_type": "credit or debit",
        "card_last_four": "the last 4 digits",
    },
    "transfer_money": {
        "amount": "the amount",
        "from_account": "which account to transfer from",
        "to_account": "which account to transfer to",
    },
    "check_balance": {
        "account_type": "the account type (checking, savings, or credit)",
    },
    "pay_bill": {
        "payee": "who to pay",
        "amount": "the amount",
    },
    "get_statement": {
        "account_type": "the account type (checking, savings, or credit)",
    },
    "report_fraud": {
        "card_type": "credit or debit",
    },
}

SUCCESS_TEMPLATES: dict[str, str] = {
    "cancel_card": "Done. Your {card_type} card ending in {card_last_four} has been cancelled.",
    "replace_card": "Done. A new {card_type} card will arrive in 5-7 business days.",
    "activate_card": "Your card ending in {card_last_four} is now active.",
    "reset_pin": "Your PIN has been reset. You'll receive a new PIN by mail in 3-5 days.",
    "transfer_money": "Done. Transferred ${amount:.2f} from {from_account} to {to_account}.",
    "check_balance": "Your {account_type} balance is ${balance:.2f}.",
    "pay_bill": "Done. Paid ${amount:.2f} to {payee}.",
    "get_statement": "I'm sending your {account_type} statement to your registered email.",
    "report_fraud": "I've flagged your {card_type} card for review. Our fraud team will contact you within 24 hours.",
    "speak_to_human": "Connecting you to an agent now. Please hold.",
    "greeting": "Hello! Welcome to BankCo. How can I help you today?",
    "goodbye": "Goodbye! Thanks for calling BankCo.",
    "thank_you": "You're welcome! Is there anything else I can help with?",
}


# ---------------------------------------------------------------------------
# SLM Client — stateless wrapper around an OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a tool-calling model working on:\n"
        "<task_description>You are a voice assistant for BankCo, a retail bank. "
        "The user input is automatically transcribed speech from an ASR system, so "
        "it may contain transcription errors, homophones, filler words, or unusual "
        "phrasings. Parse the user's request and return the appropriate function call "
        "despite any transcription artifacts. If you can identify the intent, call the "
        "matching function. Extract any mentioned argument values; omit arguments not "
        "mentioned. If you cannot understand what the user wants, call intent_unclear(). "
        "Use conversation history to understand context from previous turns.</task_description>\n\n"
        "Respond to the conversation history by generating an appropriate tool call that "
        "satisfies the user request. Generate only the tool call according to the provided "
        "tool schema, do not generate anything else. Always respond with a tool call.\n\n"
    ),
}


class SLMClient:
    """Lightweight client for a llama.cpp / Ollama / vLLM server."""

    def __init__(self, model_name: str, api_key: str = "EMPTY", port: int = 11434):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key=api_key,
        )

    def invoke(self, conversation_history: list[dict]) -> dict | str:
        """Send *full* conversation history to the SLM and return a parsed
        function-call dict ``{"name": ..., "arguments": ...}`` or an error
        string if no valid tool call could be extracted.
        """
        messages = [SYSTEM_PROMPT] + conversation_history

        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            tools=TOOLS,
            tool_choice="required",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        response = chat_response.choices[0].message

        # --- Path A: proper tool_calls in the response ---
        if response.tool_calls:
            fn = response.tool_calls[0].function
            arguments = fn.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            return {"name": fn.name, "arguments": arguments}

        # --- Path B: model returned JSON in content (fallback) ---
        if response.content:
            try:
                parsed = json.loads(response.content.strip())
                if "name" in parsed:
                    args = parsed.get("arguments", parsed.get("parameters", {}))
                    if isinstance(args, str):
                        args = json.loads(args)
                    return {"name": parsed["name"], "arguments": args}
            except (json.JSONDecodeError, KeyError):
                pass

        return f"No valid tool call in SLM response, model returned {response}"


# ---------------------------------------------------------------------------
# Text Orchestrator
# ---------------------------------------------------------------------------
class TextOrchestrator:
    """Deterministic dialogue manager sitting between the user and the SLM."""

    def __init__(self, slm_client: SLMClient, debug: bool = False):
        self.slm = slm_client
        self.debug = debug
        self.conversation_history: list[dict] = []

    def process_utterance(self, transcript: str) -> str | None:
        """Full turn: user text in -> bot response out."""
        # 0. Exit if the user wants to quit
        if transcript.lower() in ("quit", "exit"):
            return None

        # 1. Append user turn
        self.conversation_history.append({"role": "user", "content": transcript})

        # 2. Call SLM
        function_call = self.slm.invoke(self.conversation_history)

        if self.debug:
            print(f"  [DEBUG] SLM returned: {function_call}")

        # 3. If the SLM failed to return a valid call, treat as unclear
        if isinstance(function_call, str):
            self.conversation_history.append({"role": "assistant", "content": ""})
            return self.generate_clarification_response()

        # 4. Record assistant turn in history (tool_calls format)
        args_str = (
            json.dumps(function_call["arguments"])
            if isinstance(function_call["arguments"], dict)
            else function_call["arguments"]
        )
        tool_call_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                        "arguments": args_str,
                    },
                }
            ],
        }
        self.conversation_history.append(tool_call_msg)

        # 5. Route through orchestrator logic — None signals end of conversation
        return self.handle_function_call(function_call)

    def reset(self) -> None:
        self.conversation_history = []

    def handle_function_call(self, function_call: dict) -> str | None:
        name = function_call["name"]
        arguments = function_call.get("arguments", {})

        if name == "goodbye":
            return None

        if name == "intent_unclear":
            return self.generate_clarification_response()

        # Check for missing required args
        missing = self.get_missing_args(name, arguments)
        if missing:
            return self.generate_slot_elicitation(name, missing, arguments)

        # All slots filled — execute
        return self.execute_and_respond(name, arguments)

    def get_missing_args(self, function_name: str, arguments: dict) -> list[str]:
        required = FUNCTION_REQUIRED_ARGS.get(function_name, [])
        return [arg for arg in required if arguments.get(arg) is None]

    def generate_clarification_response(self) -> str:
        capabilities = [
            "check your balance",
            "transfer money",
            "cancel or replace cards",
            "pay bills",
            "report fraud",
            "or connect you to an agent",
        ]
        return (
            "I didn't quite understand that. Could you tell me what you need? "
            f"I can help you {', '.join(capabilities)}."
        )

    def generate_slot_elicitation(
        self, function: str, missing_args: list[str], current_args: dict
    ) -> str:
        individual = INDIVIDUAL_SLOT_PROMPTS.get(function, {})
        questions = [
            individual.get(arg, f"the {arg.replace('_', ' ')}") for arg in missing_args
        ]
        if len(questions) == 1:
            return f"Could you provide {questions[0]}?"
        return f"Could you provide {', '.join(questions[:-1])}, and {questions[-1]}?"

    def execute_and_respond(self, function: str, arguments: dict) -> str:
        api_result = self.call_backend_api(function, arguments)
        template = SUCCESS_TEMPLATES.get(function, "Done.")
        return template.format(**arguments, **api_result)

    def call_backend_api(self, function: str, arguments: dict) -> dict:
        """Simulate a backend — returns extra data needed by templates."""
        if function == "check_balance":
            return {"balance": round(random.uniform(100, 25_000), 2)}
        return {}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="BankCo text assistant (Stage 1)")
    parser.add_argument(
        "--model", type=str, default="model", help="Model name served by the backend"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port of the OpenAI-compatible server"
    )
    parser.add_argument(
        "--api-key", type=str, default="EMPTY", help="API key (default EMPTY)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print raw SLM output each turn"
    )
    args = parser.parse_args()

    slm = SLMClient(model_name=args.model, api_key=args.api_key, port=args.port)
    orchestrator = TextOrchestrator(slm, debug=args.debug)

    print("BankCo Assistant (type 'quit' or 'exit' to stop)\n")
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            response = orchestrator.process_utterance(user_input)
            if response is None:
                print("Bot: Goodbye! Thanks for calling BankCo.")
                break
            print(f"Bot: {response}")
    except (KeyboardInterrupt, EOFError):
        print("\nBot: Goodbye! Thanks for calling BankCo.")


if __name__ == "__main__":
    main()
