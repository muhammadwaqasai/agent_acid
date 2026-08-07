"""
examples/llm_agent_demo.py

This is the REAL DEAL: an actual AI agent (via OpenAI) decides what
steps to take to complete a task. Your engine executes each step,
checks guardrails, and rolls back automatically if anything goes wrong.

SETUP REQUIRED:
1. Install the OpenAI SDK:
       pip install openai
2. Set your API key as an environment variable before running:

   Windows (PowerShell):
       $env:OPENAI_API_KEY = "sk-..."

   Mac/Linux:
       export OPENAI_API_KEY="sk-..."

3. Run:
       py examples/llm_agent_demo.py     (Windows)
       python3 examples/llm_agent_demo.py  (Mac/Linux)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_acid.core import ReversibleTool
from agent_acid.guardrails import no_negative_numbers, max_value
from agent_acid.llm_agent import ToolSpec, LLMAgentRunner


# --- Fake "real world" systems (safe to test with, no real APIs touched) ---
fake_accounts = {}
fake_charges = []


def create_account_execute(kwargs):
    user_id = kwargs["name"].lower()
    fake_accounts[user_id] = {"name": kwargs["name"], "email": kwargs["email"]}
    return {"user_id": user_id}

def create_account_compensate(kwargs, result):
    fake_accounts.pop(result["user_id"], None)

create_account_tool = ReversibleTool(
    name="create_account",
    description="Creates a new customer account with a name and email",
    execute=create_account_execute,
    compensate=create_account_compensate,
)


def charge_card_execute(kwargs):
    charge = {"user_id": kwargs["user_id"], "amount": kwargs["amount"]}
    fake_charges.append(charge)
    return charge

def charge_card_compensate(kwargs, result):
    fake_charges.append({"user_id": result["user_id"], "amount": -result["amount"], "note": "REFUNDED"})

charge_card_tool = ReversibleTool(
    name="charge_card",
    description="Charges the customer's card a given dollar amount",
    execute=charge_card_execute,
    compensate=charge_card_compensate,
    guardrails=[
        no_negative_numbers("amount"),
        max_value("amount", limit=500),  # hard safety cap: never auto-charge over $500
    ],
)


# --- Give each tool a JSON schema describing its arguments (so the AI knows how to call it) ---
tool_specs = [
    ToolSpec(
        tool=create_account_tool,
        parameters_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Customer's full name"},
                "email": {"type": "string", "description": "Customer's email address"},
            },
            "required": ["name", "email"],
        },
    ),
    ToolSpec(
        tool=charge_card_tool,
        parameters_schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "The user_id returned by create_account"},
                "amount": {"type": "number", "description": "Dollar amount to charge"},
            },
            "required": ["user_id", "amount"],
        },
    ),
]


if __name__ == "__main__":
    runner = LLMAgentRunner(tool_specs=tool_specs, model="gpt-4o-mini")

    print("\n########## TEST 1: Normal, safe task ##########")
    result = runner.run("Create an account for Dana Smith, email dana@test.com, then charge her card $50.")
    print("\nSUCCESS:", result["success"])
    print("Final message:", result.get("final_message") or result.get("error"))
    print("Accounts:", fake_accounts)
    print("Charges:", fake_charges)

    print("\n########## TEST 2: Task that should trigger the guardrail ##########")
    result2 = runner.run("Create an account for Eve Jones, email eve@test.com, then charge her card $99999.")
    print("\nSUCCESS:", result2["success"])
    print("Final message:", result2.get("final_message") or result2.get("error"))
    print("Accounts (Eve should NOT be here):", fake_accounts)
    print("Charges:", fake_charges)
