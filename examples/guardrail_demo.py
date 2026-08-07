"""
examples/guardrail_demo.py

Proves the guardrail layer: a step can SUCCEED (no crash) but still
produce a bad/unsafe value. The engine must catch it and roll back
everything, exactly like a real crash would.

Run with:
    python examples/guardrail_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_acid.core import ReversibleTool, TransactionContext, AgentTransactionEngine
from agent_acid.guardrails import no_negative_numbers, valid_email, max_value, forbidden_keys


# --- Fake "real world" systems ---
fake_accounts = {}
fake_charges = []


# Tool 1: create an account (safe, no guardrail needed here)
def create_account_execute(kwargs):
    fake_accounts[kwargs["user_id"]] = {"name": kwargs["name"]}
    return {"user_id": kwargs["user_id"]}

def create_account_compensate(kwargs, result):
    fake_accounts.pop(result["user_id"], None)

create_account_tool = ReversibleTool(
    name="create_account",
    description="Creates a customer account",
    execute=create_account_execute,
    compensate=create_account_compensate,
)


# Tool 2: charge a card — this is where an AI hallucination could be dangerous.
# Imagine the AI was supposed to charge $49.99 but hallucinated $4999999.
def charge_card_execute(kwargs):
    charge = {"user_id": kwargs["user_id"], "amount": kwargs["amount"]}
    fake_charges.append(charge)
    return charge

def charge_card_compensate(kwargs, result):
    if result in fake_charges:
        fake_charges.remove(result)
        fake_charges.append({"user_id": result["user_id"], "amount": -result["amount"], "note": "REFUNDED"})

charge_card_tool = ReversibleTool(
    name="charge_card",
    description="Charges the customer's card",
    execute=charge_card_execute,
    compensate=charge_card_compensate,
    guardrails=[
        no_negative_numbers("amount"),
        max_value("amount", limit=500),   # hard business rule: never auto-charge over $500
    ],
)


engine = AgentTransactionEngine()

print("\n########## GUARDRAIL TEST: AI hallucinates a huge charge amount ##########")
ctx = TransactionContext()
plan = [
    (create_account_tool, {"user_id": "u9", "name": "Charlie"}),
    # Step technically "succeeds" (no Python error) — but the amount is insane.
    (charge_card_tool, {"user_id": "u9", "amount": 4999999}),
]
engine.execute_plan(ctx, plan)

print("Accounts AFTER rollback (Charlie should be GONE):", fake_accounts)
print("Charges AFTER rollback (should be refunded/reversed):", fake_charges)
