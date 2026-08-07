"""
examples/shadow_execution_demo.py

Proves the shadow execution layer: a plan gets fully simulated BEFORE
any real action happens. If guardrails would fail, or a judge rejects
the plan, NOTHING ever touches the real world -- not even a
create-then-rollback cycle. This is prevention, not cleanup.

Run with:
    python examples/shadow_execution_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_acid.core import ReversibleTool
from agent_acid.guardrails import no_negative_numbers, max_value, cumulative_max
from agent_acid.shadow import ShadowExecutionEngine


# --- Real world (should NEVER be touched if shadow mode rejects the plan) ---
real_accounts = {}
real_charges = []


def create_account_execute(kwargs):
    user_id = kwargs["name"].lower()
    real_accounts[user_id] = {"name": kwargs["name"]}
    print(f"   [REAL WORLD] Account created: {user_id}")
    return {"user_id": user_id}

def create_account_compensate(kwargs, result):
    real_accounts.pop(result["user_id"], None)

def create_account_simulate(kwargs):
    # Pure prediction, no real account touched
    return {"user_id": kwargs["name"].lower()}

create_account_tool = ReversibleTool(
    name="create_account", description="",
    execute=create_account_execute, compensate=create_account_compensate,
)
create_account_tool.simulate = create_account_simulate  # attach simulate fn


def charge_card_execute(kwargs):
    charge = {"user_id": kwargs["user_id"], "amount": kwargs["amount"]}
    real_charges.append(charge)
    print(f"   [REAL WORLD] Card charged: ${kwargs['amount']}")
    return charge

def charge_card_compensate(kwargs, result):
    real_charges.append({"user_id": result["user_id"], "amount": -result["amount"]})

def charge_card_simulate(kwargs):
    # Pure prediction: what WOULD happen, no real charge
    return {"user_id": kwargs["user_id"], "amount": kwargs["amount"]}

charge_card_tool = ReversibleTool(
    name="charge_card", description="",
    execute=charge_card_execute, compensate=charge_card_compensate,
    guardrails=[no_negative_numbers("amount"), max_value("amount", limit=500)],
    stateful_guardrails=[cumulative_max("charge_card", "amount", session_limit=1000)],
)
charge_card_tool.simulate = charge_card_simulate


engine = ShadowExecutionEngine()  # no LLM judge for this demo -- guardrails alone are enough to prove the point

print("\n########## PLAN: salami-sliced $1200 charge (3x $400) ##########\n")
plan = [
    (create_account_tool, {"name": "Grace Lee"}),
    (charge_card_tool, {"user_id": "grace lee", "amount": 400}),
    (charge_card_tool, {"user_id": "grace lee", "amount": 400}),
    (charge_card_tool, {"user_id": "grace lee", "amount": 400}),  # total 1200 -> should be caught in SHADOW mode
]

result = engine.run(plan, plan_description="Set up a customer account and process their subscription payment.")

print("\n--- RESULT ---")
print("Executed for real:", result["executed"])
if not result["executed"]:
    print("Rejected in shadow mode. Reason:", result["reason"])

print("\n--- REAL WORLD STATE (the whole point of this test) ---")
print("Real accounts created:", real_accounts)
print("Real charges made:", real_charges)

print("\n--- VERDICT ---")
if not result["executed"] and not real_accounts and not real_charges:
    print("PREVENTED: The bad plan was caught in shadow mode. The real world was NEVER touched.")
    print("Compare this to rollback-only systems, where the account and first two charges")
    print("would have been created for real, then undone after the fact.")
else:
    print("REVIEW NEEDED: real world was touched, check output above.")
