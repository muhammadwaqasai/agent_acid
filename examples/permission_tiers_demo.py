"""
examples/permission_tiers_demo.py

Proves the 3-tier risk system:
  GREEN  -> executes immediately, no extra check
  YELLOW -> must pass a verify_fn before executing
  RED    -> pauses completely, waits for a human to approve/reject

Run with:
    python examples/permission_tiers_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_acid.core import ReversibleTool, TransactionContext, AgentTransactionEngine
from agent_acid.permissions import RiskLevel

real_world = {"emails_sent": [], "accounts_updated": [], "refunds": []}


# --- GREEN: safe, low-stakes action, auto-executes ---
def send_email_execute(kwargs):
    real_world["emails_sent"].append(kwargs["to"])
    return {"to": kwargs["to"]}

def send_email_compensate(kwargs, result):
    real_world["emails_sent"].remove(result["to"])

send_email_tool = ReversibleTool(
    name="send_reply_email", description="Sends a routine reply email",
    execute=send_email_execute, compensate=send_email_compensate,
    risk_level=RiskLevel.GREEN,
)


# --- YELLOW: medium risk, needs a verify_fn to pass first ---
def update_account_execute(kwargs):
    real_world["accounts_updated"].append(kwargs["field"])
    return {"field": kwargs["field"]}

def update_account_compensate(kwargs, result):
    real_world["accounts_updated"].remove(result["field"])

def verify_account_update(kwargs):
    # Simple business rule: never allow changing 'email' or 'password' via this path
    if kwargs["field"] in ("email", "password"):
        return False, f"Field '{kwargs['field']}' cannot be changed through this workflow"
    return True, "OK"

update_account_tool = ReversibleTool(
    name="update_customer_account", description="Updates a customer account field",
    execute=update_account_execute, compensate=update_account_compensate,
    risk_level=RiskLevel.YELLOW, verify_fn=verify_account_update,
)


# --- RED: high risk, requires explicit human approval before ANY execution ---
def refund_execute(kwargs):
    real_world["refunds"].append(kwargs["amount"])
    print(f"   [REAL WORLD] Refund issued: ${kwargs['amount']}")
    return {"amount": kwargs["amount"]}

def refund_compensate(kwargs, result):
    real_world["refunds"].remove(result["amount"])

refund_tool = ReversibleTool(
    name="issue_refund", description="Issues a refund to a customer",
    execute=refund_execute, compensate=refund_compensate,
    risk_level=RiskLevel.RED,
    risk_reason="Refunds over $0 always require human sign-off before processing.",
)


engine = AgentTransactionEngine()

print("\n########## STEP 1: GREEN action (auto-executes) ##########")
ctx1 = TransactionContext()
engine.execute_plan(ctx1, [(send_email_tool, {"to": "customer@test.com"})])
print("Result:", real_world)

print("\n########## STEP 2: YELLOW action that PASSES verification ##########")
ctx2 = TransactionContext()
engine.execute_plan(ctx2, [(update_account_tool, {"field": "shipping_address"})])
print("Result:", real_world)

print("\n########## STEP 3: YELLOW action that FAILS verification ##########")
ctx3 = TransactionContext()
engine.execute_plan(ctx3, [(update_account_tool, {"field": "password"})])
print("Result (password should NOT be in accounts_updated):", real_world)

print("\n########## STEP 4: RED action -- pauses, does NOT execute yet ##########")
ctx4 = TransactionContext()
engine.execute_plan(ctx4, [(refund_tool, {"amount": 500})])
print("Status:", ctx4.status)
print("Refunds so far (should be EMPTY -- nothing executed yet):", real_world["refunds"])

print("\n########## STEP 5: Human APPROVES the pending refund ##########")
engine.approve_pending(ctx4)
print("Refunds now (should show 500 -- executed only after approval):", real_world["refunds"])

print("\n########## STEP 6: A second RED action, this time REJECTED ##########")
ctx5 = TransactionContext()
engine.execute_plan(ctx5, [(refund_tool, {"amount": 99999})])
engine.reject_pending(ctx5, reason="Amount looks suspicious, needs manual review")
print("Refunds after rejection (99999 should NEVER appear):", real_world["refunds"])

print("\n--- FINAL REAL WORLD STATE ---")
print(real_world)
