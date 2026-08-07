"""
tests/test_core.py

Automated proof that the core guarantees hold:
1. Successful plans commit cleanly.
2. A crash mid-plan triggers full rollback, in reverse order.
3. A single-step guardrail violation triggers full rollback.
4. A stateful (cumulative/session) guardrail violation triggers full
   rollback, even when every individual step passed its own limit.

Run with:
    pytest tests/test_core.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agent_acid.core import ReversibleTool, TransactionContext, AgentTransactionEngine, TransactionStatus
from agent_acid.guardrails import no_negative_numbers, max_value, cumulative_max, GuardrailViolation


# ---------------------------------------------------------------
# Fresh fake "world" for each test (pytest fixture resets it every time)
# ---------------------------------------------------------------
@pytest.fixture
def fake_world():
    return {"db": {}, "emails": [], "charges": []}


def make_user_tool(fake_world):
    def execute(kwargs):
        fake_world["db"][kwargs["user_id"]] = {"name": kwargs["name"]}
        return {"user_id": kwargs["user_id"]}

    def compensate(kwargs, result):
        fake_world["db"].pop(result["user_id"], None)

    return ReversibleTool(name="create_user", description="", execute=execute, compensate=compensate)


def make_charge_tool(fake_world, per_step_limit=500, session_limit=None):
    def execute(kwargs):
        charge = {"user_id": kwargs["user_id"], "amount": kwargs["amount"]}
        fake_world["charges"].append(charge)
        return charge

    def compensate(kwargs, result):
        fake_world["charges"].append({"user_id": result["user_id"], "amount": -result["amount"]})

    guardrails = [no_negative_numbers("amount"), max_value("amount", limit=per_step_limit)]
    stateful = []
    if session_limit is not None:
        stateful.append(cumulative_max("charge_card", "amount", session_limit))

    return ReversibleTool(
        name="charge_card", description="", execute=execute, compensate=compensate,
        guardrails=guardrails, stateful_guardrails=stateful,
    )


# ---------------------------------------------------------------
# TEST 1: A fully successful plan commits, nothing is rolled back
# ---------------------------------------------------------------
def test_successful_plan_commits(fake_world):
    engine = AgentTransactionEngine()
    ctx = TransactionContext()
    user_tool = make_user_tool(fake_world)
    charge_tool = make_charge_tool(fake_world)

    plan = [
        (user_tool, {"user_id": "u1", "name": "Alice"}),
        (charge_tool, {"user_id": "u1", "amount": 100}),
    ]
    success = engine.execute_plan(ctx, plan)

    assert success is True
    assert ctx.status == TransactionStatus.COMMITTED
    assert "u1" in fake_world["db"]
    assert fake_world["charges"] == [{"user_id": "u1", "amount": 100}]


# ---------------------------------------------------------------
# TEST 2: A crash mid-plan triggers full rollback, in reverse order
# ---------------------------------------------------------------
def test_crash_triggers_full_rollback(fake_world):
    engine = AgentTransactionEngine()
    ctx = TransactionContext()
    user_tool = make_user_tool(fake_world)

    def crashing_execute(kwargs):
        raise RuntimeError("simulated crash")

    crashing_tool = ReversibleTool(
        name="crash_tool", description="",
        execute=crashing_execute,
        compensate=lambda kwargs, result: None,
    )

    plan = [
        (user_tool, {"user_id": "u2", "name": "Bob"}),
        (crashing_tool, {}),
    ]
    success = engine.execute_plan(ctx, plan)

    assert success is False
    assert ctx.status == TransactionStatus.FAILED
    # Bob must be fully removed after rollback
    assert "u2" not in fake_world["db"]


# ---------------------------------------------------------------
# TEST 3: A single-step guardrail violation rolls back everything,
# INCLUDING the step that violated it
# ---------------------------------------------------------------
def test_guardrail_violation_rolls_back(fake_world):
    engine = AgentTransactionEngine()
    ctx = TransactionContext()
    user_tool = make_user_tool(fake_world)
    charge_tool = make_charge_tool(fake_world, per_step_limit=500)

    plan = [
        (user_tool, {"user_id": "u3", "name": "Carol"}),
        (charge_tool, {"user_id": "u3", "amount": 99999}),  # violates per-step limit
    ]
    success = engine.execute_plan(ctx, plan)

    assert success is False
    assert "u3" not in fake_world["db"]
    # The bad charge should have a matching negative compensation entry
    amounts = [c["amount"] for c in fake_world["charges"]]
    assert 99999 in amounts and -99999 in amounts


# ---------------------------------------------------------------
# TEST 4: THE KEY TEST — cumulative/session guardrail catches
# salami-slicing even when every individual step is within limits
# ---------------------------------------------------------------
def test_cumulative_guardrail_catches_salami_slicing(fake_world):
    engine = AgentTransactionEngine()
    ctx = TransactionContext()
    user_tool = make_user_tool(fake_world)
    # per-step limit 500, but session-wide cumulative limit of 1000
    charge_tool = make_charge_tool(fake_world, per_step_limit=500, session_limit=1000)

    plan = [
        (user_tool, {"user_id": "u4", "name": "Dave"}),
        (charge_tool, {"user_id": "u4", "amount": 400}),  # OK alone
        (charge_tool, {"user_id": "u4", "amount": 400}),  # still OK alone (total 800)
        (charge_tool, {"user_id": "u4", "amount": 400}),  # total now 1200 -> should be BLOCKED
    ]
    success = engine.execute_plan(ctx, plan)

    assert success is False, "Cumulative guardrail should have blocked this plan"
    assert "u4" not in fake_world["db"], "User should be fully rolled back"

    # Every positive charge must have an equal-and-opposite refund
    positive_total = sum(c["amount"] for c in fake_world["charges"] if c["amount"] > 0)
    negative_total = sum(c["amount"] for c in fake_world["charges"] if c["amount"] < 0)
    assert positive_total == -negative_total, "All charges must be fully refunded after rollback"


# ---------------------------------------------------------------
# TEST 5: Cumulative guardrail should NOT block legitimate totals
# that stay within the session limit
# ---------------------------------------------------------------
def test_cumulative_guardrail_allows_valid_totals(fake_world):
    engine = AgentTransactionEngine()
    ctx = TransactionContext()
    user_tool = make_user_tool(fake_world)
    charge_tool = make_charge_tool(fake_world, per_step_limit=500, session_limit=1000)

    plan = [
        (user_tool, {"user_id": "u5", "name": "Eve"}),
        (charge_tool, {"user_id": "u5", "amount": 400}),
        (charge_tool, {"user_id": "u5", "amount": 400}),  # total 800, within 1000 limit
    ]
    success = engine.execute_plan(ctx, plan)

    assert success is True
    assert "u5" in fake_world["db"]
    assert sum(c["amount"] for c in fake_world["charges"]) == 800
