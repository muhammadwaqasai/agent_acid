"""
tests/test_shadow.py

Proves the shadow execution engine's core guarantee: a plan that would
violate a guardrail is rejected BEFORE any real-world side effect
happens -- not created-then-rolled-back, but never touched at all.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agent_acid.core import ReversibleTool
from agent_acid.guardrails import max_value, cumulative_max
from agent_acid.shadow import ShadowExecutionEngine


@pytest.fixture
def real_world():
    return {"accounts": {}, "charges": []}


def make_tools(real_world):
    def create_execute(kwargs):
        real_world["accounts"][kwargs["name"]] = True
        return {"user_id": kwargs["name"]}

    def create_compensate(kwargs, result):
        real_world["accounts"].pop(result["user_id"], None)

    def create_simulate(kwargs):
        return {"user_id": kwargs["name"]}

    account_tool = ReversibleTool(
        name="create_account", description="",
        execute=create_execute, compensate=create_compensate,
    )
    account_tool.simulate = create_simulate

    def charge_execute(kwargs):
        real_world["charges"].append(kwargs["amount"])
        return {"amount": kwargs["amount"]}

    def charge_compensate(kwargs, result):
        real_world["charges"].append(-result["amount"])

    def charge_simulate(kwargs):
        return {"amount": kwargs["amount"]}

    charge_tool = ReversibleTool(
        name="charge_card", description="",
        execute=charge_execute, compensate=charge_compensate,
        guardrails=[max_value("amount", limit=500)],
        stateful_guardrails=[cumulative_max("charge_card", "amount", session_limit=1000)],
    )
    charge_tool.simulate = charge_simulate

    return account_tool, charge_tool


def test_bad_plan_never_touches_real_world(real_world):
    account_tool, charge_tool = make_tools(real_world)
    engine = ShadowExecutionEngine()

    plan = [
        (account_tool, {"name": "Grace"}),
        (charge_tool, {"amount": 400}),
        (charge_tool, {"amount": 400}),
        (charge_tool, {"amount": 400}),  # total 1200 -> should be caught in shadow mode
    ]
    result = engine.run(plan)

    assert result["executed"] is False
    # THE KEY ASSERTION: real world must be completely untouched, not created-then-rolled-back
    assert real_world["accounts"] == {}
    assert real_world["charges"] == []


def test_good_plan_executes_for_real(real_world):
    account_tool, charge_tool = make_tools(real_world)
    engine = ShadowExecutionEngine()

    plan = [
        (account_tool, {"name": "Alice"}),
        (charge_tool, {"amount": 300}),
    ]
    result = engine.run(plan)

    assert result["executed"] is True
    assert result["success"] is True
    assert "Alice" in real_world["accounts"]
    assert real_world["charges"] == [300]


def test_shadow_fallback_when_no_simulate_fn(real_world):
    """Tools without a simulate() function should still work via the
    execute+immediate-compensate fallback, leaving no net real-world change."""
    def execute(kwargs):
        real_world["charges"].append(kwargs["amount"])
        return {"amount": kwargs["amount"]}

    def compensate(kwargs, result):
        real_world["charges"].append(-result["amount"])

    # Deliberately NOT attaching a .simulate function
    tool = ReversibleTool(
        name="charge_card", description="",
        execute=execute, compensate=compensate,
        guardrails=[max_value("amount", limit=500)],
    )

    engine = ShadowExecutionEngine()
    plan = [(tool, {"amount": 9999})]  # violates guardrail
    result = engine.run(plan)

    assert result["executed"] is False
    # Fallback executed-then-compensated during the shadow pass itself,
    # so net effect should cancel out to zero.
    assert sum(real_world["charges"]) == 0
