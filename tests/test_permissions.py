"""
tests/test_permissions.py

Proves the risk-tiered permission layer:
- GREEN executes immediately
- YELLOW requires verify_fn to pass; failing it blocks + rolls back
- RED pauses entirely -- nothing executes until explicit human approval
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent_acid.core import ReversibleTool, TransactionContext, AgentTransactionEngine, TransactionStatus
from agent_acid.permissions import RiskLevel


@pytest.fixture
def world():
    return {"log": []}


def make_tool(world, risk_level, verify_fn=None, risk_reason=""):
    def execute(kwargs):
        world["log"].append(kwargs["val"])
        return {"val": kwargs["val"]}

    def compensate(kwargs, result):
        world["log"].remove(result["val"])

    return ReversibleTool(
        name="action", description="", execute=execute, compensate=compensate,
        risk_level=risk_level, verify_fn=verify_fn, risk_reason=risk_reason,
    )


def test_green_executes_immediately(world):
    tool = make_tool(world, RiskLevel.GREEN)
    engine = AgentTransactionEngine()
    ctx = TransactionContext()
    success = engine.execute_plan(ctx, [(tool, {"val": "a"})])

    assert success is True
    assert world["log"] == ["a"]


def test_yellow_passes_when_verified(world):
    tool = make_tool(world, RiskLevel.YELLOW, verify_fn=lambda kwargs: (True, "ok"))
    engine = AgentTransactionEngine()
    ctx = TransactionContext()
    success = engine.execute_plan(ctx, [(tool, {"val": "b"})])

    assert success is True
    assert world["log"] == ["b"]


def test_yellow_blocks_when_verification_fails(world):
    tool = make_tool(world, RiskLevel.YELLOW, verify_fn=lambda kwargs: (False, "not allowed"))
    engine = AgentTransactionEngine()
    ctx = TransactionContext()
    success = engine.execute_plan(ctx, [(tool, {"val": "c"})])

    assert success is False
    assert world["log"] == []  # never executed at all


def test_red_pauses_and_does_not_execute(world):
    tool = make_tool(world, RiskLevel.RED, risk_reason="needs sign-off")
    engine = AgentTransactionEngine()
    ctx = TransactionContext()
    success = engine.execute_plan(ctx, [(tool, {"val": "d"})])

    assert success is False
    assert ctx.status == TransactionStatus.PENDING_APPROVAL
    assert ctx.pending_action is not None
    assert world["log"] == []  # THE KEY ASSERTION: nothing executed yet


def test_red_executes_only_after_approval(world):
    tool = make_tool(world, RiskLevel.RED)
    engine = AgentTransactionEngine()
    ctx = TransactionContext()
    engine.execute_plan(ctx, [(tool, {"val": "e"})])

    assert world["log"] == []  # still nothing
    engine.approve_pending(ctx)
    assert world["log"] == ["e"]  # NOW it executed


def test_red_rejected_never_executes(world):
    tool = make_tool(world, RiskLevel.RED)
    engine = AgentTransactionEngine()
    ctx = TransactionContext()
    engine.execute_plan(ctx, [(tool, {"val": "f"})])

    engine.reject_pending(ctx, reason="looks wrong")
    assert world["log"] == []  # never executed, and never will be
    assert ctx.pending_action is None  # cleared after rejection
