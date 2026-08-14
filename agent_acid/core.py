"""
agent_acid.core
================
The heart of the "Undo Engine" for AI Agents.

Building blocks:
1. ReversibleTool   - a tool that knows how to execute AND undo itself,
                       with an optional risk tier (GREEN/YELLOW/RED)
2. TransactionContext - keeps a history/log of everything that happened,
                       plus any action currently paused for approval
3. AgentTransactionEngine - runs steps; handles rollback, guardrails,
                       and now risk-tiered permission checks
"""

import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any, Dict, List, Tuple, Optional

from agent_acid.guardrails import Guardrail, run_guardrails, GuardrailViolation, StatefulGuardrail, run_stateful_guardrails
from agent_acid.permissions import RiskLevel, check_permission, ApprovalRequired, VerificationFailed, PendingAction

logger = logging.getLogger("agent_acid")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class TransactionStatus(Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    PENDING_APPROVAL = "pending_approval"


@dataclass
class ReversibleTool:
    """
    A wrapper around any action that also knows how to undo itself.

    risk_level : GREEN (default, auto-execute), YELLOW (needs verify_fn
                 to pass), or RED (needs explicit human approval)
    verify_fn  : for YELLOW tools -- (kwargs) -> (ok: bool, reason: str)
    risk_reason: for RED tools -- shown to whoever approves/rejects
    """
    name: str
    description: str
    execute: Callable[[Dict[str, Any]], Any]
    compensate: Callable[[Dict[str, Any], Any], None]
    guardrails: List[Guardrail] = field(default_factory=list)
    stateful_guardrails: List[StatefulGuardrail] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.GREEN
    verify_fn: Optional[Callable[[Dict[str, Any]], Tuple[bool, str]]] = None
    risk_reason: str = ""


@dataclass
class StepRecord:
    tool: ReversibleTool
    kwargs: Dict[str, Any]
    result: Any


@dataclass
class TransactionContext:
    """Tracks everything that has happened in one 'plan' execution."""
    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    history: List[StepRecord] = field(default_factory=list)
    status: TransactionStatus = TransactionStatus.PENDING
    pending_action: Optional[PendingAction] = None

    def log_step(self, tool: ReversibleTool, kwargs: dict, result: Any):
        self.history.append(StepRecord(tool=tool, kwargs=kwargs, result=result))


class AgentTransactionEngine:
    """
    The supervisor. Executes steps, checks guardrails, checks risk-tier
    permissions, and rolls back everything on failure.
    """

    def execute_plan(
        self,
        ctx: TransactionContext,
        steps: List[Tuple[ReversibleTool, Dict[str, Any]]],
    ) -> bool:
        logger.info(f"\n=== Starting transaction {ctx.transaction_id} ===")

        for tool, kwargs in steps:
            # Permission check happens BEFORE anything touches the real world
            try:
                pending = check_permission(tool, kwargs)
            except VerificationFailed as e:
                logger.info(f"[VERIFICATION FAILED] {tool.name}: {e}")
                self._rollback(ctx)
                ctx.status = TransactionStatus.FAILED
                logger.info(f"=== Transaction {ctx.transaction_id} FAILED VERIFICATION & ROLLED BACK ===\n")
                return False

            if pending is not None:
                logger.info(f"[PAUSED FOR APPROVAL] {tool.name}: {pending.reason}")
                ctx.pending_action = pending
                ctx.status = TransactionStatus.PENDING_APPROVAL
                logger.info(f"=== Transaction {ctx.transaction_id} PAUSED — awaiting human approval ===\n")
                return False  # caller should check ctx.status / ctx.pending_action

            logger.info(f"[EXECUTING] {tool.name} with {kwargs}")
            try:
                result = tool.execute(kwargs)
                ctx.log_step(tool, kwargs, result)
                if tool.guardrails:
                    run_guardrails(tool.guardrails, kwargs, result)
                if tool.stateful_guardrails:
                    run_stateful_guardrails(tool.stateful_guardrails, ctx, kwargs, result)
                logger.info(f"   -> OK: {result}")
            except GuardrailViolation as e:
                logger.info(f"[GUARDRAIL BLOCKED] {tool.name}: {e}")
                logger.info("[ROLLBACK TRIGGERED] Undoing completed steps (including this one)...")
                self._rollback(ctx)
                ctx.status = TransactionStatus.FAILED
                logger.info(f"=== Transaction {ctx.transaction_id} BLOCKED BY GUARDRAIL & ROLLED BACK ===\n")
                return False
            except Exception as e:
                logger.info(f"[FAILED] {tool.name} raised: {e}")
                logger.info("[ROLLBACK TRIGGERED] Undoing completed steps...")
                self._rollback(ctx)
                ctx.status = TransactionStatus.FAILED
                logger.info(f"=== Transaction {ctx.transaction_id} FAILED & ROLLED BACK ===\n")
                return False

        ctx.status = TransactionStatus.COMMITTED
        logger.info(f"=== Transaction {ctx.transaction_id} COMMITTED successfully ===\n")
        return True

    def approve_pending(self, ctx: TransactionContext) -> Any:
        """A human approved the held RED action. Execute it for real now."""
        if ctx.pending_action is None:
            raise ValueError("No pending action to approve on this context.")

        pending = ctx.pending_action
        logger.info(f"[APPROVED BY HUMAN] Executing previously-held action: {pending.tool.name}")
        ctx.pending_action = None
        ctx.status = TransactionStatus.PENDING

        logger.info(f"[EXECUTING] {pending.tool.name} with {pending.kwargs}")
        result = pending.tool.execute(pending.kwargs)
        ctx.log_step(pending.tool, pending.kwargs, result)
        if pending.tool.guardrails:
            run_guardrails(pending.tool.guardrails, pending.kwargs, result)
        if pending.tool.stateful_guardrails:
            run_stateful_guardrails(pending.tool.stateful_guardrails, ctx, pending.kwargs, result)
        logger.info(f"   -> OK: {result}")
        return result

    def reject_pending(self, ctx: TransactionContext, reason: str = ""):
        """A human rejected the pending action. It was NEVER executed,
        so there's nothing to roll back -- just clear the pending state."""
        if ctx.pending_action is None:
            raise ValueError("No pending action to reject on this context.")

        logger.info(f"[REJECTED BY HUMAN] {ctx.pending_action.tool.name} will NOT be executed. "
                    f"Reason: {reason or 'not specified'}")
        ctx.pending_action = None
        ctx.status = TransactionStatus.FAILED

    def execute_step(self, ctx: TransactionContext, tool: ReversibleTool, kwargs: Dict[str, Any]) -> Any:
        pending = check_permission(tool, kwargs)
        if pending is not None:
            ctx.pending_action = pending
            ctx.status = TransactionStatus.PENDING_APPROVAL
            raise ApprovalRequired(pending)

        logger.info(f"[EXECUTING] {tool.name} with {kwargs}")
        result = tool.execute(kwargs)
        ctx.log_step(tool, kwargs, result)
        if tool.guardrails:
            run_guardrails(tool.guardrails, kwargs, result)
        if tool.stateful_guardrails:
            run_stateful_guardrails(tool.stateful_guardrails, ctx, kwargs, result)
        logger.info(f"   -> OK: {result}")
        return result

    def commit(self, ctx: TransactionContext):
        ctx.status = TransactionStatus.COMMITTED
        logger.info(f"=== Transaction {ctx.transaction_id} COMMITTED successfully ===\n")

    def abort(self, ctx: TransactionContext):
        logger.info("[ROLLBACK TRIGGERED] Undoing completed steps...")
        self._rollback(ctx)
        ctx.status = TransactionStatus.FAILED
        logger.info(f"=== Transaction {ctx.transaction_id} ABORTED & ROLLED BACK ===\n")

    def _rollback(self, ctx: TransactionContext):
        while ctx.history:
            step = ctx.history.pop()
            logger.info(f"[UNDO] {step.tool.name}")
            try:
                step.tool.compensate(step.kwargs, step.result)
                logger.info(f"   -> undone OK")
            except Exception as rollback_err:
                logger.info(f"   -> CRITICAL: compensation failed: {rollback_err}")
        ctx.status = TransactionStatus.ROLLED_BACK
