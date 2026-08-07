"""
agent_acid.core
================
The heart of the "Undo Engine" for AI Agents.

Three building blocks:
1. ReversibleTool   - a tool that knows how to execute AND undo itself
2. TransactionContext - keeps a history/log of everything that happened
3. AgentTransactionEngine - runs a plan step by step; if anything fails,
   it automatically undoes everything in reverse order (LIFO)
"""

import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any, Dict, List, Tuple, Optional

from agent_acid.guardrails import Guardrail, run_guardrails, GuardrailViolation, StatefulGuardrail, run_stateful_guardrails

# --- Logging setup (so you get clean, readable output) ---
logger = logging.getLogger("agent_acid")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class TransactionStatus(Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class ReversibleTool:
    """
    A wrapper around any action that also knows how to undo itself.

    name        : human readable identifier
    description : what the tool does (useful later for LLM function-calling)
    execute     : function that performs the real action, returns a result
    compensate  : function that undoes the action, given the original input
                  and the result that execute() produced
    """
    name: str
    description: str
    execute: Callable[[Dict[str, Any]], Any]
    compensate: Callable[[Dict[str, Any], Any], None]
    guardrails: List[Guardrail] = field(default_factory=list)
    stateful_guardrails: List[StatefulGuardrail] = field(default_factory=list)


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

    def log_step(self, tool: ReversibleTool, kwargs: dict, result: Any):
        self.history.append(StepRecord(tool=tool, kwargs=kwargs, result=result))


class AgentTransactionEngine:
    """
    The supervisor. Give it a list of (tool, kwargs) steps.
    It executes them in order. If any step raises an exception,
    it automatically calls .compensate() on every already-completed
    step, in reverse order, so the system ends up exactly as it
    started.
    """

    def execute_plan(
        self,
        ctx: TransactionContext,
        steps: List[Tuple[ReversibleTool, Dict[str, Any]]],
    ) -> bool:
        logger.info(f"\n=== Starting transaction {ctx.transaction_id} ===")

        for tool, kwargs in steps:
            logger.info(f"[EXECUTING] {tool.name} with {kwargs}")
            try:
                result = tool.execute(kwargs)

                # Guardrail check happens BEFORE we consider this step "safe".
                # If it fails here, the action already happened in the real
                # world, so we log it first so rollback knows to undo it too.
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

    def execute_step(
        self,
        ctx: TransactionContext,
        tool: ReversibleTool,
        kwargs: Dict[str, Any],
    ) -> Any:
        """
        Execute ONE step live (used when a real AI agent is deciding steps
        one at a time, rather than a pre-made plan). Raises on failure —
        the caller (e.g. the LLM agent loop) decides whether to call
        abort() to roll everything back, or keep going.
        """
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
        """Mark a live transaction as successfully finished. Nothing to undo."""
        ctx.status = TransactionStatus.COMMITTED
        logger.info(f"=== Transaction {ctx.transaction_id} COMMITTED successfully ===\n")

    def abort(self, ctx: TransactionContext):
        """Publicly callable rollback — undoes everything done so far in a live transaction."""
        logger.info("[ROLLBACK TRIGGERED] Undoing completed steps...")
        self._rollback(ctx)
        ctx.status = TransactionStatus.FAILED
        logger.info(f"=== Transaction {ctx.transaction_id} ABORTED & ROLLED BACK ===\n")

    def _rollback(self, ctx: TransactionContext):
        # Undo in reverse order: last action taken is undone first
        while ctx.history:
            step = ctx.history.pop()
            logger.info(f"[UNDO] {step.tool.name}")
            try:
                step.tool.compensate(step.kwargs, step.result)
                logger.info(f"   -> undone OK")
            except Exception as rollback_err:
                # This is the one truly dangerous case: undo itself failed.
                logger.info(f"   -> CRITICAL: compensation failed: {rollback_err}")
        ctx.status = TransactionStatus.ROLLED_BACK
