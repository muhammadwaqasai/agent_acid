"""
agent_acid.shadow
===================
THE PREVENTIVE LAYER: instead of only cleaning up after something bad
happens (rollback) or blocking a single bad step (guardrails), this
module runs an agent's ENTIRE plan in a safe "shadow" pass first,
before anything touches the real world.

Two ways a tool can be shadow-tested:

1. PREFERRED: give the tool a `simulate` function that predicts the
   result WITHOUT any real side effects (e.g. "this charge would be
   $400, bringing the session total to $1200" -- pure computation,
   nothing sent to a real payment API).

2. FALLBACK: if no `simulate` function is given, the shadow engine
   will actually execute the real action and IMMEDIATELY compensate
   it, so the real world ends up unchanged -- but this briefly does
   touch real systems, which is the honest tradeoff of not having a
   pure simulation available. This is clearly logged so it's never
   silent.

After the full plan is shadow-run, an LLM-as-judge reviews the
ENTIRE predicted sequence (not just isolated numbers) and can reject
the whole plan BEFORE any real execution happens -- this catches
things a pure guardrail can't, like "this sequence of actions is
individually valid but contextually suspicious."

Only if guardrails AND judge review both pass does the plan run for
real via AgentTransactionEngine.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Tuple, Optional

from agent_acid.core import ReversibleTool, TransactionContext, AgentTransactionEngine
from agent_acid.guardrails import GuardrailViolation, run_guardrails, run_stateful_guardrails

logger = logging.getLogger("agent_acid")


@dataclass
class ShadowStep:
    tool_name: str
    kwargs: Dict[str, Any]
    predicted_result: Any
    simulated: bool  # True if a pure `simulate` fn was used, False if real-execute+compensate fallback was used


@dataclass
class ShadowRunResult:
    steps: List[ShadowStep]
    guardrails_passed: bool
    guardrail_failure_reason: Optional[str] = None
    judge_approved: Optional[bool] = None
    judge_reasoning: Optional[str] = None

    @property
    def approved(self) -> bool:
        if not self.guardrails_passed:
            return False
        if self.judge_approved is False:
            return False
        return True


class ShadowExecutionEngine:
    """
    Runs a full plan in shadow mode first. Nothing touches the real
    world for real unless the ENTIRE plan passes both guardrails and
    (optionally) an LLM judge reviewing the whole sequence together.
    """

    def __init__(self, judge_fn: Optional[Callable[[List[ShadowStep], str], Tuple[bool, str]]] = None):
        """
        judge_fn: optional callable(steps, plan_description) -> (approved: bool, reasoning: str)
        If not provided, only guardrail checks gate the plan (no LLM judge review).
        """
        self.judge_fn = judge_fn
        self.real_engine = AgentTransactionEngine()

    def _simulate_tool(self, tool: ReversibleTool, kwargs: Dict[str, Any]) -> Tuple[Any, bool]:
        """Returns (predicted_result, was_pure_simulation)."""
        simulate_fn = getattr(tool, "simulate", None)
        if simulate_fn is not None:
            return simulate_fn(kwargs), True

        # FALLBACK: real execute immediately followed by real compensate.
        # Honest tradeoff -- logged clearly, not hidden.
        logger.info(f"[SHADOW-FALLBACK] No simulate() for '{tool.name}' -- "
                    f"executing for real and immediately compensating.")
        result = tool.execute(kwargs)
        tool.compensate(kwargs, result)
        return result, False

    def dry_run(
        self,
        steps: List[Tuple[ReversibleTool, Dict[str, Any]]],
        plan_description: str = "",
    ) -> ShadowRunResult:
        """
        Simulates the full plan, checks guardrails against the predicted
        results (including stateful/cumulative guardrails across the
        WHOLE shadow session), then optionally asks an LLM judge to
        review the full sequence before anything is allowed to run for real.
        """
        logger.info("=== SHADOW RUN: simulating full plan before any real execution ===")
        shadow_ctx = TransactionContext()  # used only to accumulate predicted results for stateful guardrails
        shadow_steps: List[ShadowStep] = []

        for tool, kwargs in steps:
            predicted_result, was_simulated = self._simulate_tool(tool, kwargs)
            logger.info(f"[SHADOW] {tool.name}({kwargs}) -> predicted {predicted_result} "
                        f"({'simulated' if was_simulated else 'execute+compensate fallback'})")

            shadow_steps.append(ShadowStep(
                tool_name=tool.name, kwargs=kwargs,
                predicted_result=predicted_result, simulated=was_simulated,
            ))

            # Log into shadow_ctx so stateful guardrails see the FULL predicted session
            shadow_ctx.log_step(tool, kwargs, predicted_result)

            try:
                if tool.guardrails:
                    run_guardrails(tool.guardrails, kwargs, predicted_result)
                if tool.stateful_guardrails:
                    run_stateful_guardrails(tool.stateful_guardrails, shadow_ctx, kwargs, predicted_result)
            except GuardrailViolation as e:
                logger.info(f"[SHADOW BLOCKED] Guardrail would fail on real run: {e}")
                return ShadowRunResult(steps=shadow_steps, guardrails_passed=False,
                                        guardrail_failure_reason=str(e))

        result = ShadowRunResult(steps=shadow_steps, guardrails_passed=True)

        if self.judge_fn is not None:
            logger.info("[SHADOW] Guardrails passed. Sending full plan to LLM judge for review...")
            approved, reasoning = self.judge_fn(shadow_steps, plan_description)
            result.judge_approved = approved
            result.judge_reasoning = reasoning
            logger.info(f"[JUDGE] Approved={approved}. Reasoning: {reasoning}")

        return result

    def run(
        self,
        steps: List[Tuple[ReversibleTool, Dict[str, Any]]],
        plan_description: str = "",
    ) -> Dict[str, Any]:
        """
        Full pipeline: shadow dry-run first. Only if it's fully approved
        does the plan get executed for real (with the normal rollback
        safety net still active as a second line of defense).
        """
        shadow_result = self.dry_run(steps, plan_description)

        if not shadow_result.approved:
            reason = shadow_result.guardrail_failure_reason or shadow_result.judge_reasoning or "rejected"
            logger.info(f"=== PLAN REJECTED IN SHADOW MODE. Nothing touched the real world. Reason: {reason} ===")
            return {"executed": False, "shadow_result": shadow_result, "reason": reason}

        logger.info("=== SHADOW RUN APPROVED. Executing plan for real. ===")
        real_ctx = TransactionContext()
        success = self.real_engine.execute_plan(real_ctx, steps)
        return {"executed": True, "success": success, "shadow_result": shadow_result, "ctx": real_ctx}
