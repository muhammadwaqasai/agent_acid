"""
agent_acid.guardrails
======================
Catches "silent" failures — cases where a step technically succeeds
(no Python exception) but the RESULT is wrong, unsafe, or violates a
business rule (e.g. AI hallucinated a bad email, negative price, or
tried to touch a forbidden field).

A guardrail is just a function: (kwargs, result) -> None
It should raise a GuardrailViolation if something is wrong.
Raising this exception triggers the SAME rollback path as a crash.
"""

from typing import Callable, Any, Dict, List
from dataclasses import dataclass, field


class GuardrailViolation(Exception):
    """Raised when a step's output breaks a business rule / safety check."""
    pass


# A guardrail function signature: (kwargs, result) -> None (raises if invalid)
GuardrailFn = Callable[[Dict[str, Any], Any], None]


@dataclass
class Guardrail:
    name: str
    check: GuardrailFn


# ---------------------------------------------------------------
# A small library of ready-to-use guardrails you can attach to any tool
# ---------------------------------------------------------------

def no_negative_numbers(field_name: str) -> Guardrail:
    """Fails if result[field_name] is negative."""
    def check(kwargs, result):
        value = result.get(field_name) if isinstance(result, dict) else None
        if value is not None and value < 0:
            raise GuardrailViolation(
                f"Field '{field_name}' is negative ({value}) — likely a hallucinated/bad value"
            )
    return Guardrail(name=f"no_negative:{field_name}", check=check)


def valid_email(field_name: str) -> Guardrail:
    """Fails if result[field_name] doesn't look like a real email."""
    def check(kwargs, result):
        value = result.get(field_name) if isinstance(result, dict) else None
        if value is not None and ("@" not in value or "." not in value.split("@")[-1]):
            raise GuardrailViolation(
                f"Field '{field_name}' = '{value}' is not a valid-looking email"
            )
    return Guardrail(name=f"valid_email:{field_name}", check=check)


def max_value(field_name: str, limit: float) -> Guardrail:
    """Fails if result[field_name] exceeds a hard business limit (e.g. spend cap)."""
    def check(kwargs, result):
        value = result.get(field_name) if isinstance(result, dict) else None
        if value is not None and value > limit:
            raise GuardrailViolation(
                f"Field '{field_name}' = {value} exceeds allowed limit of {limit}"
            )
    return Guardrail(name=f"max_value:{field_name}<= {limit}", check=check)


def forbidden_keys(keys: List[str]) -> Guardrail:
    """Fails if the AI's kwargs try to touch fields it should never touch
    (e.g. 'is_admin', 'password_hash', 'account_balance')."""
    def check(kwargs, result):
        touched = [k for k in keys if k in kwargs]
        if touched:
            raise GuardrailViolation(
                f"Attempted to modify forbidden field(s): {touched}"
            )
    return Guardrail(name=f"forbidden_keys:{keys}", check=check)


# ---------------------------------------------------------------
# STATEFUL GUARDRAILS
# These look at the WHOLE transaction so far, not just one step.
# This is what catches "salami slicing": an attacker (or a manipulated
# AI) splitting one big forbidden action into several small ones that
# each individually look fine, but add up to something dangerous.
# ---------------------------------------------------------------

@dataclass
class StatefulGuardrail:
    name: str
    check: Callable[[Any, Dict[str, Any], Any], None]  # (ctx, kwargs, result) -> None, raises on violation


def cumulative_max(tool_name: str, field_name: str, session_limit: float) -> StatefulGuardrail:
    """
    Tracks the running TOTAL of `field_name` across every past call to
    `tool_name` in this same transaction, plus the current one. If the
    running total exceeds `session_limit`, blocks and rolls back
    EVERYTHING in the session — even though every individual step was
    under the per-step limit.

    Example: block if total charges across the whole session exceed $1000,
    even if each individual charge was only $400.
    """
    def check(ctx, kwargs, result):
        total = 0
        for step in ctx.history:
            if step.tool.name == tool_name and isinstance(step.result, dict):
                total += step.result.get(field_name, 0)
        if total > session_limit:
            raise GuardrailViolation(
                f"Cumulative '{field_name}' across '{tool_name}' calls this session "
                f"is {total}, exceeding session limit of {session_limit} "
                f"(possible salami-slicing attempt)"
            )
    return StatefulGuardrail(name=f"cumulative_max:{tool_name}.{field_name}<={session_limit}", check=check)


def run_stateful_guardrails(guardrails: List[StatefulGuardrail], ctx, kwargs: Dict[str, Any], result: Any):
    for g in guardrails:
        g.check(ctx, kwargs, result)


def run_guardrails(guardrails: List[Guardrail], kwargs: Dict[str, Any], result: Any):
    """Runs every guardrail attached to a step. Raises on first violation."""
    for g in guardrails:
        g.check(kwargs, result)
