"""
agent_acid.permissions
========================
THE PERMISSION LAYER: not every action an AI agent takes carries the
same risk. This module adds a risk tier to each tool:

  GREEN  -> auto-execute, same as agent_acid always did
  YELLOW -> must pass a verify_fn check first (e.g. an LLM-as-judge,
            or a simple business rule) before it's allowed to run
  RED    -> execution PAUSES entirely. Nothing happens -- not even a
            create-then-rollback -- until a human explicitly approves
            or rejects the held action.

This is the difference between "the system decided this was safe" and
"a human confirmed this specific risky action before it happened."
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from enum import Enum


class RiskLevel(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class PendingAction:
    """A RED-risk action that has been evaluated but NOT executed,
    waiting for a human to approve or reject it."""
    tool: Any          # the ReversibleTool (kept generic here to avoid circular import)
    kwargs: Dict[str, Any]
    reason: str


class ApprovalRequired(Exception):
    """Raised when a RED-risk action is hit. Carries the PendingAction
    so the caller can surface it to a human (dashboard, Slack message,
    CLI prompt, etc.) for approval."""
    def __init__(self, pending: PendingAction):
        self.pending = pending
        super().__init__(f"Action '{pending.tool.name}' requires human approval: {pending.reason}")


class VerificationFailed(Exception):
    """Raised when a YELLOW-risk action's verify_fn rejects it."""
    pass


def check_permission(tool, kwargs: Dict[str, Any]) -> Optional[PendingAction]:
    """
    Checks a tool's risk_level before it's allowed to execute.

    Returns:
      - None if the action is cleared to run right now (GREEN, or
        YELLOW that passed verification)
      - A PendingAction if this is a RED action that must be paused
        for human approval (the caller should raise ApprovalRequired)

    Raises:
      - VerificationFailed if this is a YELLOW action whose verify_fn
        rejected it
    """
    if tool.risk_level == RiskLevel.GREEN:
        return None

    if tool.risk_level == RiskLevel.YELLOW:
        if tool.verify_fn is None:
            # No verifier attached -- treat as GREEN rather than silently failing
            return None
        ok, reason = tool.verify_fn(kwargs)
        if not ok:
            raise VerificationFailed(f"'{tool.name}' failed verification: {reason}")
        return None

    if tool.risk_level == RiskLevel.RED:
        reason = tool.risk_reason or "This action is marked as high-risk and requires explicit approval."
        return PendingAction(tool=tool, kwargs=kwargs, reason=reason)

    return None
