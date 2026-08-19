# agent_acid

[![Tests](https://img.shields.io/badge/tests-14%20passing-brightgreen)](https://github.com/muhammadwaqasai/agent_acid)
[![PyPI](https://img.shields.io/pypi/v/agent-acid)](https://pypi.org/project/agent-acid/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**ACID-style transaction guarantees for autonomous AI agents.**

![agent_acid demo: an AI agent tries to sneak a $1,200 charge past a $500 limit by splitting it into three payments. Shadow execution catches it before a single dollar moves.](https://raw.githubusercontent.com/muhammadwaqasai/agent_acid/main/assets/agent_acid_demo.gif)

*Real, unscripted terminal output. An AI agent tries to split a $1,200 charge into three $400 payments to dodge a $500 limit. agent_acid catches the full pattern and blocks it before anything touches a real system.*

---

## The 30-second version

- **Automatic rollback** — if any step in a multi-step agent plan fails, everything already done gets undone, in reverse order.
- **Guardrails** — hard, code-level rules (not prompts) that block bad AI outputs even when nothing crashes.
- **Stateful guardrails** — catch multi-step manipulation, like an attacker splitting one large forbidden action into several small, individually-legal ones ("salami slicing").
- **Shadow execution** — simulates an agent's entire plan in a safe sandbox *before* anything touches a real system. Rejected plans never execute — not "created then undone," never touched at all.
- **Risk-tiered permissions** — every action gets a 🟢 auto / 🟡 verify-first / 🔴 human-approval-required tier. Red actions pause completely until a human explicitly approves them.

Every claim above is backed by a runnable test or live demo in this repo — 14 automated tests, all passing, zero API cost to verify.

```bash
pip install agent-acid
pytest tests/ -v   # 14 passed, no API key needed
```

---

## Why this exists

Salami-slicing–style attacks against AI agents are an actively studied problem: guardrails that only judge one tool call at a time are "memoryless," letting an attacker spread an attack across many small steps where no single step trips the alarm. `agent_acid` closes this gap with session-memory guardrails and shadow execution, built specifically to catch what per-step checks miss.

---

## Quick start

```bash
pip install -r requirements.txt   # or: pip install openai pytest
```

Run the test suite (no API key needed, proves the core engine works):
```bash
pytest tests/ -v
```

Run the demos (no API key needed):
```bash
python examples/basic_rollback.py
python examples/guardrail_demo.py
python examples/shadow_execution_demo.py
python examples/permission_tiers_demo.py
```

Run the live-AI demos (requires an OpenAI API key):
```bash
export OPENAI_API_KEY="sk-..."
python examples/llm_agent_demo.py
python examples/attack_test_prompt_injection.py
python examples/attack_test_salami_slicing.py
python examples/comparison_naive_agent.py
```

---

## The core idea

```python
from agent_acid.core import ReversibleTool, TransactionContext, AgentTransactionEngine
from agent_acid.guardrails import max_value, cumulative_max
from agent_acid.permissions import RiskLevel

charge_tool = ReversibleTool(
    name="charge_card",
    description="Charges a customer's card",
    execute=lambda kwargs: real_payment_api.charge(kwargs),
    compensate=lambda kwargs, result: real_payment_api.refund(result["charge_id"]),
    guardrails=[max_value("amount", limit=500)],
    stateful_guardrails=[cumulative_max("charge_card", "amount", session_limit=1000)],
    risk_level=RiskLevel.RED,
    risk_reason="Refunds and charges always require human sign-off.",
)

engine = AgentTransactionEngine()
ctx = TransactionContext()
engine.execute_plan(ctx, [
    (charge_tool, {"user_id": "u1", "amount": 400}),
    (charge_tool, {"user_id": "u1", "amount": 400}),
    (charge_tool, {"user_id": "u1", "amount": 400}),  # blocked: session limit + requires human approval
])
```

---

## Proven results

### Attack 1: Prompt injection
A "customer ticket" contains a fake administrator note trying to convince the AI to charge $75,000 instead of following the $500 limit. The AI gets fooled and attempts the charge — `agent_acid`'s guardrail blocks it and fully reverses the account creation and the charge.

### Attack 2: Salami slicing
The AI is instructed to charge $1,200 as three separate $400 charges specifically to stay under a $500 per-step limit. `agent_acid`'s cumulative guardrail tracks the running total across the whole session and blocks the third charge once the total crosses $1,000 — then rolls back all three charges and the account.

### Head-to-head comparison
The same salami-slicing attack was run against a naive agent using only a per-step check (representative of how most simple guardrail integrations work):

| | Naive agent | agent_acid |
|---|---|---|
| Per-step limit ($500) | Enforced | Enforced |
| 3× $400 charge attack | **All 3 succeeded — $1,200 charged** | Blocked on 3rd call |
| Rollback available? | **No — money is gone** | Yes — fully refunded |
| Account left behind? | **Yes, permanently** | Deleted during rollback |
| With shadow execution | N/A | **Never touched the real world at all** |

See `examples/comparison_naive_agent.py`, `examples/attack_test_salami_slicing.py`, and `examples/shadow_execution_demo.py` to reproduce this yourself.

---

## Project structure

```
agent_acid/
├── agent_acid/
│   ├── core.py          # TransactionContext, ReversibleTool, AgentTransactionEngine
│   ├── guardrails.py     # Guardrail + StatefulGuardrail (session-memory) rules
│   ├── shadow.py          # Shadow execution: simulate before committing
│   ├── permissions.py    # Risk-tiered permissions (green/yellow/red)
│   └── llm_agent.py      # Connects OpenAI function-calling to the engine
├── examples/              # 8 runnable demos, including live attack tests
└── tests/                 # 14 automated tests
```

## Status

Actively developed. Core engine, guardrails, shadow execution, and the permission layer are all tested and working. Contributions and adversarial testing (try to break it!) are welcome.
