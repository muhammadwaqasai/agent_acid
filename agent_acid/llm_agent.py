"""
agent_acid.llm_agent
======================
Connects a REAL AI agent (OpenAI's function calling / tool use) to the
AgentTransactionEngine.

How it works:
1. You register your ReversibleTools with a JSON schema describing their
   arguments (so OpenAI knows how to call them).
2. You give the agent a task in plain English (e.g. "Sign up Dana and
   charge her $50").
3. The AI decides, one step at a time, which tool to call and with what
   arguments.
4. Each step is executed through engine.execute_step() — meaning
   guardrails run automatically.
5. If any step fails (crash OR guardrail violation), the engine
   immediately rolls back everything done so far, and we tell the AI
   the transaction failed.
6. If the AI finishes all steps successfully, we commit.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List

from agent_acid.core import ReversibleTool, TransactionContext, AgentTransactionEngine


@dataclass
class ToolSpec:
    """Pairs a ReversibleTool with the JSON schema OpenAI needs to call it."""
    tool: ReversibleTool
    parameters_schema: Dict[str, Any]  # standard JSON Schema for the arguments

    def to_openai_format(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.tool.name,
                "description": self.tool.description,
                "parameters": self.parameters_schema,
            },
        }


class LLMAgentRunner:
    """
    Runs a live agent loop: AI decides steps -> engine executes + guards
    each one -> rollback automatically on any failure.
    """

    def __init__(self, tool_specs: List[ToolSpec], model: str = "gpt-4o-mini", api_key: str = None):
        # Import here so the rest of the package works even if `openai`
        # isn't installed (only needed if you actually use this class).
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.tool_specs = {spec.tool.name: spec for spec in tool_specs}
        self.engine = AgentTransactionEngine()

    def run(self, task: str, max_steps: int = 8) -> Dict[str, Any]:
        """
        Give the agent a plain-English task. It will call tools one at a
        time until it says it's done, or a step fails and everything is
        rolled back.

        Returns a dict summary: {"success": bool, "transcript": [...]}
        """
        ctx = TransactionContext()
        tools_payload = [spec.to_openai_format() for spec in self.tool_specs.values()]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an autonomous agent. Complete the user's task by calling "
                    "the available tools, one at a time. Only call a tool when you have "
                    "all the information you need. When the task is fully done, reply "
                    "with plain text (no tool call) summarizing what you did."
                ),
            },
            {"role": "user", "content": task},
        ]

        transcript = []

        for step_num in range(max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools_payload,
            )
            choice = response.choices[0]
            msg = choice.message

            # Case 1: the AI decided it's done (no tool call, just text)
            if not msg.tool_calls:
                transcript.append({"role": "assistant", "content": msg.content})
                self.engine.commit(ctx)
                return {"success": True, "transcript": transcript, "final_message": msg.content}

            # Case 2: the AI wants to call a tool
            messages.append(msg)  # keep the AI's own message in history

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    kwargs = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    kwargs = {}

                spec = self.tool_specs.get(tool_name)
                if spec is None:
                    # AI hallucinated a tool name that doesn't exist — treat as failure
                    self.engine.abort(ctx)
                    return {
                        "success": False,
                        "transcript": transcript,
                        "error": f"AI called unknown tool '{tool_name}'",
                    }

                transcript.append({"role": "tool_call", "tool": tool_name, "args": kwargs})

                try:
                    result = self.engine.execute_step(ctx, spec.tool, kwargs)
                    transcript.append({"role": "tool_result", "tool": tool_name, "result": result})

                    # Tell the AI what happened so it can decide the next step
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    })
                except Exception as e:
                    # Crash OR guardrail violation -> roll everything back, stop.
                    transcript.append({"role": "error", "tool": tool_name, "error": str(e)})
                    self.engine.abort(ctx)
                    return {
                        "success": False,
                        "transcript": transcript,
                        "error": f"Step '{tool_name}' failed: {e}. Transaction fully rolled back.",
                    }

        # Ran out of steps without the AI declaring itself done
        self.engine.abort(ctx)
        return {
            "success": False,
            "transcript": transcript,
            "error": "Max steps reached without completion. Rolled back for safety.",
        }
