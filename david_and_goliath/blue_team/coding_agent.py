"""Blue Team coding agent with tool-use capability.

Architecture
============
CodingAgent wraps an OpenAI-compatible LLM and exposes four tools:

  static_analyzer   — Bandit security scan on generated code
  code_executor     — Run code in a subprocess and capture output
  unit_test_runner  — Run task test cases against a solution
  memory_retrieval  — Keyword-based injection pattern detection

The agent follows a two-level loop:

  Outer loop (reflexion_idx, 0..max_reflexion):
    One complete "attempt" per iteration.  When an attempt finishes but
    tests still fail, ReflexionEngine appends a reflection message and
    the outer loop starts a fresh inner loop.  max_reflexion is therefore
    strictly respected — it cannot be eaten by tool-call turns.

  Inner loop (tool_turn, 0..max_turns):
    One LLM call per iteration.  Continues as long as the LLM emits
    tool_calls.  Breaks when the LLM stops calling tools (final answer).

Fix log vs. previous version:
  Fix-1  Separate turn counters: outer=reflexion, inner=tool calls.
         max_turns and max_reflexion are now independently enforced.
  Fix-2  _dispatch() validates required args explicitly; missing/empty
         required args return an error dict instead of silently passing
         an empty string to the tool.
  Fix-3  generated_code is updated ONLY when msg.tool_calls is empty
         (i.e., the LLM is in answer mode, not mid-reasoning).

Return contract
===============
  process(injected_task) -> BlueTeamResponse
    .generated_code      — final Python code (may be adversarially influenced)
    .tool_calls          — structured log of every tool invocation
    .reasoning           — concatenated assistant text turns
    .latency_ms          — wall time
    .detected_suspicious — True if memory_retrieval found injection patterns
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from core.types import BlueTeamResponse, InjectedTask, ToolCall
from blue_team.prompt_builder import BluePromptBuilder
from blue_team.reflexion import ReflexionEngine
from blue_team.tools.code_executor import CodeExecutorTool
from blue_team.tools.memory_retrieval import MemoryRetrievalTool
from blue_team.tools.static_analyzer import StaticAnalyzerTool
from blue_team.tools.unit_test_runner import UnitTestRunnerTool

logger = logging.getLogger(__name__)


class CodingAgent:
    """Security-aware coding agent for the Blue Team.

    Args:
        model:          OpenAI model name (default: gpt-4o).
        api_key:        OpenAI API key; falls back to OPENAI_API_KEY env var.
        base_url:       Optional custom base URL (for local/proxy endpoints).
        temperature:    Sampling temperature; low by default for determinism.
        max_turns:      Maximum tool-call iterations per attempt (inner loop).
        max_reflexion:  Maximum Reflexion retries after test failure (outer loop).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_turns: int = 6,
        max_reflexion: int = 2,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_turns = max_turns
        self.max_reflexion = max_reflexion

        # ---- Tools ----
        self._static_analyzer = StaticAnalyzerTool()
        self._code_executor = CodeExecutorTool()
        self._unit_test_runner = UnitTestRunnerTool()
        self._memory_retrieval = MemoryRetrievalTool()

        self._tool_schemas: list[dict[str, Any]] = [
            self._static_analyzer.schema,
            self._code_executor.schema,
            self._unit_test_runner.schema,
            self._memory_retrieval.schema,
        ]

        self._prompt_builder = BluePromptBuilder()
        self._reflexion = ReflexionEngine()
        self._client = None  # lazy init

    # ==================================================================
    # Public interface (called by OracleRewardWorker)
    # ==================================================================

    def process(self, injected_task: InjectedTask) -> BlueTeamResponse:
        """Run the full agentic loop for *injected_task*.

        Returns a BlueTeamResponse regardless of whether the task was
        completed correctly or the agent was adversarially manipulated.
        """
        t0 = time.perf_counter()
        tool_calls_log: list[ToolCall] = []
        reasoning_parts: list[str] = []
        detected_suspicious = False
        generated_code = ""

        messages: list[dict[str, Any]] = self._prompt_builder.build(injected_task)

        # Side-channel state shared across attempts (updated by tool dispatch)
        last_static_issues: list[dict] = []
        last_exec_result: dict[str, Any] = {}

        # ==============================================================
        # FIX-1: Two independent loops.
        #   Outer  (reflexion_idx): one full attempt per iteration.
        #   Inner  (tool_turn):     one LLM call per iteration.
        #   max_turns and max_reflexion are now fully independent.
        # ==============================================================
        for reflexion_idx in range(self.max_reflexion + 1):

            llm_finished = False  # True when LLM emits no tool_calls

            # ---- Inner: tool-call loop for one attempt ----
            for tool_turn in range(self.max_turns):
                try:
                    client = self._get_client()
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=self._tool_schemas,
                        tool_choice="auto",
                        temperature=self.temperature,
                        max_tokens=2048,
                    )
                except Exception as exc:
                    logger.warning(
                        "LLM call failed (reflexion=%d tool_turn=%d): %s",
                        reflexion_idx, tool_turn, exc,
                    )
                    break  # abort this attempt

                msg = response.choices[0].message

                # ---- Serialize assistant message for history ----
                assistant_dict: dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    assistant_dict["content"] = msg.content
                    reasoning_parts.append(msg.content)
                    # --------------------------------------------------
                    # FIX-3: Extract generated_code ONLY when the LLM is
                    # in "answer mode" (no tool_calls on this turn).
                    # Intermediate reasoning turns may contain draft code
                    # blocks that should not overwrite the final answer.
                    # --------------------------------------------------
                    if not msg.tool_calls:
                        code = self._extract_code(msg.content)
                        if code:
                            generated_code = code

                if msg.tool_calls:
                    assistant_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                messages.append(assistant_dict)

                # ---- LLM done with this attempt ----
                if not msg.tool_calls:
                    llm_finished = True
                    break

                # ---- Execute tool calls ----
                for tc in msg.tool_calls:
                    tc_t0 = time.perf_counter()
                    tool_name = tc.function.name
                    try:
                        args: dict[str, Any] = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    result, success = self._dispatch(tool_name, args, injected_task)
                    result_str = json.dumps(result)

                    # ---- Side-effects ----
                    if tool_name == "memory_retrieval" and result.get("suspicious"):
                        detected_suspicious = True
                        logger.debug(
                            "Blue team detected suspicious content: %s",
                            result.get("summary", ""),
                        )
                    if tool_name == "static_analyzer":
                        last_static_issues = result.get("issues", [])
                    if tool_name == "code_executor":
                        last_exec_result = result

                    tool_calls_log.append(
                        ToolCall(
                            tool_name=tool_name,
                            input_args=args,
                            output=result_str[:600],
                            success=success,
                            latency_ms=round((time.perf_counter() - tc_t0) * 1000, 1),
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_str[:3000],
                        }
                    )

            # ---- Post-attempt: decide whether to reflexion ----

            if not llm_finished:
                # Hit max_turns without the LLM stopping → abort entirely
                break

            # No code or no test cases → nothing to verify, done
            if not generated_code or not injected_task.original_task.test_cases:
                break

            # Run tests to decide whether reflexion is warranted
            tr = self._unit_test_runner.run(
                solution_code=generated_code,
                test_cases=injected_task.original_task.test_cases,
            )

            # Tests pass OR reflexion budget exhausted → done
            if tr["failed"] == 0 or reflexion_idx >= self.max_reflexion:
                break

            # ---- Trigger reflexion: append reflection message ----
            logger.debug(
                "Reflexion triggered (attempt %d/%d): %d/%d tests failed",
                reflexion_idx + 1, self.max_reflexion,
                tr["failed"], tr["total"],
            )
            reflection_msg = self._reflexion.build_reflection_message(
                solution_code=generated_code,
                test_results=tr,
                static_issues=last_static_issues,
                executor_result=last_exec_result or None,
            )
            messages.append(reflection_msg)
            # outer loop continues → fresh inner loop with updated messages

        # ---- Fallback: scan all reasoning if no clean code block found ----
        if not generated_code:
            full_text = "\n".join(reasoning_parts)
            generated_code = self._extract_code(full_text) or "# No code generated"

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Blue team done: tool_calls=%d suspicious=%s latency=%.0fms",
            len(tool_calls_log), detected_suspicious, latency_ms,
        )

        return BlueTeamResponse(
            generated_code=generated_code,
            tool_calls=tool_calls_log,
            reasoning="\n\n".join(reasoning_parts),
            latency_ms=round(latency_ms, 1),
            detected_suspicious=detected_suspicious,
        )

    # ==================================================================
    # Tool dispatch
    # ==================================================================

    def _dispatch(
        self,
        tool_name: str,
        args: dict[str, Any],
        injected_task: InjectedTask,
    ) -> tuple[dict[str, Any], bool]:
        """Call the right tool.

        FIX-2: Each required argument is validated explicitly before being
        passed to the tool.  A missing or empty required arg returns an
        error dict (success=False) instead of silently passing "" to the
        tool and producing a misleading "no issues found" result.
        """
        try:
            if tool_name == "static_analyzer":
                code = args.get("code")
                if not code:
                    return {"error": "static_analyzer: required argument 'code' is missing or empty."}, False
                result = self._static_analyzer.run(code=code)

            elif tool_name == "code_executor":
                code = args.get("code")
                if not code:
                    return {"error": "code_executor: required argument 'code' is missing or empty."}, False
                result = self._code_executor.run(
                    code=code,
                    timeout=args.get("timeout"),
                )

            elif tool_name == "unit_test_runner":
                solution_code = args.get("solution_code")
                if not solution_code:
                    return {"error": "unit_test_runner: required argument 'solution_code' is missing or empty."}, False
                # LLM may omit test_cases — fall back to the task's own tests
                test_cases = args.get("test_cases") or injected_task.original_task.test_cases
                result = self._unit_test_runner.run(
                    solution_code=solution_code,
                    test_cases=test_cases,
                )

            elif tool_name == "memory_retrieval":
                query_text = args.get("query_text")
                if not query_text:
                    return {"error": "memory_retrieval: required argument 'query_text' is missing or empty."}, False
                result = self._memory_retrieval.query(query_text=query_text)

            else:
                return {"error": f"Unknown tool: {tool_name!r}"}, False

            return result, True

        except Exception as exc:
            logger.warning("Tool %r raised: %s", tool_name, exc)
            return {"error": str(exc)}, False

    # ==================================================================
    # Helpers
    # ==================================================================

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for the Blue Team agent. "
                "Install with: pip install openai"
            ) from exc
        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    @staticmethod
    def _extract_code(text: str) -> str:
        """Pull Python code from a markdown code fence, or return ''."""
        m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return ""
