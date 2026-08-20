"""Offline corner-case suite for the redesigned BaseChatAgent loop.

Every abnormal case is driven by a scripted backend (no network) so behavior
is deterministic. Each case asserts: classification, recovery/termination,
meta recording, and append-only wire order (linearity precondition).

Run:  .venv/bin/python test/agents/agent_loop/offline_suite.py
Logs: test/agents/agent_loop/results/offline_<timestamp>.log
"""

from __future__ import annotations

import asyncio
import copy
import datetime
import json
import sys
import traceback
from pathlib import Path
from typing import Any, List, Optional

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from openai import APIConnectionError, InternalServerError  # noqa: E402

from camel.agents.base_chat_agent import (  # noqa: E402
    BaseChatAgent,
    ResponseFeedback,
    TerminationReason,
)
from camel.models import BaseModelBackend, ModelProcessingError  # noqa: E402
from camel.types import ChatCompletion, ModelType  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_PATH = RESULTS_DIR / (
    "offline_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
)
_LOG = open(LOG_PATH, "w")


def log(*parts: Any) -> None:
    line = " ".join(str(p) for p in parts)
    print(line)
    _LOG.write(line + "\n")
    _LOG.flush()


# --------------------------------------------------------------------------
# Scripted backend
# --------------------------------------------------------------------------

def completion(
    *,
    content: Optional[str] = None,
    tool_calls: Optional[List[dict]] = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    meta_info: Optional[dict] = None,
) -> ChatCompletion:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    choice: dict = {
        "index": 0,
        "message": message,
        "finish_reason": finish_reason,
    }
    if meta_info is not None:
        choice["meta_info"] = meta_info
    return ChatCompletion.model_validate(
        {
            "id": "scripted",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


def tc(id_: str, name: str, arguments: str) -> dict:
    return {
        "id": id_,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class ScriptedBackend(BaseModelBackend):
    """Replays a script of completions/exceptions; records wire requests."""

    def __init__(self, script: List[Any], max_tokens: int = 0):
        config = {"max_tokens": max_tokens} if max_tokens else {}
        super().__init__(
            model_type=ModelType.GPT_4O_MINI, model_config_dict=config
        )
        self.script = list(script)
        self.requests: List[List[dict]] = []

    async def _arun(self, messages, response_format=None, tools=None):
        self.requests.append([copy.deepcopy(dict(m)) for m in messages])
        if not self.script:
            raise AssertionError("script exhausted: unexpected model call")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def _run(self, *a, **k):
        raise NotImplementedError

    @property
    def token_counter(self):
        raise NotImplementedError

    def check_model_config(self):
        pass


def shell(command: str) -> str:
    r"""Run a command.

    Args:
        command (str): The command.

    Returns:
        str: Output.
    """
    return f"ran:{command}"


def make_agent(script, **kwargs) -> tuple:
    backend = ScriptedBackend(script, max_tokens=kwargs.pop("max_tokens", 0))
    agent = BaseChatAgent(
        system_message="sys", model=backend, tools=[shell], **kwargs
    )
    return agent, backend


def assert_append_only(backend: ScriptedBackend) -> None:
    """Each wire request must verbatim-extend the previous one (linearity)."""
    for prev, cur in zip(backend.requests, backend.requests[1:]):
        assert cur[: len(prev)] == prev, (
            f"append-only violated:\nprev={prev}\ncur={cur[:len(prev)]}"
        )


def roles(agent: BaseChatAgent) -> List[str]:
    return [m["role"] for m in agent.message_list]


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

async def case_normal_tool_loop():
    agent, backend = make_agent(
        [
            completion(tool_calls=[tc("c1", "shell", '{"command": "ls"}')],
                       finish_reason="tool_calls"),
            completion(content="done"),
        ]
    )
    result = await agent.astep("task")
    assert not result.terminated
    assert agent.termination_reason is TerminationReason.TASK_COMPLETE
    assert roles(agent) == ["system", "user", "assistant", "tool", "assistant"]
    assert agent.message_list[3]["content"] == "ran:ls"
    assert_append_only(backend)


async def case_zero_arg_convention():
    # arguments "" is the zero-arg convention, executed as {}. (null cannot
    # appear on the wire: the OpenAI tool-call type requires a string.)
    for raw in ("",):
        agent, backend = make_agent(
            [
                completion(tool_calls=[tc("c1", "shell", raw)],
                           finish_reason="tool_calls"),
                completion(content="done"),
            ]
        )
        result = await agent.astep("task")
        assert agent.termination_reason is TerminationReason.TASK_COMPLETE
        # shell requires `command`; execution fails inside the tool but the
        # call itself is valid and answered — never silently dropped.
        assert roles(agent)[3] == "tool"
        assert agent.meta_info_record["invalid_tool_call_count"] == 0
        assert_append_only(backend)


async def case_invalid_args_then_recover():
    agent, backend = make_agent(
        [
            completion(tool_calls=[tc("bad", "shell", '{"command": "trunc')],
                       finish_reason="tool_calls"),
            completion(tool_calls=[tc("ok", "shell", '{"command": "ls"}')],
                       finish_reason="tool_calls"),
            completion(content="done"),
        ]
    )
    result = await agent.astep("task")
    assert agent.termination_reason is TerminationReason.TASK_COMPLETE
    msgs = agent.message_list
    assert msgs[3]["role"] == "tool" and msgs[3]["tool_call_id"] == "bad"
    assert "not a valid JSON object" in msgs[3]["content"]
    assert agent.meta_info_record["invalid_tool_call_count"] == 1
    assert agent.meta_info_record["feedback_events"][0]["kind"] == (
        "invalid_args"
    )
    assert_append_only(backend)


async def case_mixed_valid_invalid_positional():
    agent, backend = make_agent(
        [
            completion(
                tool_calls=[
                    tc("v1", "shell", '{"command": "a"}'),
                    tc("bad", "shell", "{broken"),
                    tc("v2", "shell", '{"command": "b"}'),
                ],
                finish_reason="tool_calls",
            ),
            completion(content="done"),
        ]
    )
    await agent.astep("task")
    msgs = agent.message_list
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    # All three ids answered, strictly in tool_calls positional order.
    assert [m["tool_call_id"] for m in tool_msgs] == ["v1", "bad", "v2"]
    assert tool_msgs[0]["content"] == "ran:a"
    assert "not a valid JSON object" in tool_msgs[1]["content"]
    assert tool_msgs[2]["content"] == "ran:b"
    assert agent.meta_info_record["max_tool_calls_per_turn"] == 3
    assert_append_only(backend)


async def case_consecutive_malformed_exhaustion():
    bad = completion(tool_calls=[tc("b", "shell", "{oops")],
                     finish_reason="tool_calls")
    agent, backend = make_agent(
        [bad, bad, bad], max_consecutive_malformed=2
    )
    result = await agent.astep("task")
    assert result.terminated
    assert agent.termination_reason is TerminationReason.MALFORMED_TOOL_CALL
    assert len(backend.requests) == 3  # budget 2 => 2 retries then stop
    assert agent.meta_info_record["invalid_tool_call_count"] == 3
    assert_append_only(backend)


async def case_length_feedback_then_recover():
    agent, backend = make_agent(
        [
            completion(content="cut off mid", finish_reason="length"),
            completion(content="done"),
        ]
    )
    result = await agent.astep("task")
    assert agent.termination_reason is TerminationReason.TASK_COMPLETE
    assert roles(agent) == ["system", "user", "assistant", "user", "assistant"]
    assert "cut off" in agent.message_list[3]["content"]
    assert agent.meta_info_record["feedback_events"][0]["kind"] == "truncated"
    assert_append_only(backend)


async def case_consecutive_length_exhaustion():
    trunc = completion(content="partial", finish_reason="length")
    agent, backend = make_agent(
        [trunc, trunc, trunc], max_consecutive_length=2
    )
    result = await agent.astep("task")
    assert result.terminated
    assert agent.termination_reason is TerminationReason.MAX_TOKENS_REACHED
    assert len(backend.requests) == 3
    # No trailing feedback message after the final exhausted turn.
    assert agent.message_list[-1]["role"] == "assistant"
    assert_append_only(backend)


async def case_truncated_tool_calls_never_execute():
    executed = []

    def probe(command: str) -> str:
        r"""Probe.

        Args:
            command (str): Cmd.

        Returns:
            str: Out.
        """
        executed.append(command)
        return "x"

    backend = ScriptedBackend(
        [
            completion(tool_calls=[tc("c", "probe", '{"command": "rm"}')],
                       finish_reason="length"),
            completion(content="done"),
        ]
    )
    agent = BaseChatAgent(system_message="sys", model=backend, tools=[probe])
    await agent.astep("task")
    assert executed == []  # truncation outranks tool execution
    assert agent.meta_info_record["feedback_events"][0]["kind"] == "truncated"
    assert_append_only(backend)


async def case_parser_failed_markup_marker():
    agent, backend = make_agent(
        [
            completion(content='text <tool_call>{"name": "shell"}'),
            completion(content="done"),
        ],
        tool_markup_markers=["<tool_call>"],
    )
    result = await agent.astep("task")
    assert agent.termination_reason is TerminationReason.TASK_COMPLETE
    assert roles(agent) == ["system", "user", "assistant", "user", "assistant"]
    assert agent.meta_info_record["feedback_events"][0]["kind"] == (
        "parser_failed"
    )
    assert_append_only(backend)


async def case_markers_disabled_degrades_to_complete():
    agent, backend = make_agent(
        [completion(content='text <tool_call>{"name": "shell"}')]
    )
    result = await agent.astep("task")
    assert agent.termination_reason is TerminationReason.TASK_COMPLETE
    assert len(backend.requests) == 1


async def case_handler_hint_parser_failed():
    def handler(response: ChatCompletion):
        meta = getattr(response.choices[0], "meta_info", None)
        if isinstance(meta, dict) and meta.get("parse_error"):
            return ResponseFeedback(
                reason=str(meta["parse_error"]), content="Fix the format."
            )
        return None

    agent, backend = make_agent(
        [
            completion(content="garbled",
                       meta_info={"parse_error": "bad_format"}),
            completion(content="done"),
        ],
        response_feedback_handler=handler,
    )
    await agent.astep("task")
    assert agent.termination_reason is TerminationReason.TASK_COMPLETE
    assert agent.message_list[3]["content"] == "Fix the format."
    assert agent.meta_info_record["response_feedback_reasons"] == [
        "bad_format"
    ]
    assert_append_only(backend)


async def case_proactive_token_budget():
    # total_tokens(120) + max_tokens(50) > token_limit(150): the guard runs
    # before the NEXT request, so the current turn's tool calls still
    # execute and every id is answered — then the loop stops.
    agent, backend = make_agent(
        [completion(tool_calls=[tc("c", "shell", '{"command": "ls"}')],
                    finish_reason="tool_calls",
                    prompt_tokens=100, completion_tokens=20)],
        token_limit=150,
        max_tokens=50,
    )
    result = await agent.astep("task")
    assert result.terminated
    assert (
        agent.termination_reason is TerminationReason.CONTEXT_WINDOW_OVERFLOW
    )
    assert len(backend.requests) == 1  # no doomed second request
    assert agent.message_list[-1]["role"] == "tool"  # final batch answered
    assert agent.message_list[-1]["content"] == "ran:ls"
    assert agent.meta_info_record["context_tokens"] == 120


async def case_complete_near_ceiling_is_task_complete():
    # A final answer that lands near the ceiling needs no further request,
    # so it must be TASK_COMPLETE — never mislabeled as overflow.
    agent, backend = make_agent(
        [completion(content="done", prompt_tokens=140, completion_tokens=9)],
        token_limit=150,
        max_tokens=50,
    )
    result = await agent.astep("task")
    assert not result.terminated
    assert agent.termination_reason is TerminationReason.TASK_COMPLETE
    assert len(backend.requests) == 1


async def case_truncated_at_ceiling_no_dangling_feedback():
    # Truncation at the ceiling: a retry request cannot fit, so the loop
    # terminates as overflow WITHOUT appending a feedback message no model
    # call would ever consume.
    agent, backend = make_agent(
        [completion(content="partial", finish_reason="length",
                    prompt_tokens=120, completion_tokens=20)],
        token_limit=150,
        max_tokens=50,
        max_consecutive_length=5,
    )
    result = await agent.astep("task")
    assert result.terminated
    assert (
        agent.termination_reason is TerminationReason.CONTEXT_WINDOW_OVERFLOW
    )
    assert agent.message_list[-1]["role"] == "assistant"  # no dangling user


async def case_overflow_400_graceful():
    class Overflow400(Exception):
        status_code = 400

    error = Overflow400(
        "The input (70000 tokens) is longer than the model's context length "
        "(65536 tokens)."
    )
    agent, backend = make_agent([error])
    result = await agent.astep("task")  # graceful, no raise
    assert result.terminated
    assert (
        agent.termination_reason is TerminationReason.CONTEXT_WINDOW_OVERFLOW
    )
    assert result.msgs == []
    assert agent.meta_info_record["iteration_count"] == 0


async def case_max_iteration_counts_feedback_turns():
    trunc = completion(content="partial", finish_reason="length")
    agent, backend = make_agent(
        [trunc, trunc],
        max_iteration=2,
        max_consecutive_length=10,
    )
    result = await agent.astep("task")
    assert result.terminated
    assert (
        agent.termination_reason
        is TerminationReason.MAX_TOOL_ITERATIONS_REACHED
    )
    assert len(backend.requests) == 2  # feedback turns consume the budget
    assert agent.message_list[-1]["role"] == "assistant"  # no dangling user


async def case_max_iteration_answers_final_tools():
    call = completion(tool_calls=[tc("c", "shell", '{"command": "ls"}')],
                      finish_reason="tool_calls")
    agent, backend = make_agent([call], max_iteration=1)
    result = await agent.astep("task")
    assert result.terminated
    assert (
        agent.termination_reason
        is TerminationReason.MAX_TOOL_ITERATIONS_REACHED
    )
    assert agent.message_list[-1]["role"] == "tool"  # id answered before stop


async def case_infra_retry_then_success():
    agent, backend = make_agent(
        [
            APIConnectionError(request=None),
            completion(content="done"),
        ],
        retry_attempts=2,
        retry_delay=0.0,
    )
    result = await agent.astep("task")
    assert agent.termination_reason is TerminationReason.TASK_COMPLETE
    assert len(backend.requests) == 2


async def case_infra_retry_exhaustion_raises():
    agent, backend = make_agent(
        [APIConnectionError(request=None), APIConnectionError(request=None)],
        retry_attempts=2,
        retry_delay=0.0,
    )
    raised = None
    try:
        await agent.astep("task")
    except ModelProcessingError as error:
        raised = error
    assert raised is not None  # infra failure is a trial-level error
    assert agent.termination_reason is TerminationReason.CONNECTION_ERROR


async def case_counters_reset_on_valid_turn():
    bad = completion(tool_calls=[tc("b", "shell", "{oops")],
                     finish_reason="tool_calls")
    good = completion(tool_calls=[tc("g", "shell", '{"command": "ls"}')],
                      finish_reason="tool_calls")
    # bad, bad, good (reset), bad, bad, good (reset), done — budget 2 never
    # exhausts because valid turns break the streaks.
    agent, backend = make_agent(
        [bad, bad, good, bad, bad, good, completion(content="done")],
        max_consecutive_malformed=2,
    )
    result = await agent.astep("task")
    assert agent.termination_reason is TerminationReason.TASK_COMPLETE
    assert agent.meta_info_record["invalid_tool_call_count"] == 4
    assert_append_only(backend)


async def case_meta_recording_complete():
    agent, backend = make_agent(
        [
            completion(content="partial", finish_reason="length",
                       prompt_tokens=10, completion_tokens=5),
            completion(tool_calls=[tc("c", "shell", '{"command": "ls"}')],
                       finish_reason="tool_calls",
                       prompt_tokens=20, completion_tokens=7),
            completion(content="done", prompt_tokens=30, completion_tokens=3),
        ]
    )
    result = await agent.astep("task")
    meta = agent.meta_info_record
    assert meta["iteration_count"] == 3
    assert meta["prompt_tokens"] == 60
    assert meta["completion_tokens"] == 15
    assert meta["total_tokens"] == 75
    assert meta["context_tokens"] == 33  # last turn total
    assert meta["total_tool_calls"] == 1
    assert len(meta["feedback_events"]) == 1
    assert meta["termination_reason"] is TerminationReason.TASK_COMPLETE
    assert result.info["training"]["iteration_count"] == 3


CASES = [
    case_normal_tool_loop,
    case_zero_arg_convention,
    case_invalid_args_then_recover,
    case_mixed_valid_invalid_positional,
    case_consecutive_malformed_exhaustion,
    case_length_feedback_then_recover,
    case_consecutive_length_exhaustion,
    case_truncated_tool_calls_never_execute,
    case_parser_failed_markup_marker,
    case_markers_disabled_degrades_to_complete,
    case_handler_hint_parser_failed,
    case_proactive_token_budget,
    case_complete_near_ceiling_is_task_complete,
    case_truncated_at_ceiling_no_dangling_feedback,
    case_overflow_400_graceful,
    case_max_iteration_counts_feedback_turns,
    case_max_iteration_answers_final_tools,
    case_infra_retry_then_success,
    case_infra_retry_exhaustion_raises,
    case_counters_reset_on_valid_turn,
    case_meta_recording_complete,
]


async def main() -> int:
    log(f"offline suite — {datetime.datetime.now().isoformat()}")
    failed = 0
    for case in CASES:
        try:
            await case()
            log(f"PASS  {case.__name__}")
        except Exception:
            failed += 1
            log(f"FAIL  {case.__name__}\n{traceback.format_exc()}")
    log(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
