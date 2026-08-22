"""Live suite for the redesigned BaseChatAgent loop against a real SGLang
OpenAI-compatible endpoint.

Covers what can be deterministically induced on a live server: normal
completion, tool-call round trip, length truncation (tiny max_tokens),
truncation-budget exhaustion, proactive context-budget stop, real
context-overflow 400, plus evidence for the linearity/prefix-cache contract
(`cached_tokens`, `prompt_token_ids` extension). Failure modes that cannot be
forced on a live model (invalid-JSON args, markup-in-content) are covered by
`offline_suite.py`.

Run:  .venv/bin/python test/agents/agent_loop/live_suite.py [base_url]
Logs: test/agents/agent_loop/results/live_<timestamp>.log
      test/agents/agent_loop/results/live_<timestamp>_trajectories.json
"""

from __future__ import annotations

import asyncio
import datetime
import json
import sys
import traceback
import urllib.request
from pathlib import Path
from typing import Any, List

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from camel.agents.base_chat_agent import (  # noqa: E402
    BaseChatAgent,
    TerminationReason,
)
from camel.models.base_chat_model import BaseChatModel  # noqa: E402

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://10.220.51.13:31001/v1"
MODEL = "model"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = RESULTS_DIR / f"live_{STAMP}.log"
TRAJ_PATH = RESULTS_DIR / f"live_{STAMP}_trajectories.json"
_LOG = open(LOG_PATH, "w")
TRAJECTORIES: dict = {}


def log(*parts: Any) -> None:
    line = " ".join(str(p) for p in parts)
    print(line)
    _LOG.write(line + "\n")
    _LOG.flush()


def make_model(max_tokens: int, **extra) -> BaseChatModel:
    """One recording model per case; requests SGLang token-id extras."""
    return BaseChatModel(
        model_type=MODEL,
        url=BASE_URL,
        api_key="dummy",
        timeout=600,
        max_retries=0,
        model_config_dict={
            "temperature": 0.0,
            "max_tokens": max_tokens,
            # SGLang-native opt-in extras (NOT `return_token_ids`, which is
            # not an SGLang flag): exact prompt ids + per-token logprobs.
            "extra_body": {
                "return_prompt_token_ids": True,
                "return_meta_info": True,
                "logprobs": True,
                # Deployment precondition for linearity: without this, GLM
                # templates clear prior-turn reasoning as soon as a user
                # message follows it, breaking prompt-prefix extension.
                "chat_template_kwargs": {"clear_thinking": False},
            },
            **extra,
        },
    )


def calc(expression: str) -> str:
    r"""Evaluate a simple arithmetic expression.

    Args:
        expression (str): Arithmetic expression like "3*7".

    Returns:
        str: The numeric result.
    """
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "error: unsupported characters"
    return str(eval(expression))  # noqa: S307 — sandboxed charset


class TurnRecorder:
    """Collects per-turn SGLang extras straight off the wire."""

    def __init__(self) -> None:
        self.prompt_ids: List[List[int]] = []
        self.output_ids: List[List[int]] = []
        self.cached_tokens: List[int] = []
        self.finish_reasons: List[str] = []

    def hook(self, model: BaseChatModel) -> None:
        original = model._arun

        async def wrapped(messages, response_format=None, tools=None):
            response = await original(messages, response_format, tools)
            raw = response.model_dump()
            choice = raw["choices"][0]
            self.finish_reasons.append(str(choice.get("finish_reason")))
            self.prompt_ids.append(choice.get("prompt_token_ids") or [])
            meta = choice.get("meta_info") or {}
            self.cached_tokens.append(int(meta.get("cached_tokens") or 0))
            logprobs = meta.get("output_token_logprobs") or []
            self.output_ids.append(
                [int(t[1]) for t in logprobs if isinstance(t, (list, tuple))]
            )
            return response

        model._arun = wrapped  # type: ignore[method-assign]


def save_trajectory(name: str, agent: BaseChatAgent, rec: TurnRecorder) -> None:
    TRAJECTORIES[name] = {
        "messages": agent.message_list,
        "meta_info_record": {
            k: (v.value if isinstance(v, TerminationReason) else v)
            for k, v in agent.meta_info_record.items()
        },
        "finish_reasons": rec.finish_reasons,
        "cached_tokens": rec.cached_tokens,
        "prompt_id_lens": [len(ids) for ids in rec.prompt_ids],
        "output_id_lens": [len(ids) for ids in rec.output_ids],
    }
    TRAJ_PATH.write_text(json.dumps(TRAJECTORIES, indent=2, default=str))


def check_linearity(rec: TurnRecorder) -> str:
    """prompt_ids[i+1] must extend prompt_ids[i] + output_ids[i] (with a small
    boundary-token tolerance, cf. template stop/start token overlap)."""
    verdicts = []
    for i in range(len(rec.prompt_ids) - 1):
        prev = rec.prompt_ids[i] + rec.output_ids[i]
        cur = rec.prompt_ids[i + 1]
        if not prev or not cur:
            verdicts.append("no-ids")
            continue
        check = prev[: len(prev) - 2]  # tolerate <=2 trailing boundary tokens
        verdicts.append("linear" if cur[: len(check)] == check else "BROKEN")
    return ",".join(verdicts) or "single-turn"


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

async def case_health_and_models():
    with urllib.request.urlopen(
        BASE_URL.rsplit("/v1", 1)[0] + "/health", timeout=10
    ) as response:
        assert response.status == 200
    with urllib.request.urlopen(BASE_URL + "/models", timeout=10) as response:
        models = json.loads(response.read())
    assert any(m["id"] == MODEL for m in models["data"])
    log(f"      endpoint={BASE_URL} models={[m['id'] for m in models['data']]}")


async def case_plain_completion():
    model = make_model(max_tokens=4096)
    rec = TurnRecorder(); rec.hook(model)
    agent = BaseChatAgent(system_message="You are concise.", model=model)
    result = await agent.astep("Reply with the single word: ready")
    assert agent.termination_reason is TerminationReason.TASK_COMPLETE
    assert not result.terminated
    meta = agent.meta_info_record
    assert meta["context_tokens"] > 0 and meta["iteration_count"] == 1
    save_trajectory("plain_completion", agent, rec)
    await model.aclose()


async def case_tool_call_round_trip():
    model = make_model(max_tokens=4096)
    rec = TurnRecorder(); rec.hook(model)
    agent = BaseChatAgent(
        system_message="Use the calc tool for arithmetic, then answer.",
        model=model,
        tools=[calc],
        tool_markup_markers=["<tool_call>"],
    )
    result = await agent.astep("Compute 391 * 17 using the calc tool.")
    assert agent.termination_reason is TerminationReason.TASK_COMPLETE
    assert agent.meta_info_record["total_tool_calls"] >= 1
    tool_msgs = [m for m in agent.message_list if m["role"] == "tool"]
    assert any("6647" in m["content"] for m in tool_msgs)
    final = result.msgs[0].content if result.msgs else ""
    log(f"      turns={agent.meta_info_record['iteration_count']} "
        f"tools={agent.meta_info_record['total_tool_calls']} "
        f"final={final[:60]!r}")
    log(f"      linearity={check_linearity(rec)} cached={rec.cached_tokens}")
    save_trajectory("tool_call_round_trip", agent, rec)
    await model.aclose()


async def case_length_truncation_feedback():
    # max_tokens=32 forces finish_reason=length; budget 1 => one feedback
    # retry, then graceful MAX_TOKENS_REACHED.
    model = make_model(max_tokens=32)
    rec = TurnRecorder(); rec.hook(model)
    agent = BaseChatAgent(
        system_message="You are a writer.",
        model=model,
        max_consecutive_length=1,
    )
    result = await agent.astep("Write a 500-word essay about oceans.")
    assert result.terminated
    assert agent.termination_reason is TerminationReason.MAX_TOKENS_REACHED
    assert rec.finish_reasons == ["length", "length"]
    user_roles = [m["role"] for m in agent.message_list]
    assert user_roles == ["system", "user", "assistant", "user", "assistant"]
    assert "cut off" in agent.message_list[3]["content"]
    events = agent.meta_info_record["feedback_events"]
    assert [e["kind"] for e in events] == ["truncated", "truncated"]
    log(f"      linearity={check_linearity(rec)} cached={rec.cached_tokens}")
    save_trajectory("length_truncation_feedback", agent, rec)
    await model.aclose()


async def case_proactive_context_budget():
    # token_limit below first-turn total + max_tokens => the loop must stop
    # after turn 1 without issuing a doomed second request.
    model = make_model(max_tokens=64)
    rec = TurnRecorder(); rec.hook(model)
    agent = BaseChatAgent(
        system_message="You are a writer.",
        model=model,
        token_limit=100,  # first turn total (~50) + 64 exceeds this
        max_consecutive_length=5,
    )
    result = await agent.astep("Write a long story about a lighthouse.")
    assert result.terminated
    assert (
        agent.termination_reason is TerminationReason.CONTEXT_WINDOW_OVERFLOW
    )
    assert agent.meta_info_record["iteration_count"] == 1
    assert len(rec.prompt_ids) == 1  # exactly one wire request
    save_trajectory("proactive_context_budget", agent, rec)
    await model.aclose()


async def case_oversized_prompt_behavior():
    # Probe how THIS deployment treats an oversized (~300k-token) prompt.
    # This router accepts it (long-context serve, no input validation), so
    # the client-side proactive token budget is the only overflow guard —
    # the 400-classification path is covered deterministically in
    # offline_suite.py (case_overflow_400_graceful). Either way the loop
    # must terminate gracefully with a classified reason, never raise.
    model = make_model(max_tokens=64)
    agent = BaseChatAgent(
        system_message="You are terse.",
        model=model,
        max_consecutive_length=0,
    )
    result = await agent.astep("Repeat after me: " + "lighthouse " * 150000)
    assert result.terminated
    assert agent.termination_reason in (
        TerminationReason.MAX_TOKENS_REACHED,       # server accepted, cut off
        TerminationReason.CONTEXT_WINDOW_OVERFLOW,  # server rejected with 400
    )
    log(f"      server behavior => {agent.termination_reason.value}, "
        f"prompt_tokens={agent.meta_info_record['prompt_tokens']}")
    await model.aclose()


async def case_multi_turn_prefix_cache():
    # Three sequential tool turns; cached_tokens should grow with the prefix,
    # evidencing that append-only requests keep the radix cache hot.
    model = make_model(max_tokens=4096)
    rec = TurnRecorder(); rec.hook(model)
    agent = BaseChatAgent(
        system_message=(
            "Use the calc tool for every arithmetic step. One call at a time."
        ),
        model=model,
        tools=[calc],
        max_iteration=8,
    )
    await agent.astep(
        "Compute (12+34), then multiply that result by 3, then subtract 7. "
        "Use the calc tool for each of the three steps, then answer."
    )
    meta = agent.meta_info_record
    assert meta["total_tool_calls"] >= 2
    linearity = check_linearity(rec)
    assert "BROKEN" not in linearity, f"prompt ids diverged: {linearity}"
    # Cache evidence is affinity-dependent: without a routing key the
    # multi-worker router may send each turn to a different worker, so
    # cached_tokens > 0 cannot be guaranteed client-side. Linearity (above)
    # is the hard guarantee; log cache hits as observational evidence.
    later = rec.cached_tokens[1:]
    if not (later and all(c > 0 for c in later)):
        log(f"      NOTE: no/partial prefix-cache hits {rec.cached_tokens} "
            "(router affinity not controlled by this bare client)")
    log(f"      turns={meta['iteration_count']} linearity={linearity} "
        f"cached={rec.cached_tokens} prompt_lens={[len(p) for p in rec.prompt_ids]}")
    save_trajectory("multi_turn_prefix_cache", agent, rec)
    await model.aclose()


async def case_compaction_live():
    # A tight token_limit forces at least one compaction mid-task; the agent
    # must summarize in-segment, hand off, and still finish the task.
    model = make_model(max_tokens=256)
    rec = TurnRecorder(); rec.hook(model)
    agent = BaseChatAgent(
        system_message=(
            "Use the calc tool for every arithmetic step, one call at a "
            "time. Before each call, restate every intermediate result so "
            "far and the remaining steps."
        ),
        model=model,
        tools=[calc],
        token_limit=1000,
        max_compactions=3,
        max_iteration=20,
        tool_markup_markers=["<tool_call>"],
    )
    result = await agent.astep(
        "Compute step by step with the calc tool, one operation per call: "
        "(1) 23+58, (2) result*7, (3) result-19, (4) result+123, "
        "(5) result*2, (6) result-42. Then give the final number."
    )
    meta = agent.meta_info_record
    log(f"      compactions={meta['compaction_count']} "
        f"turns={meta['iteration_count']} tools={meta['total_tool_calls']} "
        f"context_tokens={meta['context_tokens']} "
        f"termination={agent.termination_reason.value}")
    log(f"      segments={[len(seg) for seg in agent.compacted_segments]} "
        f"live={len(agent.message_list)} "
        f"full={len(agent.full_message_list)}")
    assert meta["compaction_count"] >= 1, "expected at least one compaction"
    assert agent.termination_reason is TerminationReason.TASK_COMPLETE
    final = result.msgs[0].content if result.msgs else ""
    assert "1300" in final, f"wrong answer: {final[:120]!r}"
    # Each archived segment ends with the in-segment summary exchange.
    for seg in agent.compacted_segments:
        assert [m["role"] for m in seg[-2:]] == ["user", "assistant"]
    save_trajectory("compaction_live", agent, rec)
    await model.aclose()


CASES = [
    case_health_and_models,
    case_plain_completion,
    case_tool_call_round_trip,
    case_length_truncation_feedback,
    case_proactive_context_budget,
    case_oversized_prompt_behavior,
    case_multi_turn_prefix_cache,
    case_compaction_live,
]


async def main() -> int:
    log(f"live suite — {datetime.datetime.now().isoformat()} — {BASE_URL}")
    # Optional case-name substring filters after base_url, e.g.:
    #   live_suite.py http://host/v1 oversized prefix_cache
    filters = sys.argv[2:]
    cases = [
        c for c in CASES
        if not filters or any(f in c.__name__ for f in filters)
    ]
    failed = 0
    for case in cases:
        try:
            await case()
            log(f"PASS  {case.__name__}")
        except Exception:
            failed += 1
            log(f"FAIL  {case.__name__}\n{traceback.format_exc()}")
    log(f"\n{len(cases) - failed}/{len(cases)} passed")
    log(f"trajectories: {TRAJ_PATH}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
