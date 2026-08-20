# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
import asyncio
import json
import time
from typing import Any, ClassVar, Dict, List, Optional, Type, cast

import pytest
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_function_tool_call import (
    Function,
)
from pydantic import BaseModel

from camel.agents.base import BaseAgent
from camel.agents.base_chat_agent import (
    BaseChatAgent,
    ResponseFeedback,
    TerminationReason,
)
from camel.agents.chat_agent import ChatAgent
from camel.memories import AgentMemory
from camel.messages import OpenAIMessage
from camel.models import ModelProcessingError
from camel.models.stub_model import StubModel
from camel.types import (
    ChatCompletion,
    ModelType,
)


class SequencedModel(StubModel):
    def __init__(self, responses: List[ChatCompletion]):
        super().__init__(ModelType.STUB)
        self.responses = responses
        self.received: List[List[OpenAIMessage]] = []

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        self.received.append(
            cast(List[OpenAIMessage], [dict(message) for message in messages])
        )
        return self.responses.pop(0)


def completion(
    content=None, tool_calls=None, *, finish_reason=None, meta_info=None
):
    payload = {
        "id": "test-id",
        "model": "test-model",
        "object": "chat.completion",
        "created": int(time.time()),
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason
                or ("tool_calls" if tool_calls else "stop"),
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
                **({"meta_info": meta_info} if meta_info else {}),
            }
        ],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6,
        },
    }
    return ChatCompletion.model_validate(payload)


@pytest.mark.asyncio
async def test_append_only_agent_executes_tool_with_unprocessed_memory():
    def add(a: int, b: int) -> int:
        return a + b

    tool_call = ChatCompletionMessageFunctionToolCall(
        id="call-1",
        type="function",
        function=Function(name="add", arguments='{"a": 2, "b": 3}'),
    )
    backend = SequencedModel(
        [completion(tool_calls=[tool_call]), completion(content="five")]
    )
    agent = BaseChatAgent(
        system_message="Be concise.",
        model=backend,
        tools=[add],
        token_limit=100,
        step_timeout=None,
    )

    response = await agent.astep("calculate")

    assert isinstance(agent, BaseAgent)
    assert not isinstance(agent, ChatAgent)
    assert isinstance(agent.memory, AgentMemory)
    assert response.msgs[0].content == "five"
    assert [message["role"] for message in agent.message_list] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert agent.message_list[3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "5",
    }
    assert [message["role"] for message in backend.received[1]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    memory_messages, _ = agent.memory.get_context()
    assert memory_messages == agent.message_list
    assert agent.termination_reason is TerminationReason.TASK_COMPLETE
    assert response.info["training"]["total_tool_calls"] == 1


@pytest.mark.asyncio
async def test_user_turns_append_and_reset_clears_trajectory():
    backend = SequencedModel(
        [completion(content="one"), completion(content="two")]
    )
    agent = BaseChatAgent(model=backend, token_limit=100, step_timeout=None)

    await agent.astep("first")
    await agent.astep("second")
    assert [message["role"] for message in agent.message_list] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]

    agent.reset()
    assert agent.message_list == []
    assert agent.memory.retrieve() == []
    assert agent.termination_reason is TerminationReason.NOT_TERMINATED


def test_error_reason_walks_wrapped_causes():
    try:
        try:
            raise asyncio.TimeoutError("server stalled")
        except asyncio.TimeoutError as cause:
            raise RuntimeError("model failed") from cause
    except RuntimeError as error:
        assert TerminationReason.from_error(error) is TerminationReason.TIMEOUT

    class ModelConnectionError(Exception):
        pass

    assert (
        TerminationReason.from_error(ModelConnectionError())
        is TerminationReason.CONNECTION_ERROR
    )


def test_error_reason_detects_openai_context_overflow_payload():
    class OpenAIStyleBadRequestError(Exception):
        code = "context_length_exceeded"
        type = "invalid_request_error"
        body: ClassVar[Dict[str, str]] = {
            "message": "This model's maximum context length is 32768 tokens.",
            "code": "context_length_exceeded",
        }

    assert (
        TerminationReason.from_error(OpenAIStyleBadRequestError())
        is TerminationReason.CONTEXT_WINDOW_OVERFLOW
    )


def test_error_reason_detects_http_response_context_overflow():
    class Response:
        status_code = 400
        text = "The input is longer than the model's context length."

        def json(self):
            return {
                "error": {
                    "message": (
                        "Requested token count exceeds the model's maximum "
                        "context length"
                    )
                }
            }

    class HTTPStatusError(Exception):
        response = Response()

    try:
        raise RuntimeError("model request failed") from HTTPStatusError()
    except RuntimeError as error:
        assert (
            TerminationReason.from_error(error)
            is TerminationReason.CONTEXT_WINDOW_OVERFLOW
        )


def test_error_reason_does_not_treat_all_bad_requests_as_context_overflow():
    class Response:
        status_code = 400
        text = "The input JSON is invalid."

        def json(self):
            return {
                "error": {
                    "message": "The input JSON is invalid.",
                    "type": "BadRequestError",
                    "code": 400,
                }
            }

    class HTTPStatusError(Exception):
        response = Response()

    assert (
        TerminationReason.from_error(HTTPStatusError())
        is TerminationReason.UNCLASSIFIED_ERROR
    )


@pytest.mark.asyncio
async def test_multiple_tool_calls_execute_sequentially_without_a_cap():
    active = 0
    max_active = 0
    events = []

    async def echo(value: int) -> int:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        events.append(("start", value))
        await asyncio.sleep(0)
        events.append(("end", value))
        active -= 1
        return value

    calls = [
        ChatCompletionMessageFunctionToolCall(
            id=f"call-{i}",
            type="function",
            function=Function(name="echo", arguments=f'{{"value": {i}}}'),
        )
        for i in range(2)
    ]
    backend = SequencedModel(
        [completion(tool_calls=calls), completion(content="done")]
    )
    agent = BaseChatAgent(
        model=backend,
        tools=[echo],
        token_limit=100,
        step_timeout=None,
    )

    result = await agent.astep("call tools")

    assert [record.result for record in result.info["tool_calls"]] == [0, 1]
    assert max_active == 1
    assert events == [("start", 0), ("end", 0), ("start", 1), ("end", 1)]
    assert not hasattr(agent, "parallel_internal_tools")
    assert agent.meta_info_record["max_tool_calls_per_turn"] == 2
    assert agent.meta_info_record["total_tool_calls"] == 2
    assert agent.termination_reason is TerminationReason.TASK_COMPLETE


@pytest.mark.asyncio
async def test_length_finish_reason_tracks_output_token_limit():
    response = completion(content="truncated", finish_reason="length")
    agent = BaseChatAgent(
        model=SequencedModel([response]),
        token_limit=100,
        max_consecutive_length=0,
        step_timeout=None,
    )

    result = await agent.astep("write")

    assert result.terminated
    assert agent.termination_reason is TerminationReason.MAX_TOKENS_REACHED
    assert agent.message_list[-1] == {
        "role": "assistant",
        "content": "truncated",
    }


@pytest.mark.asyncio
async def test_feedback_preserves_assistant_then_appends_user_correction():
    def feedback(response: ChatCompletion):
        meta = getattr(response.choices[0], "meta_info", None)
        if isinstance(meta, dict) and meta.get("parse_error"):
            return ResponseFeedback(
                reason=str(meta["parse_error"]),
                content="The response was malformed; try again.",
            )
        return None

    first = completion(
        content="partial response",
        meta_info={"parse_error": "malformed_tool_json"},
    )
    backend = SequencedModel([first, completion(content="done")])
    agent = BaseChatAgent(
        model=backend,
        token_limit=100,
        response_feedback_handler=feedback,
        max_response_feedback=1,
        step_timeout=None,
    )

    result = await agent.astep("work")

    assert result.msgs[0].content == "done"
    assert [message["role"] for message in agent.message_list] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert agent.message_list[1]["content"] == "partial response"
    assert [message["role"] for message in backend.received[1]] == [
        "user",
        "assistant",
        "user",
    ]
    assert agent.meta_info_record["response_feedback_count"] == 1
    assert agent.meta_info_record["response_feedback_reasons"] == [
        "malformed_tool_json"
    ]


@pytest.mark.asyncio
async def test_feedback_exhaustion_preserves_every_assistant_response():
    def feedback(response: ChatCompletion):
        del response
        return ResponseFeedback(reason="malformed", content="Try again.")

    agent = BaseChatAgent(
        model=SequencedModel(
            [completion(content="bad one"), completion(content="bad two")]
        ),
        token_limit=100,
        response_feedback_handler=feedback,
        max_response_feedback=1,
        step_timeout=None,
    )

    result = await agent.astep("work")

    assert result.terminated
    assert (
        agent.termination_reason is TerminationReason.MALFORMED_TOOL_CALL
    )
    assert [message["role"] for message in agent.message_list] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert agent.message_list[-1]["content"] == "bad two"


@pytest.mark.asyncio
async def test_feedback_can_intercept_invalid_tool_json_after_raw_append():
    malformed_call = ChatCompletionMessageFunctionToolCall(
        id="bad-call",
        type="function",
        function=Function(name="echo", arguments='{"value":'),
    )

    def feedback(response: ChatCompletion):
        tool_calls = response.choices[0].message.tool_calls or []
        if not tool_calls:
            return None
        arguments = tool_calls[0].function.arguments
        try:
            json.loads(arguments)
        except ValueError:
            return ResponseFeedback(
                reason="invalid_arguments_json",
                content="Tool arguments were invalid JSON; try again.",
            )
        return None

    backend = SequencedModel(
        [completion(tool_calls=[malformed_call]), completion(content="done")]
    )
    agent = BaseChatAgent(
        model=backend,
        token_limit=100,
        response_feedback_handler=feedback,
        max_response_feedback=1,
        step_timeout=None,
    )

    await agent.astep("work")

    # Invalid arguments are answered with a tool-role error result anchored
    # to the call id (never a silent skip), so every id stays paired.
    assert [message["role"] for message in agent.message_list] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert agent.message_list[1]["tool_calls"][0]["id"] == "bad-call"
    assert agent.message_list[2]["tool_call_id"] == "bad-call"
    assert "not a valid JSON object" in agent.message_list[2]["content"]
    assert agent.meta_info_record["invalid_tool_call_count"] == 1


@pytest.mark.asyncio
async def test_empty_assistant_is_not_filtered_from_response_or_memory():
    agent = BaseChatAgent(
        model=SequencedModel([completion(content="")]),
        token_limit=100,
        step_timeout=None,
    )

    result = await agent.astep("work")

    assert len(result.msgs) == 1
    assert result.msgs[0].content == ""
    assert agent.message_list[-1] == {"role": "assistant", "content": ""}


@pytest.mark.asyncio
async def test_completed_tool_result_survives_later_step_timeout():
    async def work(value: str):
        if value == "slow":
            await asyncio.sleep(1)
        return value

    calls = [
        ChatCompletionMessageFunctionToolCall(
            id=f"call-{value}",
            type="function",
            function=Function(
                name="work", arguments=json.dumps({"value": value})
            ),
        )
        for value in ("fast", "slow")
    ]
    agent = BaseChatAgent(
        model=SequencedModel([completion(tool_calls=calls)]),
        tools=[work],
        token_limit=100,
        step_timeout=0.05,
    )

    with pytest.raises(TimeoutError):
        await agent.astep("work")

    assert agent.message_list[-1] == {
        "role": "tool",
        "tool_call_id": "call-fast",
        "content": "fast",
    }


@pytest.mark.asyncio
async def test_prompt_over_token_limit_tracks_context_overflow():
    # The overflow guard runs before the NEXT request: a tool-calling turn
    # over the limit executes its tools, then the loop stops instead of
    # issuing a request that cannot fit. A final answer over the limit is
    # TASK_COMPLETE — no further request is ever needed.
    call = ChatCompletionMessageFunctionToolCall(
        id="over",
        type="function",
        function=Function(name="echo", arguments='{"value": 1}'),
    )

    def echo(value: int) -> str:
        r"""Echo.

        Args:
            value (int): Value.

        Returns:
            str: Echoed value.
        """
        return str(value)

    agent = BaseChatAgent(
        model=SequencedModel([completion(tool_calls=[call])]),
        tools=[echo],
        token_limit=3,
        step_timeout=None,
    )

    result = await agent.astep("write")

    assert result.terminated
    assert (
        agent.termination_reason is TerminationReason.CONTEXT_WINDOW_OVERFLOW
    )
    assert agent.message_list[-1]["role"] == "tool"

    finisher = BaseChatAgent(
        model=SequencedModel([completion(content="answer")]),
        token_limit=3,
        step_timeout=None,
    )
    result = await finisher.astep("write")
    assert not result.terminated
    assert finisher.termination_reason is TerminationReason.TASK_COMPLETE
    assert finisher.message_list[-1]["content"] == "answer"


@pytest.mark.asyncio
async def test_mcp_async_call_path_is_preserved():
    class MCPCallable:
        async def async_call(self, value: str):
            return f"mcp:{value}"

    class MCPFunctionTool:
        func = MCPCallable()

        def get_openai_tool_schema(self):
            return {
                "type": "function",
                "function": {"name": "mcp_echo", "parameters": {}},
            }

    tool_call = ChatCompletionMessageFunctionToolCall(
        id="mcp-call",
        type="function",
        function=Function(name="mcp_echo", arguments='{"value": "hello"}'),
    )
    backend = SequencedModel(
        [completion(tool_calls=[tool_call]), completion(content="done")]
    )
    agent = BaseChatAgent(model=backend, token_limit=100, step_timeout=None)
    agent._internal_tools["mcp_echo"] = MCPFunctionTool()

    result = await agent.astep("use MCP")

    assert result.info["tool_calls"][0].result == "mcp:hello"
    assert agent.message_list[-2]["content"] == "mcp:hello"


def test_sync_step_is_intentionally_unsupported():
    agent = BaseChatAgent(
        model=SequencedModel([completion(content="unused")]), token_limit=100
    )

    with pytest.raises(NotImplementedError, match="use astep"):
        agent.step("hello")


# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
