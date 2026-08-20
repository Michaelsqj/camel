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
"""Minimal append-only chat agent for reproducible training rollouts."""

from __future__ import annotations

import asyncio
import copy
import functools
import json
import random
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

from openai import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel

from camel.agents._types import (
    InvalidToolCall,
    ModelResponse,
    ToolCallRequest,
)
from camel.agents._utils import (
    convert_to_function_tool,
    get_info_dict,
    handle_logprobs,
    safe_model_dump,
)
from camel.agents.base import BaseAgent
from camel.logger import get_logger
from camel.memories import (
    AgentMemory,
    BaseContextCreator,
    ChatHistoryMemory,
    ContextRecord,
    MemoryRecord,
)
from camel.messages import BaseMessage, OpenAIMessage
from camel.models import BaseModelBackend, ModelManager, ModelProcessingError
from camel.responses import ChatAgentResponse
from camel.toolkits import FunctionTool, RegisteredAgentToolkit
from camel.types import (
    ChatCompletion,
    ChatCompletionMessageFunctionToolCall,
    OpenAIBackendRole,
    RoleType,
)
from camel.types.agents import ToolCallingRecord
from camel.utils import BaseTokenCounter
from camel.utils.agent_context import set_current_agent_id
from camel.utils.tool_result import ToolResult

logger = get_logger(__name__)

_RAW_OPENAI_MESSAGE_KEY = "_base_chat_agent_raw_openai_message"


@dataclass(frozen=True)
class ResponseFeedback:
    """Optional user feedback requested after preserving a model response."""

    reason: str
    content: str


ResponseFeedbackHandler = Callable[
    [ChatCompletion], Optional[ResponseFeedback]
]


class _AppendOnlyContextCreator(BaseContextCreator):
    """Return memory records in storage order without filtering or sorting."""

    def __init__(self, token_limit: int) -> None:
        """Create an exact-order context creator."""
        self._token_limit = token_limit

    @property
    def token_counter(self) -> BaseTokenCounter:
        """Token accounting comes from server usage; no local counter."""
        raise RuntimeError("No local token counter is configured")

    @property
    def token_limit(self) -> int:
        """Return the model context limit associated with this creator."""
        return self._token_limit

    def create_context(
        self, records: List[ContextRecord]
    ) -> Tuple[List[OpenAIMessage], int]:
        """Convert every record to its exact stored OpenAI message."""
        messages: List[OpenAIMessage] = []
        for record in records:
            memory_record = record.memory_record
            meta = memory_record.message.meta_dict or {}
            raw_message = meta.get(_RAW_OPENAI_MESSAGE_KEY)
            if isinstance(raw_message, dict):
                messages.append(
                    cast(OpenAIMessage, copy.deepcopy(raw_message))
                )
            else:
                messages.append(memory_record.to_openai_message())
        # Token accounting is the server's usage report; a client-side
        # recount (wrong tokenizer, O(n) per access) adds nothing.
        return messages, 0


class TerminationReason(str, Enum):
    """Why a training step ended.

    Values and cause-chain behaviour mirror
    ``strands_env.core.types.TerminationReason`` without introducing a Strands
    runtime dependency into CAMEL.
    """

    NOT_TERMINATED = "not_terminated"
    TASK_COMPLETE = "task_complete"
    MAX_TOKENS_REACHED = "max_tokens_reached"
    CONTEXT_WINDOW_OVERFLOW = "context_window_overflow"
    MAX_TOOL_ITERATIONS_REACHED = "max_tool_iterations_reached"
    MAX_TOOL_CALLS_REACHED = "max_tool_calls_reached"
    MAX_MESSAGES_REACHED = "max_messages_reached"
    MALFORMED_TOOL_CALL = "malformed_tool_call"
    RECURSION_DEPTH_EXCEEDED = "recursion_depth_exceeded"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    UNCLASSIFIED_ERROR = "unclassified_error"

    @staticmethod
    def _cause_chain(error: BaseException):
        """Yield each unique exception through explicit and implicit causes."""
        seen: set[int] = set()
        current: Optional[BaseException] = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            yield current
            current = current.__cause__ or current.__context__

    @staticmethod
    def _error_payloads(error: BaseException) -> List[Dict[str, Any]]:
        """Return OpenAI-style error payloads attached to an exception."""
        payloads: List[Dict[str, Any]] = []
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            payloads.append(body)

        response = getattr(error, "response", None)
        json_fn = getattr(response, "json", None)
        if callable(json_fn):
            try:
                body = json_fn()
            except Exception:
                body = None
            if isinstance(body, dict):
                payload = body.get("error", body)
                if isinstance(payload, dict):
                    payloads.append(payload)

        return payloads

    @staticmethod
    def _error_status_code(error: BaseException) -> Optional[int]:
        """Return an HTTP status code from SDK or raw HTTP exceptions."""
        status_code = getattr(error, "status_code", None)
        if status_code is None:
            response = getattr(error, "response", None)
            status_code = getattr(response, "status_code", None)
        return int(status_code) if isinstance(status_code, int) else None

    @classmethod
    def _is_bad_request_error(
        cls, error: BaseException, payloads: List[Dict[str, Any]]
    ) -> bool:
        """Return whether the exception represents a client request error."""
        if cls._error_status_code(error) == 400:
            return True
        error_type = type(error).__name__.lower()
        if error_type == "badrequesterror":
            return True
        return any(
            str(payload.get("type", "")).lower()
            in {"badrequesterror", "invalid_request_error"}
            for payload in payloads
        )

    @staticmethod
    def _error_message(
        error: BaseException, payloads: List[Dict[str, Any]]
    ) -> str:
        """Return provider message text from structured fields."""
        fragments = [str(error), str(getattr(error, "message", ""))]
        fragments.extend(
            str(payload.get("message", "")) for payload in payloads
        )

        response = getattr(error, "response", None)
        text = getattr(response, "text", None)
        if text is not None:
            fragments.append(str(text))

        return " ".join(fragment for fragment in fragments if fragment).lower()

    @classmethod
    def _is_context_overflow_error(cls, error: BaseException) -> bool:
        """Detect prompt/context overflow from OpenAI-compatible errors.

        OpenAI exposes a semantic ``context_length_exceeded`` code. vLLM and
        SGLang expose HTTP 400/BadRequest errors whose message names the
        model context length.
        """
        payloads = cls._error_payloads(error)
        if any(
            payload.get("code") == "context_length_exceeded"
            for payload in payloads
        ):
            return True

        message = cls._error_message(error, payloads)
        return cls._is_bad_request_error(error, payloads) and (
            "context length" in message or "context window" in message
        )

    @classmethod
    def from_error(cls, error: Optional[BaseException]) -> "TerminationReason":
        """Classify an exception and its causes into a training termination."""
        if error is None:
            return cls.TASK_COMPLETE

        chain = list(cls._cause_chain(error))
        if any(cls._is_context_overflow_error(exc) for exc in chain):
            return cls.CONTEXT_WINDOW_OVERFLOW
        error_text = " ".join(
            " ".join(
                [
                    type(exc).__name__.lower(),
                    str(exc).lower(),
                    str(getattr(exc, "message", "")).lower(),
                ]
            )
            for exc in chain
        )
        if "maxtoken" in error_text or "max token" in error_text:
            return cls.MAX_TOKENS_REACHED
        if "maxtooliteration" in error_text:
            return cls.MAX_TOOL_ITERATIONS_REACHED
        if "maxtoolcall" in error_text:
            return cls.MAX_TOOL_CALLS_REACHED
        if "maxmessage" in error_text:
            return cls.MAX_MESSAGES_REACHED
        if any(isinstance(exc, RecursionError) for exc in chain):
            return cls.RECURSION_DEPTH_EXCEEDED
        if "timeout" in error_text:
            return cls.TIMEOUT
        if any(
            marker in error_text
            for marker in ("connection", "disconnected", "connecterror")
        ):
            return cls.CONNECTION_ERROR
        return cls.UNCLASSIFIED_ERROR


class BaseChatAgent(BaseAgent):
    """An async-only CAMEL agent with raw append-only memory.

    This class deliberately inherits directly from :class:`BaseAgent`. It has
    an ``AgentMemory`` but bypasses score-based context processing, windows,
    pruning, and automatic summarization. Internal CAMEL ``FunctionTool``
    objects and async MCP tools remain supported.
    """

    def __init__(
        self,
        system_message: Optional[Union[BaseMessage, str]] = None,
        model: Optional[Union[BaseModelBackend, ModelManager]] = None,
        memory: Optional[AgentMemory] = None,
        tools: Optional[List[Union[FunctionTool, Callable[..., Any]]]] = None,
        toolkits_to_register_agent: Optional[
            List[RegisteredAgentToolkit]
        ] = None,
        token_limit: Optional[int] = None,
        max_iteration: Optional[int] = None,
        agent_id: Optional[str] = None,
        tool_execution_timeout: Optional[float] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        response_feedback_handler: Optional[ResponseFeedbackHandler] = None,
        max_response_feedback: int = 2,
        step_timeout: Optional[float] = None,
        tool_markup_markers: Optional[List[str]] = None,
        max_consecutive_length: Optional[int] = None,
        max_consecutive_malformed: Optional[int] = None,
    ) -> None:
        """Configure an async agent whose ``AgentMemory`` is never processed.

        Tools are normalized to CAMEL ``FunctionTool`` objects. ``token_limit``
        defaults to the model limit, while tool and whole-step timeouts remain
        optional so training frameworks can choose their own limits.
        """
        if model is None:
            raise ValueError("BaseChatAgent requires a model backend")
        if not isinstance(model, (BaseModelBackend, ModelManager)):
            raise TypeError("model must be a BaseModelBackend or ModelManager")
        self.model_backend = (
            model if isinstance(model, ModelManager) else ModelManager(model)
        )
        if self.model_backend.model_config_dict.get("stream", False):
            raise ValueError(
                "BaseChatAgent only supports non-streaming models"
            )

        self.agent_id = agent_id or str(uuid.uuid4())
        self._original_system_message = (
            BaseMessage.make_system_message(system_message)
            if isinstance(system_message, str)
            else system_message
        )
        self.role_name = (
            getattr(self._original_system_message, "role_name", None)
            or "assistant"
        )
        self.role_type = (
            getattr(self._original_system_message, "role_type", None)
            or RoleType.ASSISTANT
        )

        self._internal_tools: Dict[str, FunctionTool] = {
            tool.get_function_name(): tool
            for tool in [
                convert_to_function_tool(tool) for tool in (tools or [])
            ]
        }
        for toolkit in toolkits_to_register_agent or []:
            toolkit.register_agent(cast(Any, self))

        self._token_limit = (
            token_limit
            if token_limit is not None
            else self.model_backend.token_limit
        )
        self._context_creator = _AppendOnlyContextCreator(
            token_limit=self._token_limit
        )
        self._memory = memory or ChatHistoryMemory(
            context_creator=self._context_creator,
            window_size=None,
            agent_id=self.agent_id,
        )
        self._memory.agent_id = self.agent_id
        self.max_iteration = max_iteration
        self.tool_execution_timeout = tool_execution_timeout
        self.retry_attempts = max(1, retry_attempts)
        self.retry_delay = max(0.0, retry_delay)
        self.response_feedback_handler = response_feedback_handler
        self.max_response_feedback = max(0, max_response_feedback)
        # Tool-call begin markers of the model family (e.g. "<tool_call>").
        # A bare SGLang server signals a tool-call parse failure only by
        # leaving raw markup in ``content``; empty list disables the check.
        self.tool_markup_markers = list(tool_markup_markers or [])
        # Consecutive-failure budgets, reset by any fully-valid turn.
        # ``max_response_feedback`` seeds both for backward compatibility.
        self.max_consecutive_length = (
            self.max_response_feedback
            if max_consecutive_length is None
            else max(0, max_consecutive_length)
        )
        self.max_consecutive_malformed = (
            self.max_response_feedback
            if max_consecutive_malformed is None
            else max(0, max_consecutive_malformed)
        )
        self.step_timeout = step_timeout
        self.terminated = False
        self.termination_reason = TerminationReason.NOT_TERMINATED
        self.last_error: Optional[BaseException] = None
        self.meta_info_record: Dict[str, Any] = {}
        self.reset()

    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Reject synchronous use; training rollouts must call ``astep``."""
        raise NotImplementedError("BaseChatAgent is async-only; use astep()")

    def reset(self) -> None:
        """Clear the trajectory and reset all stateful model backends."""
        self.memory.clear()
        if self._original_system_message is not None:
            self._write_openai_message(
                self._original_system_message.to_openai_system_message()
            )
        self.terminated = False
        self.termination_reason = TerminationReason.NOT_TERMINATED
        self.last_error = None
        self.meta_info_record = self._new_meta_info()
        for backend in self.model_backend.models:
            reset = getattr(backend, "reset", None)
            if callable(reset):
                reset()

    @property
    def memory(self) -> AgentMemory:
        """Return the canonical append-only agent memory."""
        return self._memory

    @property
    def message_list(self) -> List[OpenAIMessage]:
        """Return the exact unprocessed messages currently stored in memory."""
        messages, _ = self._context_creator.create_context(
            self.memory.retrieve()
        )
        return messages

    @property
    def chat_history(self) -> List[OpenAIMessage]:
        """Return a defensive view of the raw training trajectory."""
        return self.message_list

    def _write_openai_message(self, message: OpenAIMessage) -> None:
        """Append one exact OpenAI message to ``AgentMemory``."""
        raw_message = copy.deepcopy(dict(message))
        role = str(raw_message.get("role", "assistant"))
        backend_role = {
            "system": OpenAIBackendRole.SYSTEM,
            "developer": OpenAIBackendRole.DEVELOPER,
            "user": OpenAIBackendRole.USER,
            "assistant": OpenAIBackendRole.ASSISTANT,
            "tool": OpenAIBackendRole.TOOL,
        }.get(role, OpenAIBackendRole.ASSISTANT)
        role_type = {
            "system": RoleType.SYSTEM,
            "developer": RoleType.SYSTEM,
            "user": RoleType.USER,
        }.get(role, RoleType.ASSISTANT)
        content = raw_message.get("content")
        reasoning_content = raw_message.get("reasoning_content")
        stored_message = BaseMessage(
            role_name=role,
            role_type=role_type,
            meta_dict={_RAW_OPENAI_MESSAGE_KEY: raw_message},
            content=content if isinstance(content, str) else "",
            reasoning_content=(
                reasoning_content
                if isinstance(reasoning_content, str)
                else None
            ),
        )
        self.memory.write_record(
            MemoryRecord(
                message=stored_message,
                role_at_backend=backend_role,
                timestamp=time.time_ns() / 1_000_000_000,
                agent_id=self.agent_id,
            )
        )

    @property
    def tool_dict(self) -> Dict[str, FunctionTool]:
        """Return registered internal tools by function name."""
        return self._internal_tools

    @staticmethod
    def _new_meta_info() -> Dict[str, Any]:
        """Create zeroed usage, tool, termination, and error metadata."""
        return {
            "iteration_count": 0,
            "termination_reason": TerminationReason.NOT_TERMINATED,
            "max_tool_calls_per_turn": 0,
            "total_tool_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error_type": None,
            "error_message": None,
            "context_tokens": 0,
            "invalid_tool_call_count": 0,
            "feedback_events": [],
            "response_feedback_count": 0,
            "response_feedback_reasons": [],
        }

    def _get_full_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI schemas for every registered internal or MCP tool."""
        return [
            tool.get_openai_tool_schema()
            for tool in self._internal_tools.values()
        ]

    @staticmethod
    def _create_token_usage_tracker() -> Dict[str, int]:
        """Create counters accumulated across all model calls in one step."""
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @staticmethod
    def _update_token_usage_tracker(
        tracker: Dict[str, int], usage: Dict[str, Any]
    ) -> None:
        """Add one completion's reported token usage into step totals."""
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            tracker[key] += int(usage.get(key) or 0)

    def _set_termination(
        self,
        reason: TerminationReason,
        error: Optional[BaseException] = None,
    ) -> None:
        """Record the final reason and optional exception in training state."""
        self.termination_reason = reason
        self.meta_info_record["termination_reason"] = reason
        self.last_error = error
        if error is not None:
            self.meta_info_record["error_type"] = type(error).__name__
            self.meta_info_record["error_message"] = str(error)

    async def astep(
        self,
        input_message: Union[BaseMessage, str],
        response_format: Optional[Type[BaseModel]] = None,
    ) -> ChatAgentResponse:
        """Append one user turn and run model/tool turns to termination.

        ``step_timeout`` wraps the complete operation, including model retries
        and sequential tool calls.
        """
        set_current_agent_id(self.agent_id)
        self.meta_info_record = self._new_meta_info()
        self.termination_reason = TerminationReason.NOT_TERMINATED
        self.last_error = None
        self.terminated = False

        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )
        self._write_openai_message(input_message.to_openai_user_message())

        task = self._run_loop(response_format)
        try:
            if self.step_timeout is None:
                return await task
            return await asyncio.wait_for(task, timeout=self.step_timeout)
        except BaseException as error:
            if self.termination_reason is TerminationReason.NOT_TERMINATED:
                self._set_termination(
                    TerminationReason.from_error(error), error
                )
            raise

    # Fixed feedback strings: they become permanent context and masked
    # observation tokens in training data — keep them short and deterministic.
    TRUNCATED_FEEDBACK = (
        "Your previous response was cut off by the output-length limit and "
        "was preserved, but nothing was executed. Continue with a shorter, "
        "complete response."
    )
    PARSER_FAILED_FEEDBACK = (
        "Your previous response was preserved, but its tool-call format was "
        "invalid and nothing was executed. Re-issue one complete, valid tool "
        "call or give a final answer."
    )
    INVALID_ARGS_RESULT = (
        "Tool call was not executed: arguments were not a valid JSON object "
        "({error}). Re-issue the call with valid JSON arguments."
    )

    async def _run_loop(
        self, response_format: Optional[Type[BaseModel]]
    ) -> ChatAgentResponse:
        """Run model and tool turns until the step terminates.

        Invariants:

        * append-only: every raw assistant response is preserved before any
          interpretation, and recovery only ever appends messages — the
          context of call ``i+1`` strictly extends call ``i`` (token-in/
          token-out linearity, prefix-cache friendly);
        * every ``tool_call_id`` of a preserved assistant message receives
          exactly one tool-role response, in positional order;
        * model-behavior outcomes return gracefully with a classified
          ``TerminationReason``; only infrastructure errors raise.
        """
        tool_call_records: List[ToolCallingRecord] = []
        usage = self._create_token_usage_tracker()
        response: Optional[ModelResponse] = None
        last_prompt_tokens = 0
        context_tokens = 0
        consecutive_length = 0
        consecutive_malformed = 0

        while True:
            # Never issue a request that cannot fit: under append-only
            # linearity the previous turn's total_tokens is an exact floor
            # for the next prompt.
            if context_tokens and self._next_request_would_overflow(
                context_tokens
            ):
                self.terminated = True
                self._set_termination(
                    TerminationReason.CONTEXT_WINDOW_OVERFLOW
                )
                break
            try:
                completion = await self._aget_model_response(
                    self.message_list,
                    response_format=response_format,
                    tool_schemas=self._get_full_tool_schemas(),
                )
            except BaseException as error:
                # Backstop: the server rejected the prompt as too long
                # (HTTP 400 naming the context length). Terminate gracefully;
                # anything else is an infrastructure failure and raises.
                if (
                    TerminationReason.from_error(error)
                    is TerminationReason.CONTEXT_WINDOW_OVERFLOW
                ):
                    self.terminated = True
                    self._set_termination(
                        TerminationReason.CONTEXT_WINDOW_OVERFLOW, error
                    )
                    break
                raise
            iteration = self.meta_info_record["iteration_count"] + 1
            self.meta_info_record["iteration_count"] = iteration

            # 1. Preserve the raw assistant message before interpretation.
            raw_assistant = completion.choices[0].message.model_dump(
                exclude_none=True
            )
            raw_assistant["role"] = "assistant"
            self._write_openai_message(cast(OpenAIMessage, raw_assistant))

            # 2. Account usage. Under append-only linearity the last
            #    ``total_tokens`` is an exact floor for the next prompt.
            response = self._parse_model_response(completion)
            completion_usage = (
                safe_model_dump(completion.usage) if completion.usage else {}
            )
            self._update_token_usage_tracker(usage, completion_usage)
            for key in usage:
                self.meta_info_record[key] = usage[key]
            last_prompt_tokens = int(
                completion_usage.get("prompt_tokens") or 0
            )
            context_tokens = int(completion_usage.get("total_tokens") or 0)
            self.meta_info_record["context_tokens"] = context_tokens

            verdict, invalid, feedback = self._classify_response(
                completion, response
            )

            if verdict == "complete":
                self._set_termination(TerminationReason.TASK_COMPLETE)
                break

            if verdict in ("truncated", "parser_failed"):
                # Preserved, nothing executed; append one corrective user
                # message and retry, bounded by the consecutive budget.
                if verdict == "truncated":
                    consecutive_length += 1
                    exhausted = (
                        consecutive_length > self.max_consecutive_length
                    )
                    reason = TerminationReason.MAX_TOKENS_REACHED
                    text = self.TRUNCATED_FEEDBACK
                    detail = None
                else:
                    consecutive_malformed += 1
                    exhausted = (
                        consecutive_malformed > self.max_consecutive_malformed
                    )
                    reason = TerminationReason.MALFORMED_TOOL_CALL
                    text = (
                        feedback.content
                        if feedback is not None
                        else self.PARSER_FAILED_FEEDBACK
                    )
                    detail = feedback.reason if feedback is not None else None
                self._record_feedback_event(iteration, verdict, detail)
                if exhausted:
                    self.terminated = True
                    self._set_termination(reason)
                    break
                # Budgets before appending: never leave a trailing feedback
                # message no model call will consume.
                if self._next_request_would_overflow(context_tokens):
                    self.terminated = True
                    self._set_termination(
                        TerminationReason.CONTEXT_WINDOW_OVERFLOW
                    )
                    break
                if self._iteration_budget_exhausted(iteration):
                    break
                self._write_openai_message({"role": "user", "content": text})
                continue

            # verdict == "calls": answer every tool_call_id in positional
            # order — real execution for valid calls, synthesized error
            # results for invalid ones (a silently skipped id would make the
            # next template render diverge from what the model sampled).
            requests = list(response.tool_call_requests or [])
            self.meta_info_record["max_tool_calls_per_turn"] = max(
                self.meta_info_record["max_tool_calls_per_turn"],
                len(requests) + len(invalid),
            )
            entries = sorted(
                [(req.index or 0, req, None) for req in requests]
                + [(bad.index, None, bad) for bad in invalid],
                key=lambda entry: entry[0],
            )
            for _, request, bad in entries:
                if request is not None:
                    record = await self._aexecute_tool(request)
                    tool_call_records.append(record)
                    self._write_openai_message(
                        {
                            "role": "tool",
                            "tool_call_id": record.tool_call_id,
                            "content": self._tool_result_content(
                                record.result
                            ),
                        }
                    )
                else:
                    self._write_openai_message(
                        {
                            "role": "tool",
                            "tool_call_id": bad.tool_call_id,
                            "content": self.INVALID_ARGS_RESULT.format(
                                error=bad.error
                            ),
                        }
                    )
            self.meta_info_record["total_tool_calls"] = len(tool_call_records)

            if invalid:
                self.meta_info_record["invalid_tool_call_count"] += len(
                    invalid
                )
                self._record_feedback_event(
                    iteration,
                    "invalid_args",
                    "; ".join(bad.error for bad in invalid),
                )
                consecutive_malformed += 1
                if consecutive_malformed > self.max_consecutive_malformed:
                    self.terminated = True
                    self._set_termination(
                        TerminationReason.MALFORMED_TOOL_CALL
                    )
                    break
            else:
                # A fully-valid turn proves the model recovered.
                consecutive_length = 0
                consecutive_malformed = 0

            # Turn budget after answering: every id got its response.
            if self._iteration_budget_exhausted(iteration):
                break

        info = get_info_dict(
            response.response_id if response is not None else None,
            usage,
            response.finish_reasons if response is not None else [],
            last_prompt_tokens,
            tool_call_records,
        )
        info["training"] = copy.deepcopy(self.meta_info_record)
        return ChatAgentResponse(
            msgs=list(response.output_messages) if response is not None else [],
            terminated=self.terminated,
            info=info,
        )

    def _classify_response(
        self, completion: ChatCompletion, response: ModelResponse
    ) -> Tuple[str, List[InvalidToolCall], Optional[ResponseFeedback]]:
        """Apply the classification ladder to one preserved response.

        Returns ``(verdict, invalid_calls, handler_feedback)`` where verdict is
        ``truncated`` | ``calls`` | ``parser_failed`` | ``complete``.
        Truncation outranks everything: tool calls in a cut-off response
        reflect an unfinished intent and are never executed.
        """
        invalid = list(response.invalid_tool_calls or [])
        if "length" in response.finish_reasons:
            return "truncated", invalid, None
        if response.tool_call_requests or invalid:
            return "calls", invalid, None
        content = completion.choices[0].message.content or ""
        if any(marker in content for marker in self.tool_markup_markers):
            return "parser_failed", invalid, None
        if self.response_feedback_handler is not None:
            feedback = self.response_feedback_handler(completion)
            if feedback is not None:
                return "parser_failed", invalid, feedback
        return "complete", invalid, None

    def _iteration_budget_exhausted(self, iteration: int) -> bool:
        """Enforce the turn budget; every model call counts."""
        if self.max_iteration is None or iteration < self.max_iteration:
            return False
        self.terminated = True
        self._set_termination(TerminationReason.MAX_TOOL_ITERATIONS_REACHED)
        return True

    def _completion_token_budget(self) -> int:
        """Output tokens to reserve for the next request (0 when unknown)."""
        value = self.model_backend.model_config_dict.get("max_tokens")
        return int(value) if isinstance(value, (int, float)) else 0

    def _next_request_would_overflow(self, context_tokens: int) -> bool:
        """Whether context so far plus the reserved output exceeds the limit."""
        return (
            self._token_limit is not None
            and context_tokens + self._completion_token_budget()
            > self._token_limit
        )

    def _record_feedback_event(
        self, iteration: int, kind: str, detail: Optional[str] = None
    ) -> None:
        """Record one abnormal-response event for training-side analysis."""
        self.meta_info_record["feedback_events"].append(
            {"iteration": iteration, "kind": kind, "detail": detail}
        )
        reasons = self.meta_info_record["response_feedback_reasons"]
        reasons.append(detail or kind)
        self.meta_info_record["response_feedback_count"] = len(reasons)

    _RETRYABLE_ERRORS = (
        RateLimitError,
        APIConnectionError,  # includes APITimeoutError
        InternalServerError,  # any 5xx
    )

    async def _aget_model_response(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]],
        tool_schemas: List[Dict[str, Any]],
    ) -> ChatCompletion:
        """Return one raw completion, retrying transient infra failures.

        Client errors (4xx) propagate immediately: the loop classifies
        context overflow; anything else is a caller bug.
        """
        last_error: Optional[BaseException] = None
        for attempt in range(self.retry_attempts):
            try:
                raw = await self.model_backend.arun(
                    messages, response_format, tool_schemas or None
                )
            except self._RETRYABLE_ERRORS as error:
                last_error = error
                if attempt + 1 < self.retry_attempts:
                    delay = random.uniform(
                        0, min(self.retry_delay * (2**attempt), 60.0)
                    )
                    await asyncio.sleep(delay)
                continue
            if raw is not None:
                break
        else:
            raise ModelProcessingError(
                "Unable to process messages: "
                f"{last_error or 'empty model response'}"
            ) from last_error

        if not isinstance(raw, ChatCompletion):
            raise TypeError(
                f"Expected ChatCompletion, got {type(raw).__name__}"
            )
        return raw

    def _parse_model_response(self, response: ChatCompletion) -> ModelResponse:
        """Convert a completion into messages and ordered tool calls."""
        output_messages: List[BaseMessage] = []
        for choice in response.choices:
            message = choice.message
            meta: Dict[str, Any] = {}
            if logprobs := handle_logprobs(choice):
                meta["logprobs_info"] = logprobs
            output_messages.append(
                BaseMessage(
                    role_name=self.role_name,
                    role_type=self.role_type,
                    meta_dict=meta,
                    content=message.content or "",
                    parsed=getattr(message, "parsed", None),
                    reasoning_content=getattr(
                        message, "reasoning_content", None
                    ),
                )
            )

        requests: List[ToolCallRequest] = []
        invalid: List[InvalidToolCall] = []
        for index, tool_call in enumerate(
            response.choices[0].message.tool_calls or []
        ):
            function_tool_call = cast(
                ChatCompletionMessageFunctionToolCall, tool_call
            )
            raw_arguments = function_tool_call.function.arguments
            try:
                # ""/null is the zero-argument convention, not a failure.
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except (TypeError, json.JSONDecodeError) as error:
                invalid.append(
                    InvalidToolCall(
                        tool_call_id=function_tool_call.id,
                        tool_name=function_tool_call.function.name,
                        error=f"invalid JSON: {error}",
                        index=index,
                    )
                )
                continue
            if not isinstance(arguments, dict):
                invalid.append(
                    InvalidToolCall(
                        tool_call_id=function_tool_call.id,
                        tool_name=function_tool_call.function.name,
                        error="arguments are not a JSON object",
                        index=index,
                    )
                )
                continue
            requests.append(
                ToolCallRequest(
                    tool_name=function_tool_call.function.name,
                    args=arguments,
                    tool_call_id=function_tool_call.id,
                    extra_content=getattr(
                        function_tool_call, "extra_content", None
                    ),
                    index=index,
                )
            )
        usage = safe_model_dump(response.usage) if response.usage else {}
        return ModelResponse(
            response=response,
            tool_call_requests=requests or None,
            invalid_tool_calls=invalid or None,
            output_messages=output_messages,
            finish_reasons=[
                str(choice.finish_reason) for choice in response.choices
            ],
            usage_dict=usage,
            response_id=response.id or "",
        )

    async def _aexecute_tool(
        self, request: ToolCallRequest
    ) -> ToolCallingRecord:
        """Execute CAMEL functions and MCP tools without memory writes."""
        tool = self._internal_tools.get(request.tool_name)
        if tool is None:
            result: Any = (
                "Tool execution failed: Tool "
                f"'{request.tool_name}' not found in registered tools"
            )
        else:
            try:
                call = self._invoke_tool(tool, request.args)
                result = (
                    await asyncio.wait_for(
                        call, timeout=self.tool_execution_timeout
                    )
                    if self.tool_execution_timeout is not None
                    else await call
                )
            except Exception as error:
                result = (
                    f"Tool execution failed: Error executing async tool "
                    f"'{request.tool_name}': {error}"
                )
                logger.warning(result)

        images = result.images if isinstance(result, ToolResult) else None
        return ToolCallingRecord(
            tool_name=request.tool_name,
            args=request.args,
            result=result,
            tool_call_id=request.tool_call_id,
            images=images,
        )

    @staticmethod
    async def _invoke_tool(tool: FunctionTool, args: Dict[str, Any]) -> Any:
        """Invoke a tool without blocking the agent event loop.

        MCP ``func.async_call`` is preferred, then ``FunctionTool.async_call``
        (which runs sync functions in its own executor); plain callables fall
        back to the loop's executor.
        """
        if hasattr(tool, "func") and hasattr(tool.func, "async_call"):
            return await tool.func.async_call(**args)
        if callable(getattr(tool, "async_call", None)):
            return await tool.async_call(**args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(tool, **args)
        )

    @staticmethod
    def _tool_result_content(result: Any) -> str:
        """Serialize a tool result for the OpenAI ``tool`` memory message."""
        return result.text if isinstance(result, ToolResult) else str(result)


__all__ = ["BaseChatAgent", "ResponseFeedback", "TerminationReason"]
