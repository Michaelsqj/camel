# BaseChatAgent loop design (SGLang-first, TITO-linear)

Design for `camel/agents/base_chat_agent.py`. Target backend contract is the
plain SGLang OpenAI-compatible signature; any conforming proxy (e.g. a session
server) works automatically. No proxy-specific field is ever required.

## Goals

1. **Linear trajectory (token-in/token-out).** Every model call's context is a
   strict append of the previous call's context: no rollback, rewrite,
   reorder, or drop. Sampled tokens stay in context verbatim, so
   `prompt[i+1] ⊇ prompt[i] + completion[i]` and prefix caches stay hot.
2. **Feedback-for-retry.** Recoverable model failures (truncation, malformed
   tool calls) append corrective feedback and retry in-context under bounded
   *consecutive* budgets.
3. **Graceful termination.** Model-behavior outcomes always return a
   `ChatAgentResponse` with a classified `TerminationReason`. Exceptions are
   reserved for infrastructure failures (connection, 5xx, timeout).
4. **Full meta recording.** Every abnormal event is recorded in
   `meta_info_record` for training-side weighting/filtering.

## Backend contract (verified against SGLang `serving_chat.py`)

| Signal | Semantics |
|---|---|
| `finish_reason` | `stop` / `length` / `abort` / `tool_calls` (rewritten from `stop` only when server parsing succeeded) |
| `tool_calls[].function.arguments` | JSON string; `""`/`null` used by zero-arg conventions; may be invalid JSON |
| `tool_calls[].id` | unique uuid by default; some parser modes round-trip model ids that **can collide** → pair results by position, never by id |
| parse failure | **silent**: `tool_calls` absent, raw markup left in `content`, `finish_reason="stop"` |
| `usage` | prompt/completion/total tokens every turn — the universal token accounting |
| overflow | HTTP 400, message contains `"context length"` |
| opt-in extras | `return_prompt_token_ids` → `choice.prompt_token_ids`; `return_meta_info` → `choice.meta_info.output_token_logprobs` etc. |

Deployment preconditions: `--allow-auto-truncate` off (silent prefix
truncation breaks linearity); chat-template kwargs that keep prior-turn
reasoning fixed for the whole rollout.

## Client obligations

1. Messages go to the wire verbatim in memory order (backend
   `preprocess_messages` must be identity — `BaseChatModel` complies).
2. Every assistant response is preserved to memory **before** interpretation.
3. Every `tool_call_id` receives exactly one tool-role response, in
   `tool_calls` order — malformed calls get *synthesized error results*, never
   silent skips (an unanswered id makes the next template render diverge from
   what the model saw).
4. Identical sampling parameters across all calls of one rollout.
5. Token accounting is the server's `usage` report only — the client never
   retokenizes (no local tokenizer matches the server's, and a recount per
   turn is O(n) waste).

## Classification ladder (per response, in order)

```
L1  finish_reason == "length"                          → TRUNCATED
L2  tool_calls present → per-call client validation:
      arguments "" / null       → {}      (valid, zero-arg convention)
      json fails / non-dict     → invalid (id-anchored, positional)
      else                      → valid
L3  no tool_calls AND (content contains a configured tool-markup marker
    OR the optional response_feedback_handler fires)   → PARSER_FAILED
L4  otherwise                                          → COMPLETE
```

L1 outranks all: a truncated turn's tool calls reflect unfinished intent and
are never executed. L3 markers (`tool_markup_markers`, e.g. `["<tool_call>"]`)
mirror the trigger token of the server's own detector; empty list disables L3.
`response_feedback_handler` is the optional enrichment hook (e.g. proxy
`meta_info` hints); the built-in ladder runs regardless.

## Recovery policy

| Verdict | Appended (always append-only) | Executes | Budget | Exhaustion → return |
|---|---|---|---|---|
| TRUNCATED | one user notice | nothing | `max_consecutive_length` | `MAX_TOKENS_REACHED` |
| invalid call | tool-role error result on that id; valid siblings execute; results strictly positional | partial | `max_consecutive_malformed` | `MALFORMED_TOOL_CALL` |
| PARSER_FAILED | one user notice | nothing | shares malformed budget | `MALFORMED_TOOL_CALL` |
| all valid | tool results in order | all | resets both counters | — |
| COMPLETE | — | — | — | `TASK_COMPLETE` |

Budgets are **consecutive** (reset on any fully-valid turn): a recovering
rollout is healthy; global boundedness comes from `max_iteration`, which
counts **every** model call including feedback retries.

Cross-cutting guards (each a graceful return):

- Turn budget → `MAX_TOOL_ITERATIONS_REACHED`. Checked before appending
  feedback (no trailing unused feedback) and after answering tool calls
  (no dangling ids).
- Token budget, proactive: `usage.total_tokens + max_tokens > token_limit`
  → `CONTEXT_WINDOW_OVERFLOW`, checked **immediately before each model
  request** (and before appending retry feedback, so none dangles). Under
  linearity, `total_tokens` is an exact floor for the next prompt. Placement
  matters: the current turn's tool calls still execute and get answered, and
  a final COMPLETE answer near the ceiling stays `TASK_COMPLETE` — overflow
  only applies when another request is actually needed. The
  400-`"context length"` classifier remains as a backstop, also graceful.
- Infra errors: jittered exponential backoff retries inside
  `_aget_model_response` (429 / connection / timeout / 5xx); raise on
  exhaustion — trial-level failure, not a trajectory outcome.

Feedback texts are fixed short strings: they become permanent context and
masked observation tokens in training data — determinism over eloquence.

## Meta recording (`meta_info_record`)

`iteration_count`, `prompt_tokens` / `completion_tokens` / `total_tokens`
(accumulated), `context_tokens` (last `total_tokens`, the linear high-water),
`termination_reason`, `max_tool_calls_per_turn`, `total_tool_calls`,
`invalid_tool_call_count`, `feedback_events: [{iteration, kind, detail}]`,
`response_feedback_count` / `response_feedback_reasons` (legacy-compatible),
`error_type` / `error_message` on infra failure.

## Out of scope here (adapter follow-ups)

- Request `return_prompt_token_ids` + `return_meta_info` + `logprobs` for
  rollout recording (`return_token_ids` is not an SGLang flag).
- Optional linearity probe: warn when `prompt_ids[i+1]` fails to extend
  `prompt_ids[i] + output_ids[i]`.

## Tests

- `test/agents/agent_loop/offline_suite.py` — scripted backend, every ladder
  row, budgets, overflow, retry, wire-order linearity. Deterministic.
- `test/agents/agent_loop/live_suite.py` — real SGLang endpoint: health,
  completion, tool loop, induced `length`, oversized-prompt behavior probe,
  proactive stop, prefix-cache evidence (`cached_tokens`), token-id linearity
  check. (The overflow-400 classification is deterministic offline; the live
  router accepts ≥300k-token inputs, so the proactive budget is the guard.)
- Results preserved under `test/agents/agent_loop/results/`.
