# Agent-loop test suites

Design: `camel/agents/BASE_CHAT_AGENT_LOOP_DESIGN.md`

- `offline_suite.py` — 19 deterministic corner cases on a scripted backend
  (classification ladder, budgets, graceful termination, positional tool
  pairing, proactive/400 overflow, infra retry, append-only wire order).
- `live_suite.py [base_url]` — 7 cases against a real SGLang endpoint
  (default `http://10.220.51.13:31001/v1`, model `model`): health, plain
  completion, tool round trip, length-truncation feedback, proactive context
  budget, oversized-prompt behavior probe, multi-turn linearity + prefix
  cache (`prompt_token_ids` extension, `cached_tokens`).

Run with the repo venv: `.venv/bin/python test/agents/agent_loop/<suite>.py`

`results/` keeps one timestamped log per run (plus `*_trajectories.json`
with full message lists, meta records, and per-turn token-id lengths for the
live runs). Key live finding: GLM templates require
`chat_template_kwargs: {"clear_thinking": false}` or a user-role feedback
message after a truncated turn clears prior reasoning and breaks
prompt-prefix linearity.
