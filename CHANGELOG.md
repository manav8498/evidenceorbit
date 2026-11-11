# Changelog

## 0.1.0 — 2026-04-21

First public cut of the latent-error sidecar.

- Canonical trajectory schema (`0.1.0`) for LLM calls, tool calls and results, observations, claims, goals, contracts, human messages, and checkpoints.
- Incremental evidence graph with `supports`, `contradicts`, `derives_from`, and `violates`.
- Hybrid detector: deterministic fast path, judge only for soft candidates. Default judge is a logged heuristic; `LLMJudge` accepts an injected client.
- Origin attribution to the producing step, not the earliest unrelated ancestor.
- Surgical recovery proposals. Propose-only by default. Destructive external effects are not auto-rewound.
- In-memory and file checkpoint adapters.
- Append-only content-addressed audit log (JSONL).
- CLI (`analyze`, `wrap`, `mcp`, `bench`, `schemas`), MCP server, ReAct adapter, LangGraph-style adapter.
- 30-trajectory benchmark. v0.1.0 result: detection recall 1.0, attribution accuracy 1.0, recovery success 1.0, false-positive rate 0.0. Fast-path mean overhead about 0.8 ms per step on the published microbench.

The Python public API and the JSON schemas are stable for the 0.1.x line.
