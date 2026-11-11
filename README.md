# EvidenceOrbit

Framework-agnostic sidecar that catches latent errors in long-horizon agent trajectories. It watches for evidence the agent already has and did not use, names the step where the bad commitment entered, and proposes a surgical rewind instead of a full restart.

Long runs fail because an early claim is wrong, later steps build on it, and a tool result or observation that contradicts it is sitting in the trace unused. Most tooling diagnoses that after the fact, and only inside one framework. EvidenceOrbit closes the loop while the trajectory is still live.

Apache-2.0. Python 3.11+. Schema and public API are stable across 0.1.x.

## Benchmark (v0.1.0)

30 curated trajectories: early wrong decisions, ignored tool results, contract violations, multi-step compounding, internal conflicts, plus clean and already-repaired controls.

| Metric | Result |
| --- | --- |
| Detection recall | 1.000 |
| Attribution accuracy | 1.000 |
| Recovery success | 1.000 |
| False-positive rate | 0.000 |
| Fast-path ingest | ~0.8 ms mean, ~1.9 ms p95 (80 unrelated steps) |

```bash
python -m evidenceorbit bench
```

## Quickstart

```bash
pip install -e ".[dev]"
python -m evidenceorbit analyze examples/traces/early_price.jsonl --format summary
python examples/react_loop.py
```

Library:

```python
from evidenceorbit import EvidenceOrbit

report = EvidenceOrbit().analyze(events)  # JSON objects or TrajectoryEvent
for detection in report.detections:
    origin = report.attribution_for(detection.id)
    proposal = report.proposal_for(detection.id)
    print(detection.explanation)
    print(origin.propagation_summary)
    print(proposal.corrective_signal.instruction)
```

Live loop: `ReActAdapter` (push) or `LangGraphAdapter` (pull on a `messages` state). Neither imports its framework. See [docs/adapters.md](docs/adapters.md).

```bash
python -m evidenceorbit mcp          # MCP tools over stdio
python -m evidenceorbit wrap -- cmd  # ingest a process stdout
python -m evidenceorbit schemas --out schemas
```

MCP tools: `check_trajectory`, `get_latent_errors`, `propose_recovery`, `apply_recovery`. Apply is gated on `confirm: true` and refuses destructive side-effects unless you explicitly arm that policy.

## How it works

```mermaid
flowchart TD
  runtime[Agent runtime]
  inst[Push events or adapter pull]
  graph[Evidence graph]
  det[Hybrid detector]
  attr[Origin attributor]
  plan[Recovery planner]
  ckpt[Checkpoint adapter]
  audit[Hash-chained audit log]

  runtime --> inst --> graph --> det --> attr --> plan --> ckpt
  det --> audit
  attr --> audit
  plan --> audit
```

1. **Events.** One ordered schema for LLM calls, tool calls, tool results, observations, claims, goals, contracts, human messages, and checkpoints. Push them yourself, or let an adapter pull them out of a message list.
2. **Evidence graph.** Claims, tool results, and observations become nodes. Edges are `supports`, `contradicts`, `derives_from`, and `violates`.
3. **Detector.** Fast path compares normalized facts (subject, predicate, value, polarity) and contract constraints. A judge runs only when text *looks* contradictory and no structured fact already decided it. High-confidence structural hits never wait on a model.
4. **Attribution.** The origin is the LLM call or claim where the bad value entered, not merely the oldest event in the session. Compounded downstream claims point back at that same origin.
5. **Recovery.** Rewind to just before the origin, restore the latest safe checkpoint, and inject the unused evidence as a corrective signal. Propose-only is the default. If a send, delete, charge, deploy, or other write already happened, the proposal is `external_blocked` and will not auto-apply.

Every detection carries a confidence, the exact event ids, and a one-sentence explanation. Judge calls record model, prompt hash, and temperature on the detection and in the audit log.

## Design rules

- Evidence first. No detection without fragments you can open.
- Surgical, not a restart. The rewind point is the origin step.
- Cheap until there is a candidate. Healthy steps do not call a model.
- Framework-agnostic. MCP, a ReAct loop, and a LangGraph-shaped node are all just event sources.
- Auditable. The log is append-only and content-addressed.
- Safe by default. Beliefs can be rewound. External side-effects cannot, unless you opt in.

## Layout

```
src/evidenceorbit/     library
  orbit.py             sidecar facade
  detect.py            hybrid detector
  attribute.py         origin attribution
  recover.py           recovery planner
  graph.py             evidence graph
  checkpoint.py        in-memory and file stores
  audit.py             hash-chained log
  adapters.py          ReAct and LangGraph-style
  mcp_server.py        MCP stdio server
  cli.py               analyze, wrap, mcp, bench, schemas
  benchmarks/          30-case corpus and scorer
schemas/               JSON Schema, version 0.1.0
docs/                  architecture, schemas, adapters
examples/              in-process loop and offline trace
```

## Documentation

- [Architecture](docs/architecture.md)
- [Schemas](docs/schemas.md)
- [Adapters](docs/adapters.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Changelog](CHANGELOG.md)

## Out of scope for v0.1

Multi-agent recovery, learning a policy from past rewinds, distributed checkpoints, and a required UI. The library is the product. A console that replays this benchmark can sit beside it; it is not part of the runtime contract.

## Status

v0.1.0 is for early adopters who can emit the trajectory schema or use one of the two adapters. File an issue with the JSONL if a latent error is missed or a clean trace is flagged. That is the bug.
