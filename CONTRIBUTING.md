# Contributing

Thanks for working on EvidenceOrbit. The project exists to close one loop: an agent commits early, the trajectory already contains contradictory evidence, and nothing acts on it. Changes should stay in service of that loop.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest
python -m evidenceorbit bench
```

Python 3.11 or newer. The runtime dependency is Pydantic v2.

## What we expect in a change

- Types stay strict. `mypy src` and `ruff check src tests` pass.
- Detection and recovery behavior changes come with a unit test and, if the failure mode is new, a benchmark case in `src/evidenceorbit/benchmarks/corpus.py`.
- Benchmark recall, attribution accuracy, and recovery success do not drop, and the false-positive rate does not rise, unless the PR explains why a case was wrong.
- New non-determinism (a model call, a random draw, a clock) is injected and written to the audit log with model id, prompt hash, and temperature.
- Default behavior stays propose-only. Do not auto-apply rewinds, and do not auto-rewind destructive tools.

## Public API

Stable for 0.1.x:

- symbols exported from `evidenceorbit`
- JSON schemas produced by `python -m evidenceorbit schemas`
- CLI commands and MCP tool names

Breaking those requires 0.2.0 and a changelog entry. Additive optional fields are fine in a patch or minor.

## Design bias

Prefer a composable function over a framework. The detector must stay cheap until a candidate contradiction exists. Every detection needs the event ids it used, a confidence, and a sentence a human can check. Recovery rewinds to just before the origin, with the unused evidence quoted back to the agent. Full restarts are a failure of the planner.

## Reporting bugs

Include the trajectory JSONL and the report (`evidenceorbit analyze trace.jsonl`). A missed latent error is a bug. A clean trajectory flagged as an error is a bug.
