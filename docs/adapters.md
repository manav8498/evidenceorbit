# Adapters

EvidenceOrbit does not import LangGraph, CrewAI, MCP client SDKs, or Claude Code. Adapters translate those runtimes into `TrajectoryEvent`s. Two ship in v0.1. Both are push/pull examples you can copy.

## ReAct / tool-calling loop

Push mode. Your loop calls the adapter as it goes.

```python
from evidenceorbit import ReActAdapter

loop = ReActAdapter(session_id="demo")
loop.goal("Book a nonstop flight to SFO under 500 dollars.")
loop.llm("Find the fare.", "The SFO fare is 420 dollars.", model="your-model")
producer = loop.last_event_id()
loop.claim(
    "The SFO fare is 420 dollars.",
    parent_ids=[producer] if producer else None,
    subject="flight",
    predicate="price",
    value=420,
)
found = loop.tool(
    "search_flights",
    {"to": "SFO"},
    "SFO nonstop fare is 880 dollars.",
    payload={"subject": "flight", "predicate": "price", "value": 880},
)
```

`found` is the list of detections that became active on that call. `loop.orbit.report` is the full analysis. `loop.orbit.propose_recovery(detection_id)` and `apply_recovery(proposal_id, confirm=True)` are the live controls.

Parent the claim on the LLM call that produced it. Otherwise attribution correctly, but less usefully, stops on the claim itself.

## LangGraph-style nodes

Pull mode. The adapter reads `state["messages"]` before and after your node. It does not import LangGraph. Dict messages and attribute-style messages both work. Give messages stable `id`s so replays are not double-ingested.

```python
from evidenceorbit import LangGraphAdapter

adapter = LangGraphAdapter(session_id="graph")

def fare_node(state):
    messages = list(state["messages"])
    messages.append({
        "id": "m_claim",
        "role": "assistant",
        "content": "The fare is 420.",
        "subject": "flight",
        "predicate": "price",
        "value": 420,
    })
    return {"messages": messages}

wrapped = adapter.wrap_node(fare_node)
```

Mapping:

| Message role | Events |
| --- | --- |
| first `user` | `goal` |
| later `user` | `human` evidence |
| `system` with `constraints` | `contract` |
| `assistant` | `llm_call`, then a `claim` parented on it, plus `tool_call`s |
| `tool` | `tool_result` |

Copy `subject`, `predicate`, and `value` onto the message or into `payload`. Free text is parsed only as a fallback (`"the fare is 880"`), and structured fields are the supported contract.

To inject a recovery without applying it, read `adapter.orbit.report.proposals` and append the `corrective_signal.instruction` as a system message in your own state reducer. That keeps side effects in your graph, not in the sidecar.

## MCP

`python -m evidenceorbit mcp` speaks MCP over stdio (Content-Length framing, protocol `2024-11-05`).

| Tool | Gate |
| --- | --- |
| `check_trajectory` | none. Body is `{events: [...]}` |
| `get_latent_errors` | none |
| `propose_recovery` | none |
| `apply_recovery` | `confirm: true`. Still refuses destructive windows unless the server was constructed with `allow_destructive` |

Any MCP host can call `check_trajectory` on a rolling window. The Python SDK is the better integration if you are already in-process; MCP is the escape hatch for agents you do not link against.

## Checkpoints

Implement `save`, `load`, and `latest_before(session_id, seq)` and pass the object as `checkpoints=` to `EvidenceOrbit`. Checkpoint events in the trajectory (`kind: checkpoint`) are stored automatically when they carry `checkpoint_id`, `agent_context`, and `environment`.

A ProcessFork-style snapshotter maps naturally: `save` records the fork id, `latest_before` returns the newest snapshot whose cursor is strictly before the origin sequence.

## Wrapping a process

`evidenceorbit wrap -- your-agent` streams stdout through. Lines that are already trajectory JSON are ingested. Anything else becomes an `observation`. The summary is written to stderr when the process exits. This is a debug harness, not a substitute for an in-process adapter: observations without structured facts only trip the soft path.
