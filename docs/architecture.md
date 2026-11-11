# Architecture

EvidenceOrbit is a sidecar. It does not replace the agent loop. It watches a canonical trajectory, keeps an evidence graph, and emits three artifacts when something is wrong: a detection, an origin, and a surgical recovery proposal.

```mermaid
flowchart TD
  runtime[Agent runtime]
  inst[Instrumentation: push events or adapter pull]
  graph[Evidence graph]
  det[Hybrid detector]
  attr[Origin attributor]
  plan[Recovery planner]
  ckpt[Checkpoint adapter]
  audit[Hash-chained audit log]

  runtime --> inst --> graph --> det
  det --> attr --> plan
  plan --> ckpt
  det --> audit
  attr --> audit
  plan --> audit
```

## Why these pieces

Long-horizon failures are usually not mysterious. A claim is made. A tool result or observation later (or already) falsifies it. Downstream claims inherit the bad commitment. Post-hoc trace viewers show this after the run. EvidenceOrbit checks it on every step and names the rewind point.

The fast path is rule-based on normalized facts so a healthy step stays under a millisecond. A hashing embedder plus a judge runs only when text looks like a contradiction and no structured fact already settled it. The default judge is deterministic and logged. A real model is injected; the library never opens a network connection by itself.

## Fact model

A fact is `(subject, predicate, value, polarity)`.

- Structured `payload.subject` / `predicate` / `value` wins over free text.
- Two asserts of the same field contradict when at least one side is structured, or when both values are numbers, or when both values sit in the same antonym group (`success`/`failure`, `nonstop`/`connecting`, weekdays, and so on).
- `deny(X)` contradicts `assert(X)` only. `deny(success)` does not contradict `assert(failure)`; those agree.
- Goals and contracts are not claims. Contract checks are a separate pass over `must` / `must_not` constraints.

## Graph

| Edge | Meaning |
| --- | --- |
| `derives_from` | parent link, or tool result to its call |
| `contradicts` | evidence falsifies a claim |
| `supports` | claim cites evidence and the facts agree |
| `violates` | claim breaks a contract constraint |

Nodes are the events themselves. The graph is rebuilt with the event log on each ingest. At trajectory sizes of hundreds of steps this is still far under the 50 ms fast-path budget. The published microbench (80 unrelated facts) is about 0.8 ms mean, 1.9 ms p95, per ingest.

## Detector outputs

| Type | When |
| --- | --- |
| `unused_counterevidence` | an active claim conflicts with tool, observation, or human evidence that the claim does not cite |
| `direct_contradiction` | the claim cites the counter-evidence and still asserts the bad value |
| `contract_violation` | an active claim breaks a `must` or `must_not` constraint |
| `internal_inconsistency` | two active claims disagree, and neither was retracted |
| `compounded_error` | a later claim derives from a contradicted claim and does not itself adopt the corrective evidence |

A claim is inactive when `status` is `retracted` or `superseded`, when a later claim lists it in `supersedes`, or when a later claim cites the corrective evidence and agrees with it. Final-state analysis therefore stays quiet after a real repair. The audit log still records the detection and the resolution.

## Attribution

The origin is not "the first event in the session." It is the earliest claim that already holds the bad fact, or the LLM call that directly produced that claim. An earlier turn that is only a parent of the producer is not blamed. Compounded errors inherit the upstream origin, and the propagation path still includes the downstream claim.

## Recovery

The rewind sequence is the origin sequence: restore the last checkpoint with `seq < origin`, drop events at or after the origin, and inject a system event that quotes the unused evidence and names the claim that must not be repeated.

| `restore_scope` | Meaning |
| --- | --- |
| `partial_belief` | no checkpoint environment to restore; rewind beliefs and inject the signal |
| `full_state` | a checkpoint holds environment and nothing destructive happened after the origin |
| `external_blocked` | a write/external tool ran at or after the origin |

`auto_executable` is true only when policy mode is `auto` and the scope is not blocked. `apply_recovery` still requires `confirm=True` unless that auto policy is on. Destructive tools (`send`, `delete`, `deploy`, `charge`, or `side_effect=write`) are refused unless `allow_destructive=True`.

## Checkpoints and audit

`CheckpointAdapter` is three methods: `save`, `load`, `latest_before`. `InMemoryCheckpointStore` and `FileCheckpointStore` ship in-tree. A ProcessFork snapshotter or a LangGraph checkpointer implements the same three methods.

Every new detection, judge verdict, attribution, proposal, resolution, and recovery decision is appended to a hash chain. The hash covers the body, timestamp, session, record type, and previous hash.

## What v0.1 does not do

No multi-agent coordination recovery, no learned policy, no distributed checkpointing, no UI requirement inside the library. Those are explicit non-goals until the single-trajectory loop is boringly reliable.
