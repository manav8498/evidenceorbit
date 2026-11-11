# Schemas

Schema version `0.1.0`. Machine-readable JSON Schema lives in [`schemas/`](../schemas) and is regenerated with:

```bash
python -m evidenceorbit schemas --out schemas
```

Models are Pydantic v2. A TypeScript port should match these fields. Unknown top-level fields are rejected. `payload` is open so tools can attach their own structure, but the keys below are the ones the detector reads.

## TrajectoryEvent

| Field | Required | Notes |
| --- | --- | --- |
| `schema_version` | no | default `0.1.0` |
| `id` | yes | stable, referenced by parents, citations, and reports |
| `seq` | yes | monotonic integer, the order the detector uses |
| `ts` | yes | ISO-8601 string |
| `session_id` | yes | one sidecar instance holds one session |
| `kind` | yes | `llm_call`, `tool_call`, `tool_result`, `observation`, `claim`, `goal`, `contract`, `human`, `checkpoint`, `system` |
| `actor` | no | default `agent` |
| `parent_ids` | no | causal parents. The producing LLM call should parent the claim it emitted |
| `text` | no | natural-language form. Used when structured facts are absent |
| `payload` | no | see below |
| `content_hash` | no | filled by the sidecar if omitted |

### Payload keys the detector understands

Claims, tool results, observations, human messages:

- `subject`, `predicate`, `value`, `polarity` (`assert` or `deny`)
- `facts`: list of additional `{subject, predicate, value, polarity}`
- `citations`: event ids the claim used
- `supersedes`: event ids this claim retires
- `status`: `active` (default), `retracted`, `superseded`

Tool calls and results:

- `tool_name`, `call_id`, `arguments`, `ok`
- `side_effect`: `read`, `none`, `write`, `external`, `destructive`

Contracts and goals:

- `constraints`: list of `{id, modality, text, subject, predicate, value}`
- `modality` is `must` or `must_not`

LLM calls:

- `model`, `temperature`, `prompt_hash`, `prompt_tokens`, `completion_tokens`, `response`

Checkpoints:

- `checkpoint_id`, `label`, `agent_context`, `environment`

## Detection

`type` is one of `unused_counterevidence`, `direct_contradiction`, `contract_violation`, `internal_inconsistency`, `compounded_error`.

`confidence` is in `[0, 1]`. Structured conflicts are `0.93`. Text facts are `0.84`. Contract violations are `0.95`. Soft-path judge verdicts carry their own confidence plus a `judge` object: `model`, `prompt_hash`, `temperature`, `contradicts`, `confidence`, `explanation`.

`claim_event_id` and `evidence_event_ids` are the only evidence the explanation is allowed to rely on.

## Attribution

`origin_event_id` is where the bad commitment entered. `propagation_path` is the causal chain from that event through to the flagged claim. `unused_evidence_ids` omits evidence the claim already cited.

## RecoveryProposal

`rewind_seq` equals the origin sequence. `checkpoint_id` is the latest checkpoint strictly before that sequence, if any. `restore_scope` is `partial_belief`, `full_state`, or `external_blocked`. `corrective_signal` contains the instruction, quotes, and `do_not_repeat`. `auto_executable` is false unless policy says otherwise and the window is safe.

## AuditRecord

Append-only. `content_hash` is SHA-256 of the canonical JSON of `schema_version`, `record_type`, `session_id`, `ts`, `prev_hash`, and `body`. `prev_hash` of the first record is 64 zeroes. `AuditLog.verify()` recomputes the chain.

## AnalysisReport

The offline and live result: detections, attributions, proposals, the graph snapshot, and the audit head.
