"""Surgical recovery planner.

Rewind target is the state just before the origin event — the last checkpoint
with seq < origin, when one exists. The corrective signal quotes the unused
evidence and names the claim that must not be repeated.

Auto-execution is off unless policy.mode == 'auto'. Destructive tool effects
inside the rewind window force restore_scope=external_blocked and refuse
auto-execution unless allow_destructive is set. Propose-only is the default.
"""

from __future__ import annotations

import re
from typing import Literal

from evidenceorbit.checkpoint import CheckpointAdapter
from evidenceorbit.models import (
    Attribution,
    CorrectiveSignal,
    Detection,
    EventKind,
    RecoveryPolicy,
    RecoveryProposal,
    TrajectoryEvent,
)

_DESTRUCTIVE = re.compile(
    r"send|delete|remove|drop|write|charge|transfer|deploy|post|put|exec|mail|pay|refund|publish",
    re.IGNORECASE,
)


def is_destructive(event: TrajectoryEvent) -> bool:
    if event.kind not in {EventKind.TOOL_CALL, EventKind.TOOL_RESULT}:
        return False
    side = event.payload.get("side_effect")
    if side in {"write", "external", "destructive"}:
        return True
    if side in {"read", "none"}:
        return False
    name = str(event.payload.get("tool_name") or event.payload.get("name") or "")
    return bool(_DESTRUCTIVE.search(name))


def plan_recovery(
    detection: Detection,
    attribution: Attribution,
    events: list[TrajectoryEvent],
    *,
    policy: RecoveryPolicy,
    checkpoints: CheckpointAdapter,
) -> RecoveryProposal:
    by_id = {event.id: event for event in events}
    claim = by_id[detection.claim_event_id]
    window = [event for event in events if attribution.origin_seq <= event.seq]
    destructive = [event.id for event in window if is_destructive(event)]
    checkpoint = checkpoints.latest_before(detection.session_id, attribution.origin_seq)
    quotes: list[str] = []
    for evidence_id in detection.evidence_event_ids:
        evidence = by_id.get(evidence_id)
        if evidence is None:
            continue
        text = evidence.text.strip() or json_brief(evidence)
        quotes.append(f"{evidence.id}: {text}")
    do_not_repeat = claim.text.strip() or (
        f"{detection.claim_fact.subject}.{detection.claim_fact.predicate}="
        f"{detection.claim_fact.value}"
        if detection.claim_fact
        else claim.id
    )
    instruction = (
        f"Rewind to before {attribution.origin_event_id} (seq {attribution.origin_seq}). "
        f"Re-reason with the contradictory evidence below. "
        f"Do not restate the commitment from {claim.id}."
    )
    scope: Literal["partial_belief", "full_state", "external_blocked"]
    if destructive:
        scope = "external_blocked"
        blocked = (
            "Destructive or external side-effects exist at or after the origin "
            f"({', '.join(destructive)}). Beliefs can be rewound; external effects cannot "
            "without allow_destructive."
        )
        auto = policy.mode == "auto" and policy.allow_destructive
    else:
        scope = "partial_belief" if checkpoint is None or not checkpoint.environment else "full_state"
        if checkpoint is None:
            scope = "partial_belief"
        blocked = None
        auto = policy.mode == "auto"
    if policy.mode != "auto":
        auto = False
    return RecoveryProposal(
        id="rec_" + detection.id.removeprefix("det_"),
        detection_id=detection.id,
        session_id=detection.session_id,
        origin_event_id=attribution.origin_event_id,
        rewind_seq=attribution.origin_seq,
        checkpoint_id=None if checkpoint is None else checkpoint.id,
        restore_scope=scope,
        corrective_signal=CorrectiveSignal(
            instruction=instruction,
            evidence_event_ids=list(detection.evidence_event_ids),
            quotes=quotes,
            do_not_repeat=do_not_repeat,
        ),
        auto_executable=auto,
        blocked_reason=blocked,
        destructive_event_ids=destructive,
        mode=policy.mode,
    )


def json_brief(event: TrajectoryEvent) -> str:
    subject = event.payload.get("subject")
    predicate = event.payload.get("predicate")
    value = event.payload.get("value")
    if subject is not None and value is not None:
        return f"{subject}.{predicate}={value}"
    return event.kind.value
