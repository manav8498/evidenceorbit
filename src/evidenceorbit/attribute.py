"""Origin attribution.

The origin is the earliest event where the bad commitment entered:
the earliest claim that already conflicts with the evidence, or the LLM
call that directly produced that claim. Ancestors further upstream that
never asserted the bad fact are not blamed.
"""

from __future__ import annotations

from evidenceorbit.facts import facts_from_event, first_contradiction
from evidenceorbit.models import Attribution, Detection, EventKind, TrajectoryEvent


def _chain(event: TrajectoryEvent, by_id: dict[str, TrajectoryEvent]) -> list[TrajectoryEvent]:
    """Self first, then ancestors, each once."""
    out = [event]
    stack = list(event.parent_ids)
    seen = {event.id}
    while stack:
        current = stack.pop()
        if current in seen or current not in by_id:
            continue
        seen.add(current)
        node = by_id[current]
        out.append(node)
        stack.extend(node.parent_ids)
    return out


def _producer(holder: TrajectoryEvent, by_id: dict[str, TrajectoryEvent]) -> TrajectoryEvent:
    producers = []
    for parent_id in holder.parent_ids:
        parent = by_id.get(parent_id)
        if parent is not None and parent.kind == EventKind.LLM_CALL:
            producers.append(parent)
    if not producers:
        return holder
    return min(producers, key=lambda item: (item.seq, item.id))


def origin_event(
    claim: TrajectoryEvent,
    evidence_ids: list[str],
    events: list[TrajectoryEvent],
) -> tuple[TrajectoryEvent, list[TrajectoryEvent]]:
    by_id = {event.id: event for event in events}
    evidence_facts = []
    for evidence_id in evidence_ids:
        evidence = by_id.get(evidence_id)
        if evidence is not None and evidence.kind == EventKind.CLAIM:
            evidence_facts.extend(facts_from_event(evidence))
        elif evidence is not None:
            evidence_facts.extend(facts_from_event(evidence))
    chain = _chain(claim, by_id)
    holders = [claim]
    for node in chain[1:]:
        if node.kind != EventKind.CLAIM:
            continue
        if evidence_facts and first_contradiction(facts_from_event(node), evidence_facts):
            holders.append(node)
        elif not evidence_facts and node.kind == EventKind.CLAIM:
            # Contract violations may not have an evidence fact on a tool.
            if first_contradiction(facts_from_event(node), facts_from_event(claim)):
                holders.append(node)
            elif facts_share_field(node, claim):
                holders.append(node)
    earliest = min(holders, key=lambda item: (item.seq, item.id))
    origin = _producer(earliest, by_id)
    path_ids = {origin.id, claim.id, earliest.id}
    path_ids.update(node.id for node in chain if node.kind in {EventKind.CLAIM, EventKind.LLM_CALL})
    # Keep the causal chain between origin and the claim, inclusive.
    path = [
        node
        for node in chain
        if node.id in path_ids and origin.seq <= node.seq <= claim.seq
    ]
    path.append(origin)
    path.append(claim)
    unique: dict[str, TrajectoryEvent] = {}
    for node in path:
        unique[node.id] = node
    ordered = sorted(unique.values(), key=lambda item: (item.seq, item.id))
    return origin, ordered


def facts_share_field(left: TrajectoryEvent, right: TrajectoryEvent) -> bool:
    for a in facts_from_event(left):
        for b in facts_from_event(right):
            if a.subject == b.subject and a.predicate == b.predicate:
                return True
    return False


def attribute(detection: Detection, events: list[TrajectoryEvent]) -> Attribution:
    by_id = {event.id: event for event in events}
    claim = by_id[detection.claim_event_id]
    if detection.type == "compounded_error":
        # Blame the upstream commitment, not the downstream echo.
        upstream = _upstream_claim(claim, detection, by_id)
        origin, path = origin_event(upstream, detection.evidence_event_ids, events)
        # Propagation still includes the downstream claim.
        if all(node.id != claim.id for node in path):
            path = sorted([*path, claim], key=lambda item: (item.seq, item.id))
    else:
        origin, path = origin_event(claim, detection.evidence_event_ids, events)
    unused = [
        evidence_id
        for evidence_id in detection.evidence_event_ids
        if evidence_id not in set(claim.payload.get("citations") or [])
    ]
    path_ids = [node.id for node in path]
    summary = (
        f"Bad commitment entered at {origin.id} ({origin.kind.value}, seq {origin.seq}) "
        f"and propagated via {' -> '.join(path_ids)} to claim {claim.id}. "
        f"Counter-evidence: {', '.join(detection.evidence_event_ids)}."
    )
    return Attribution(
        id="attr_" + detection.id.removeprefix("det_"),
        detection_id=detection.id,
        origin_event_id=origin.id,
        origin_seq=origin.seq,
        propagation_path=path_ids,
        propagation_summary=summary,
        unused_evidence_ids=unused,
    )


def _upstream_claim(
    claim: TrajectoryEvent,
    detection: Detection,
    by_id: dict[str, TrajectoryEvent],
) -> TrajectoryEvent:
    """Earliest ancestor claim that itself conflicts with the evidence, else parent claim, else self."""
    evidence_facts = []
    for evidence_id in detection.evidence_event_ids:
        node = by_id.get(evidence_id)
        if node is not None:
            evidence_facts.extend(facts_from_event(node))
    holders: list[TrajectoryEvent] = []
    for node_id in _ancestor_claim_ids(claim, by_id):
        node = by_id[node_id]
        if evidence_facts and first_contradiction(facts_from_event(node), evidence_facts):
            holders.append(node)
    if holders:
        return min(holders, key=lambda item: (item.seq, item.id))
    for parent_id in claim.parent_ids:
        parent = by_id.get(parent_id)
        if parent is not None and parent.kind == EventKind.CLAIM:
            return parent
    return claim


def _ancestor_claim_ids(event: TrajectoryEvent, by_id: dict[str, TrajectoryEvent]) -> list[str]:
    out: list[str] = []
    stack = list(event.parent_ids)
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen or current not in by_id:
            continue
        seen.add(current)
        node = by_id[current]
        if node.kind == EventKind.CLAIM:
            out.append(node.id)
        stack.extend(node.parent_ids)
    return out
