"""Hybrid latent-error detector.

Fast path: structured fact conflicts, antonym/numeric text conflicts, and
explicit contract violations. These return immediately with high confidence.

Slow path: hashing-embedder candidates whose polarity differs. A judge
(heuristic by default, LLM if injected) must confirm before a detection is
emitted. Happy-path steps that never share a conflicting fact pay only the
fact-index comparison.
"""

from __future__ import annotations

from evidenceorbit.canonical import content_hash
from evidenceorbit.embed import Embedder, HashingEmbedder, cosine
from evidenceorbit.facts import (
    constraint_facts,
    facts_agree,
    facts_from_event,
    first_contradiction,
)
from evidenceorbit.graph import EvidenceGraph
from evidenceorbit.judge import HeuristicJudge, Judge, text_is_negative
from evidenceorbit.models import (
    CONSTRAINT_KINDS,
    EVIDENCE_KINDS,
    Constraint,
    Detection,
    DetectionType,
    EdgeKind,
    EventKind,
    Fact,
    GraphEdge,
    JudgeTrace,
    TrajectoryEvent,
)

# Fast-path accept threshold. Soft candidates sit under this and go to the judge.
_HARD_CONFIDENCE = 0.93
_TEXT_CONFIDENCE = 0.84
_SOFT_CONFIDENCE = 0.5
_SOFT_COSINE = 0.55


def _detection_id(
    session_id: str,
    type_: DetectionType,
    claim_id: str,
    evidence_ids: list[str],
    constraint_id: str | None = None,
) -> str:
    digest = content_hash(
        {
            "session": session_id,
            "type": type_,
            "claim": claim_id,
            "evidence": evidence_ids,
            "constraint": constraint_id,
        }
    )
    return "det_" + digest[:16]


def _citations(event: TrajectoryEvent) -> set[str]:
    raw = event.payload.get("citations") or []
    if not isinstance(raw, list):
        return set()
    return {str(item) for item in raw}


def _inactive_claims(events: list[TrajectoryEvent]) -> set[str]:
    inactive: set[str] = set()
    for event in events:
        if event.kind != EventKind.CLAIM:
            continue
        status = str(event.payload.get("status") or "active")
        if status in {"retracted", "superseded"}:
            inactive.add(event.id)
        supersedes = event.payload.get("supersedes") or []
        if isinstance(supersedes, list):
            inactive.update(str(item) for item in supersedes)
    return inactive


def _constraints_of(event: TrajectoryEvent) -> list[Constraint]:
    raw = event.payload.get("constraints") or []
    if not isinstance(raw, list):
        return []
    out: list[Constraint] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(Constraint.model_validate(item))
    return out


class Detector:
    def __init__(
        self,
        *,
        judge: Judge | None = None,
        embedder: Embedder | None = None,
        enable_soft_path: bool = True,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.judge = judge or HeuristicJudge(embedder=self.embedder)
        self.enable_soft_path = enable_soft_path

    def run(
        self, events: list[TrajectoryEvent], graph: EvidenceGraph
    ) -> tuple[list[Detection], list[GraphEdge]]:
        ordered = sorted(events, key=lambda event: (event.seq, event.id))
        fact_index = {event.id: facts_from_event(event) for event in ordered}
        inactive = _inactive_claims(ordered)
        claims = [event for event in ordered if event.kind == EventKind.CLAIM and event.id not in inactive]
        evidences = [event for event in ordered if event.kind in EVIDENCE_KINDS]
        inactive |= self._incorporated(claims, evidences, fact_index)
        claims = [event for event in claims if event.id not in inactive]

        detections: list[Detection] = []
        edges: list[GraphEdge] = []
        seen: set[tuple[str, str, str]] = set()

        for claim in claims:
            claim_facts = fact_index[claim.id]
            cited = _citations(claim)
            for evidence in evidences:
                if evidence.id == claim.id:
                    continue
                pair = first_contradiction(claim_facts, fact_index[evidence.id])
                if pair is None:
                    if facts_agree(claim_facts, fact_index[evidence.id]) and evidence.id in cited:
                        edges.append(
                            GraphEdge(
                                source=evidence.id,
                                target=claim.id,
                                kind=EdgeKind.SUPPORTS,
                                confidence=0.9,
                                detail="cited agreement",
                            )
                        )
                    elif self.enable_soft_path and not claim_facts and not fact_index[evidence.id]:
                        soft = self._soft(claim, evidence)
                        if soft is not None:
                            detections.append(soft)
                            edges.append(
                                GraphEdge(
                                    source=evidence.id,
                                    target=claim.id,
                                    kind=EdgeKind.CONTRADICTS,
                                    confidence=soft.confidence,
                                    detail=soft.type,
                                )
                            )
                    continue
                claim_fact, evidence_fact = pair
                unused = evidence.id not in cited
                if evidence.seq >= claim.seq and not unused:
                    dtype: DetectionType = "direct_contradiction"
                elif not unused and evidence.seq < claim.seq:
                    # Cited the counter-evidence and still asserted the bad value.
                    dtype = "direct_contradiction"
                else:
                    dtype = "unused_counterevidence"
                key = (dtype, claim.id, evidence.id)
                if key in seen:
                    continue
                seen.add(key)
                confidence = _HARD_CONFIDENCE
                if claim_fact.source == "text" and evidence_fact.source == "text":
                    confidence = _TEXT_CONFIDENCE
                detection = self._hard(
                    claim, evidence, dtype, confidence, claim_fact, evidence_fact, unused
                )
                detections.append(detection)
                edges.append(
                    GraphEdge(
                        source=evidence.id,
                        target=claim.id,
                        kind=EdgeKind.CONTRADICTS,
                        confidence=confidence,
                        detail=dtype,
                    )
                )

        detections.extend(self._internal(claims, fact_index, seen))
        for detection in detections:
            if detection.type != "internal_inconsistency":
                continue
            for evidence_id in detection.evidence_event_ids:
                edges.append(
                    GraphEdge(
                        source=evidence_id,
                        target=detection.claim_event_id,
                        kind=EdgeKind.CONTRADICTS,
                        confidence=detection.confidence,
                        detail="internal_inconsistency",
                    )
                )

        contract_detections, contract_edges = self._contracts(ordered, claims, fact_index)
        detections.extend(contract_detections)
        edges.extend(contract_edges)

        compounded, compounded_edges = self._compounded(ordered, claims, detections, fact_index, inactive)
        detections.extend(compounded)
        edges.extend(compounded_edges)

        detections.sort(key=lambda item: (item.claim_event_id, item.type, item.evidence_event_ids))
        graph.set_semantic(edges)
        return detections, edges

    def _hard(
        self,
        claim: TrajectoryEvent,
        evidence: TrajectoryEvent,
        type_: DetectionType,
        confidence: float,
        claim_fact: Fact,
        evidence_fact: Fact,
        unused: bool,
    ) -> Detection:
        relation = "is not cited by the claim" if unused else "was cited but not adopted"
        explanation = (
            f"Claim {claim.id} asserts {claim_fact.subject}.{claim_fact.predicate}"
            f"={claim_fact.value} ({claim_fact.polarity}), but {evidence.kind.value} "
            f"{evidence.id} states {evidence_fact.subject}.{evidence_fact.predicate}"
            f"={evidence_fact.value} ({evidence_fact.polarity}). The evidence {relation}."
        )
        dtype = type_
        return Detection(
            id=_detection_id(claim.session_id, type_, claim.id, [evidence.id]),
            session_id=claim.session_id,
            type=dtype,
            confidence=confidence,
            claim_event_id=claim.id,
            evidence_event_ids=[evidence.id],
            explanation=explanation,
            claim_fact=claim_fact,
            evidence_fact=evidence_fact,
        )

    def _soft(self, claim: TrajectoryEvent, evidence: TrajectoryEvent) -> Detection | None:
        if not claim.text or not evidence.text:
            return None
        if text_is_negative(claim.text) == text_is_negative(evidence.text):
            return None
        similarity = cosine(self.embedder.embed(claim.text), self.embedder.embed(evidence.text))
        if similarity < _SOFT_COSINE:
            return None
        verdict: JudgeTrace = self.judge.judge(claim.text, evidence.text)
        if not verdict.contradicts:
            return None
        unused = evidence.id not in _citations(claim)
        type_: DetectionType = "unused_counterevidence" if unused else "direct_contradiction"
        return Detection(
            id=_detection_id(claim.session_id, type_, claim.id, [evidence.id]),
            session_id=claim.session_id,
            type=type_,
            confidence=verdict.confidence,
            claim_event_id=claim.id,
            evidence_event_ids=[evidence.id],
            explanation=verdict.explanation,
            judge=verdict,
        )

    def _internal(
        self,
        claims: list[TrajectoryEvent],
        fact_index: dict[str, list[Fact]],
        seen: set[tuple[str, str, str]],
    ) -> list[Detection]:
        found: list[Detection] = []
        for index, earlier in enumerate(claims):
            for later in claims[index + 1 :]:
                pair = first_contradiction(fact_index[earlier.id], fact_index[later.id])
                if pair is None:
                    continue
                key = ("internal_inconsistency", earlier.id, later.id)
                if key in seen:
                    continue
                seen.add(key)
                claim_fact, evidence_fact = pair
                found.append(
                    Detection(
                        id=_detection_id(
                            earlier.session_id, "internal_inconsistency", earlier.id, [later.id]
                        ),
                        session_id=earlier.session_id,
                        type="internal_inconsistency",
                        confidence=_HARD_CONFIDENCE,
                        claim_event_id=earlier.id,
                        evidence_event_ids=[later.id],
                        explanation=(
                            f"Active claims {earlier.id} and {later.id} disagree on "
                            f"{claim_fact.subject}.{claim_fact.predicate}: "
                            f"{claim_fact.value} vs {evidence_fact.value}. Neither was retracted."
                        ),
                        claim_fact=claim_fact,
                        evidence_fact=evidence_fact,
                    )
                )
        return found

    def _contracts(
        self,
        events: list[TrajectoryEvent],
        claims: list[TrajectoryEvent],
        fact_index: dict[str, list[Fact]],
    ) -> tuple[list[Detection], list[GraphEdge]]:
        detections: list[Detection] = []
        edges: list[GraphEdge] = []
        for event in events:
            if event.kind not in CONSTRAINT_KINDS:
                continue
            for constraint in _constraints_of(event):
                required = constraint_facts(constraint.subject, constraint.predicate, constraint.value)
                if required is None:
                    continue
                for claim in claims:
                    if claim.seq < event.seq and event.kind == EventKind.CONTRACT:
                        # A contract can be stated up front; claims after it matter.
                        pass
                    violation = _violates(constraint.modality, required, fact_index[claim.id])
                    if violation is None:
                        continue
                    detection = Detection(
                        id=_detection_id(
                            claim.session_id,
                            "contract_violation",
                            claim.id,
                            [event.id],
                            constraint.id,
                        ),
                        session_id=claim.session_id,
                        type="contract_violation",
                        confidence=0.95,
                        claim_event_id=claim.id,
                        evidence_event_ids=[event.id],
                        explanation=(
                            f"Claim {claim.id} asserts {violation.subject}.{violation.predicate}"
                            f"={violation.value}, which violates constraint {constraint.id} "
                            f"({constraint.modality}: {constraint.text})."
                        ),
                        claim_fact=violation,
                        evidence_fact=required,
                        constraint_id=constraint.id,
                    )
                    detections.append(detection)
                    edges.append(
                        GraphEdge(
                            source=claim.id,
                            target=event.id,
                            kind=EdgeKind.VIOLATES,
                            confidence=0.95,
                            detail=constraint.id,
                        )
                    )
        return detections, edges

    def _incorporated(
        self,
        claims: list[TrajectoryEvent],
        evidences: list[TrajectoryEvent],
        fact_index: dict[str, list[Fact]],
    ) -> set[str]:
        """Earlier bad claims drop out once a later claim cites the evidence and agrees with it."""
        resolved: set[str] = set()
        for evidence in evidences:
            for claim in claims:
                if evidence.id not in _citations(claim):
                    continue
                if not facts_agree(fact_index[claim.id], fact_index[evidence.id]):
                    continue
                for earlier in claims:
                    if earlier.seq >= claim.seq:
                        continue
                    if first_contradiction(fact_index[earlier.id], fact_index[evidence.id]):
                        resolved.add(earlier.id)
        return resolved

    def _compounded(
        self,
        events: list[TrajectoryEvent],
        claims: list[TrajectoryEvent],
        detections: list[Detection],
        fact_index: dict[str, list[Fact]],
        inactive: set[str],
    ) -> tuple[list[Detection], list[GraphEdge]]:
        by_id = {event.id: event for event in events}
        base: dict[str, list[Detection]] = {}
        for detection in detections:
            if detection.type == "compounded_error":
                continue
            base.setdefault(detection.claim_event_id, []).append(detection)
        if not base:
            return [], []
        out: list[Detection] = []
        edges: list[GraphEdge] = []
        already = {detection.claim_event_id for detection in detections}
        evidence_parties = {
            evidence_id for detection in detections for evidence_id in detection.evidence_event_ids
        }
        for claim in claims:
            if claim.id in already or claim.id in inactive or claim.id in evidence_parties:
                continue
            ancestors = _ancestor_ids(claim, by_id)
            upstream_ids = [item for item in ancestors if item in base]
            if not upstream_ids:
                continue
            # A correction that agrees with the contradicting evidence is not compounded.
            evidence_ids: list[str] = []
            for upstream in upstream_ids:
                for detection in base[upstream]:
                    for evidence_id in detection.evidence_event_ids:
                        if evidence_id not in evidence_ids:
                            evidence_ids.append(evidence_id)
            if _agrees_with_any(claim, evidence_ids, by_id, fact_index):
                continue
            primary = base[upstream_ids[0]][0]
            detection = Detection(
                id=_detection_id(claim.session_id, "compounded_error", claim.id, evidence_ids),
                session_id=claim.session_id,
                type="compounded_error",
                confidence=min(0.9, primary.confidence),
                claim_event_id=claim.id,
                evidence_event_ids=evidence_ids,
                explanation=(
                    f"Claim {claim.id} derives from contradicted claim {upstream_ids[0]} "
                    f"and was not corrected against evidence {', '.join(evidence_ids)}."
                ),
                claim_fact=primary.claim_fact,
                evidence_fact=primary.evidence_fact,
            )
            out.append(detection)
            for evidence_id in evidence_ids:
                edges.append(
                    GraphEdge(
                        source=evidence_id,
                        target=claim.id,
                        kind=EdgeKind.CONTRADICTS,
                        confidence=detection.confidence,
                        detail="compounded_error",
                    )
                )
        return out, edges


def _ancestor_ids(event: TrajectoryEvent, by_id: dict[str, TrajectoryEvent]) -> list[str]:
    out: list[str] = []
    stack = list(event.parent_ids)
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen or current not in by_id:
            continue
        seen.add(current)
        out.append(current)
        stack.extend(by_id[current].parent_ids)
    return out


def _agrees_with_any(
    claim: TrajectoryEvent,
    evidence_ids: list[str],
    by_id: dict[str, TrajectoryEvent],
    fact_index: dict[str, list[Fact]],
) -> bool:
    for evidence_id in evidence_ids:
        evidence = by_id.get(evidence_id)
        if evidence is None:
            continue
        if facts_agree(fact_index[claim.id], fact_index.get(evidence_id, [])):
            return True
    return False


def _violates(modality: str, required: Fact, claim_facts: list[Fact]) -> Fact | None:
    for fact in claim_facts:
        if fact.subject != required.subject or fact.predicate != required.predicate:
            continue
        same = fact.value == required.value and fact.polarity == "assert"
        denies = fact.value == required.value and fact.polarity == "deny"
        differs = fact.value != required.value and fact.polarity == "assert"
        if modality == "must_not" and same:
            return fact
        if modality == "must" and (differs or denies):
            return fact
    return None
