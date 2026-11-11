"""EvidenceOrbit sidecar.

`analyze` is the pure offline entry point. `ingest` is the live path: it
folds one event into the graph, reruns the fast detector, and appends an
audit record only when the set of active latent errors changes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from evidenceorbit.attribute import attribute
from evidenceorbit.audit import AuditLog, body_of
from evidenceorbit.canonical import content_hash
from evidenceorbit.checkpoint import (
    CheckpointAdapter,
    InMemoryCheckpointStore,
    checkpoint_from_event,
)
from evidenceorbit.detect import Detector
from evidenceorbit.graph import EvidenceGraph
from evidenceorbit.judge import Judge
from evidenceorbit.models import (
    AnalysisReport,
    Detection,
    EventKind,
    RecoveryPolicy,
    RecoveryProposal,
    RecoveryResult,
    TrajectoryEvent,
)
from evidenceorbit.recover import plan_recovery
from evidenceorbit.version import SCHEMA_VERSION


def coerce_event(event: TrajectoryEvent | dict[str, object]) -> TrajectoryEvent:
    if isinstance(event, TrajectoryEvent):
        parsed = event
    else:
        parsed = TrajectoryEvent.model_validate(event)
    if parsed.content_hash is not None:
        return parsed
    digest = content_hash(parsed.model_dump(mode="json", exclude={"content_hash"}))
    return parsed.model_copy(update={"content_hash": digest})


class EvidenceOrbit:
    """Framework-agnostic latent-error sidecar.

    Public API (stable for 0.1.x): this class, `analyze`, `ingest`,
    `propose_recovery`, `apply_recovery`, and the pydantic models in
    `evidenceorbit.models`.
    """

    def __init__(
        self,
        *,
        policy: RecoveryPolicy | None = None,
        judge: Judge | None = None,
        checkpoints: CheckpointAdapter | None = None,
        audit: AuditLog | None = None,
        audit_path: str | Path | None = None,
        clock: Callable[[], str] | None = None,
        enable_soft_path: bool = True,
    ) -> None:
        self.policy = policy or RecoveryPolicy()
        self.judge = judge
        self.checkpoints = checkpoints or InMemoryCheckpointStore()
        if audit is not None:
            self.audit = audit
        else:
            self.audit = AuditLog(audit_path, clock=clock)
        self.detector = Detector(judge=judge, enable_soft_path=enable_soft_path)
        self.events: list[TrajectoryEvent] = []
        self.graph = EvidenceGraph()
        self.report = AnalysisReport(
            session_id="",
            event_count=0,
            detections=[],
            attributions=[],
            proposals=[],
            graph=self.graph.snapshot(),
            audit_head=None,
        )
        self._seen_detect: set[str] = set()
        self._active_ids: set[str] = set()

    def analyze(self, events: Sequence[TrajectoryEvent | dict[str, object]]) -> AnalysisReport:
        """Offline, stateless analysis of a finished trajectory."""
        shadow = EvidenceOrbit(
            policy=self.policy,
            judge=self.judge,
            checkpoints=InMemoryCheckpointStore(),
            audit=AuditLog(clock=self.audit.clock),
            enable_soft_path=self.detector.enable_soft_path,
        )
        for event in events:
            shadow.ingest(event)
        return shadow.report

    def ingest(self, event: TrajectoryEvent | dict[str, object]) -> list[Detection]:
        """Push one event. Returns detections that newly became active."""
        parsed = coerce_event(event)
        if any(existing.id == parsed.id for existing in self.events):
            raise ValueError(f"Duplicate event id {parsed.id}")
        checkpoint = checkpoint_from_event(parsed)
        if checkpoint is not None:
            self.checkpoints.save(checkpoint)
        before = set(self._active_ids)
        self.events.append(parsed)
        self._recompute()
        return [item for item in self.report.detections if item.id not in before]

    def ingest_many(self, events: Sequence[TrajectoryEvent | dict[str, object]]) -> AnalysisReport:
        for event in events:
            self.ingest(event)
        return self.report

    def latent_errors(self) -> list[Detection]:
        return list(self.report.detections)

    def unused_counterevidence(self) -> list[Detection]:
        return [item for item in self.report.detections if item.type == "unused_counterevidence"]

    def propose_recovery(self, detection_id: str) -> RecoveryProposal:
        found = self.report.proposal_for(detection_id)
        if found is None:
            raise KeyError(f"No active detection {detection_id}")
        return found

    def apply_recovery(self, proposal_id: str, *, confirm: bool = False) -> RecoveryResult:
        """Apply a surgical rewind. Refuses unless confirm=True or policy auto allows it.

        Destructive external effects are never rewound unless
        `policy.allow_destructive` is set. Default policy is propose-only.
        """
        proposal = next((item for item in self.report.proposals if item.id == proposal_id), None)
        if proposal is None:
            raise KeyError(f"Unknown proposal {proposal_id}")
        if proposal.restore_scope == "external_blocked" and not self.policy.allow_destructive:
            reason = proposal.blocked_reason or "destructive side-effects are not auto-rewindable"
            self.audit.append(
                "recovery_refused",
                proposal.session_id,
                {"proposal_id": proposal_id, "reason": reason},
            )
            self.report = self.report.model_copy(update={"audit_head": self.audit.head()})
            return RecoveryResult(proposal_id=proposal_id, applied=False, reason=reason)
        allowed = bool(confirm) or (self.policy.mode == "auto" and proposal.auto_executable)
        if not allowed:
            reason = "propose_only; pass confirm=True to apply a non-destructive rewind"
            self.audit.append(
                "recovery_refused",
                proposal.session_id,
                {"proposal_id": proposal_id, "reason": reason},
            )
            self.report = self.report.model_copy(update={"audit_head": self.audit.head()})
            return RecoveryResult(proposal_id=proposal_id, applied=False, reason=reason)
        restored: str | None = None
        if proposal.checkpoint_id is not None:
            restored = self.checkpoints.load(proposal.checkpoint_id).id
        self.events = [event for event in self.events if event.seq < proposal.rewind_seq]
        injected = TrajectoryEvent(
            schema_version=SCHEMA_VERSION,
            id=f"corr_{proposal.id}",
            seq=proposal.rewind_seq,
            ts=self.audit.clock(),
            session_id=proposal.session_id,
            kind=EventKind.SYSTEM,
            actor="evidenceorbit",
            parent_ids=[],
            text=(
                proposal.corrective_signal.instruction
                + "\n"
                + "\n".join(proposal.corrective_signal.quotes)
            ),
            payload={
                "recovery_id": proposal.id,
                "do_not_repeat": proposal.corrective_signal.do_not_repeat,
                "evidence_event_ids": list(proposal.corrective_signal.evidence_event_ids),
            },
        )
        self.events.append(coerce_event(injected))
        self._recompute()
        self.audit.append(
            "recovery_applied",
            proposal.session_id,
            {
                "proposal_id": proposal.id,
                "restored_checkpoint_id": restored,
                "injected_event_id": injected.id,
                "truncated_to_seq": proposal.rewind_seq,
            },
        )
        self.report = self.report.model_copy(update={"audit_head": self.audit.head()})
        return RecoveryResult(
            proposal_id=proposal.id,
            applied=True,
            reason="rewound beliefs to before origin and injected corrective evidence",
            restored_checkpoint_id=restored,
            injected_event_id=injected.id,
            truncated_to_seq=proposal.rewind_seq,
        )

    def _recompute(self) -> None:
        sessions = {event.session_id for event in self.events}
        if len(sessions) > 1:
            raise ValueError(f"EvidenceOrbit instance holds one session, found {sorted(sessions)}")
        session_id = next(iter(sessions)) if sessions else ""
        self.graph = EvidenceGraph()
        ordered = sorted(self.events, key=lambda event: (event.seq, event.id))
        self.events = ordered
        for event in ordered:
            self.graph.add_event(event)
        self.graph.add_structural(ordered)
        detections, _edges = self.detector.run(ordered, self.graph)
        attributions = [attribute(detection, ordered) for detection in detections]
        proposals = [
            plan_recovery(
                detection,
                attr,
                ordered,
                policy=self.policy,
                checkpoints=self.checkpoints,
            )
            for detection, attr in zip(detections, attributions, strict=True)
        ]
        active = {detection.id for detection in detections}
        for detection, attr, proposal in zip(detections, attributions, proposals, strict=True):
            if detection.id in self._seen_detect:
                continue
            self.audit.append("detection", session_id, body_of(detection))
            if detection.judge is not None:
                self.audit.append("judge", session_id, body_of(detection.judge))
            self.audit.append("attribution", session_id, body_of(attr))
            self.audit.append("recovery_proposal", session_id, body_of(proposal))
            self._seen_detect.add(detection.id)
        for gone in sorted(self._active_ids - active):
            self.audit.append("resolution", session_id, {"detection_id": gone})
        self._active_ids = active
        self.report = AnalysisReport(
            session_id=session_id,
            event_count=len(ordered),
            detections=detections,
            attributions=attributions,
            proposals=proposals,
            graph=self.graph.snapshot(),
            audit_head=self.audit.head(),
        )
