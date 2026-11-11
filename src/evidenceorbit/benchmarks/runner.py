"""Score the curated benchmark against a live EvidenceOrbit sidecar."""

from __future__ import annotations

from dataclasses import dataclass, field

from evidenceorbit.benchmarks.corpus import CASES, BenchCase
from evidenceorbit.models import AnalysisReport
from evidenceorbit.orbit import EvidenceOrbit


@dataclass
class CaseScore:
    case_id: str
    failure_mode: str
    expect_detection: bool
    detected: bool
    origin_ok: bool
    recovery_ok: bool
    detail: str = ""


@dataclass
class BenchResult:
    scores: list[CaseScore] = field(default_factory=list)

    @property
    def positives(self) -> list[CaseScore]:
        return [item for item in self.scores if item.expect_detection]

    @property
    def negatives(self) -> list[CaseScore]:
        return [item for item in self.scores if not item.expect_detection]

    @property
    def detection_recall(self) -> float:
        rows = self.positives
        if not rows:
            return 1.0
        return sum(1 for item in rows if item.detected) / len(rows)

    @property
    def attribution_accuracy(self) -> float:
        rows = self.positives
        if not rows:
            return 1.0
        return sum(1 for item in rows if item.origin_ok) / len(rows)

    @property
    def recovery_success(self) -> float:
        rows = self.positives
        if not rows:
            return 1.0
        return sum(1 for item in rows if item.recovery_ok) / len(rows)

    @property
    def false_positive_rate(self) -> float:
        rows = self.negatives
        if not rows:
            return 0.0
        return sum(1 for item in rows if item.detected) / len(rows)

    @property
    def ok(self) -> bool:
        return (
            self.detection_recall == 1.0
            and self.attribution_accuracy == 1.0
            and self.recovery_success == 1.0
            and self.false_positive_rate == 0.0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "cases": len(self.scores),
            "positives": len(self.positives),
            "negatives": len(self.negatives),
            "detection_recall": self.detection_recall,
            "attribution_accuracy": self.attribution_accuracy,
            "recovery_success": self.recovery_success,
            "false_positive_rate": self.false_positive_rate,
            "ok": self.ok,
            "failures": [
                {"id": item.case_id, "detail": item.detail}
                for item in self.scores
                if item.detail
            ],
        }

    def summary(self) -> str:
        data = self.as_dict()
        lines = [
            f"EvidenceOrbit benchmark  cases={data['cases']}  "
            f"recall={data['detection_recall']:.3f}  "
            f"attribution={data['attribution_accuracy']:.3f}  "
            f"recovery={data['recovery_success']:.3f}  "
            f"fpr={data['false_positive_rate']:.3f}",
        ]
        failures = data["failures"]
        if isinstance(failures, list):
            for item in failures:
                if isinstance(item, dict):
                    lines.append(f"  FAIL {item['id']}: {item['detail']}")
        return "\n".join(lines)


def score_case(case: BenchCase, report: AnalysisReport) -> CaseScore:
    if not case.expect_detection:
        if report.detections:
            kinds = sorted({item.type for item in report.detections})
            return CaseScore(
                case.id,
                case.failure_mode,
                False,
                True,
                False,
                False,
                detail=f"false positive types={kinds}",
            )
        return CaseScore(case.id, case.failure_mode, False, False, True, True)

    matched = [
        item
        for item in report.detections
        if item.claim_event_id == case.expect_claim and item.type == case.expect_type
    ]
    if not matched:
        found = [(item.claim_event_id, item.type) for item in report.detections]
        return CaseScore(
            case.id,
            case.failure_mode,
            True,
            False,
            False,
            False,
            detail=f"missing {case.expect_claim}/{case.expect_type}; have {found}",
        )
    detection = matched[0]
    if not set(case.expect_evidence).issubset(set(detection.evidence_event_ids)):
        return CaseScore(
            case.id,
            case.failure_mode,
            True,
            True,
            False,
            False,
            detail=f"evidence {detection.evidence_event_ids} != {case.expect_evidence}",
        )
    attr = report.attribution_for(detection.id)
    proposal = report.proposal_for(detection.id)
    origin_ok = attr is not None and attr.origin_event_id == case.expect_origin
    recovery_ok = (
        proposal is not None
        and attr is not None
        and proposal.rewind_seq == attr.origin_seq
        and len(proposal.corrective_signal.quotes) > 0
        and (case.expect_checkpoint is None or proposal.checkpoint_id == case.expect_checkpoint)
        and (case.expect_scope is None or proposal.restore_scope == case.expect_scope)
        and proposal.auto_executable is False
    )
    for claim_id, type_ in case.expect_also:
        if not any(item.claim_event_id == claim_id and item.type == type_ for item in report.detections):
            return CaseScore(
                case.id,
                case.failure_mode,
                True,
                True,
                origin_ok,
                False,
                detail=f"missing compounded {claim_id}/{type_}",
            )
    detail = ""
    if not origin_ok:
        got = None if attr is None else attr.origin_event_id
        detail = f"origin {got} != {case.expect_origin}"
    elif not recovery_ok:
        scope = None if proposal is None else proposal.restore_scope
        checkpoint = None if proposal is None else proposal.checkpoint_id
        detail = f"recovery scope={scope} checkpoint={checkpoint} expected scope={case.expect_scope} cp={case.expect_checkpoint}"
    return CaseScore(
        case.id,
        case.failure_mode,
        True,
        True,
        origin_ok,
        recovery_ok and not detail,
        detail=detail,
    )


def run_benchmark(cases: list[BenchCase] | None = None) -> BenchResult:
    result = BenchResult()
    for case in cases or CASES:
        report = EvidenceOrbit().analyze(case.events)
        result.scores.append(score_case(case, report))
    return result
