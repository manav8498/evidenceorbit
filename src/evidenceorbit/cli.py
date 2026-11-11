"""Command line interface.

    evidenceorbit analyze <trace.jsonl>
    evidenceorbit wrap -- <command>
    evidenceorbit mcp
    evidenceorbit bench
    evidenceorbit schemas --out <dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from evidenceorbit.models import EventKind, TrajectoryEvent
from evidenceorbit.orbit import EvidenceOrbit
from evidenceorbit.version import __version__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidenceorbit", description="Latent-error sidecar for agent trajectories.")
    parser.add_argument("--version", action="version", version=f"evidenceorbit {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    analyze = sub.add_parser("analyze", help="Offline analysis of a JSONL trajectory")
    analyze.add_argument("trace", type=Path)
    analyze.add_argument("-o", "--output", type=Path, default=None)
    analyze.add_argument("--format", choices=("json", "summary"), default="json")

    wrap = sub.add_parser("wrap", help="Run a command and ingest stdout JSON events")
    wrap.add_argument("command", nargs=argparse.REMAINDER)

    sub.add_parser("mcp", help="Start the MCP server on stdio")

    bench = sub.add_parser("bench", help="Run the v0.1 failure-mode benchmark")
    bench.add_argument("--json", action="store_true")

    schemas = sub.add_parser("schemas", help="Write JSON Schemas for the public models")
    schemas.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "analyze":
        return _analyze(args.trace, args.output, args.format)
    if args.cmd == "wrap":
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            print("usage: evidenceorbit wrap -- <command>", file=sys.stderr)
            return 2
        return _wrap(command)
    if args.cmd == "mcp":
        from evidenceorbit.mcp_server import serve

        serve()
        return 0
    if args.cmd == "bench":
        from evidenceorbit.benchmarks.runner import run_benchmark

        result = run_benchmark()
        if args.json:
            print(json.dumps(result.as_dict(), indent=2))
        else:
            print(result.summary())
        return 0 if result.ok else 1
    if args.cmd == "schemas":
        return _schemas(args.out)
    return 2


def _load_events(path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError("JSON array expected")
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("event must be an object")
            events.append(item)
        return events
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"line {line_no} is not an object")
        events.append(item)
    return events


def _analyze(path: Path, output: Path | None, fmt: str) -> int:
    report = EvidenceOrbit().analyze(_load_events(path))
    if fmt == "summary":
        rendered = render_summary(report.session_id, report)
    else:
        rendered = report.model_dump_json(indent=2)
    if output is not None:
        output.write_text(rendered + ("\n" if fmt == "summary" else "\n"), encoding="utf-8")
    else:
        print(rendered)
    return 0


def render_summary(session_id: str, report: object) -> str:
    from evidenceorbit.models import AnalysisReport

    if not isinstance(report, AnalysisReport):
        raise TypeError("expected AnalysisReport")
    lines = [
        f"session {report.session_id or session_id}  events={report.event_count}  "
        f"detections={len(report.detections)}",
    ]
    for detection in report.detections:
        attr = report.attribution_for(detection.id)
        proposal = report.proposal_for(detection.id)
        origin = attr.origin_event_id if attr else "?"
        rewind = proposal.rewind_seq if proposal else "?"
        lines.append(
            f"- [{detection.type}] {detection.confidence:.2f} claim={detection.claim_event_id} "
            f"evidence={','.join(detection.evidence_event_ids)} origin={origin} rewind_seq={rewind}"
        )
        lines.append(f"  {detection.explanation}")
        if proposal is not None:
            lines.append(f"  recovery: {proposal.restore_scope} checkpoint={proposal.checkpoint_id}")
            lines.append(f"  signal: {proposal.corrective_signal.instruction}")
    if not report.detections:
        lines.append("no latent errors")
    return "\n".join(lines)


def _wrap(command: list[str]) -> int:
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    orbit = EvidenceOrbit()
    seq = 0
    session = "wrap"
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        seq += 1
        raw = line.strip()
        event: dict[str, object] | None = None
        if raw.startswith("{") and raw.endswith("}"):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and "kind" in parsed and "id" in parsed:
                event = parsed
        if event is None:
            event = {
                "schema_version": "0.1.0",
                "id": f"wrap_{seq:04d}",
                "seq": seq,
                "ts": f"2026-04-21T15:00:{seq % 60:02d}Z",
                "session_id": session,
                "kind": EventKind.OBSERVATION.value,
                "text": raw,
                "payload": {},
            }
        try:
            orbit.ingest(event)
        except (ValueError, Exception) as exc:
            print(f"[evidenceorbit] skipped line {seq}: {exc}", file=sys.stderr)
    code = proc.wait()
    print(render_summary(session, orbit.report), file=sys.stderr)
    return int(code)


def _schemas(out: Path) -> int:
    from evidenceorbit.models import (
        AnalysisReport,
        AuditRecord,
        Checkpoint,
        Detection,
        RecoveryProposal,
        TrajectoryEvent,
    )

    out.mkdir(parents=True, exist_ok=True)
    models: dict[str, type[BaseModel]] = {
        "trajectory-event.schema.json": TrajectoryEvent,
        "detection.schema.json": Detection,
        "recovery-proposal.schema.json": RecoveryProposal,
        "analysis-report.schema.json": AnalysisReport,
        "audit-record.schema.json": AuditRecord,
        "checkpoint.schema.json": Checkpoint,
    }
    for name, model in models.items():
        (out / name).write_text(json.dumps(model.model_json_schema(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(models)} schemas to {out}")
    return 0


def load_events_checked(path: Path) -> list[TrajectoryEvent]:
    return [TrajectoryEvent.model_validate(item) for item in _load_events(path)]
