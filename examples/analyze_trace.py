"""Offline analysis of a JSONL trajectory."""

import json
import sys
from pathlib import Path

from evidenceorbit import EvidenceOrbit


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/traces/early_price.jsonl")
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = EvidenceOrbit().analyze(events)
    print(f"{report.session_id}: {len(report.detections)} latent error(s) in {report.event_count} events")
    for detection in report.detections:
        proposal = report.proposal_for(detection.id)
        print(f"- {detection.type} @ {detection.claim_event_id}: {detection.explanation}")
        if proposal:
            print(f"  rewind before seq {proposal.rewind_seq}, scope {proposal.restore_scope}")


if __name__ == "__main__":
    main()
