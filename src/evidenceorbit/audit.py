"""Append-only, content-addressed audit log.

Each record hashes its body together with the previous record hash.
Tampering or dropping a row breaks `verify()`. JSONL is the on-disk form.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from evidenceorbit.canonical import content_hash
from evidenceorbit.models import AuditRecord
from evidenceorbit.version import SCHEMA_VERSION

_GENESIS = "0" * 64


class AuditLog:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.clock = clock or _utc_now
        self.records: list[AuditRecord] = []
        if self.path is not None and self.path.exists():
            self._load()

    def append(
        self,
        record_type: Literal[
            "detection",
            "resolution",
            "attribution",
            "recovery_proposal",
            "recovery_applied",
            "recovery_refused",
            "judge",
        ],
        session_id: str,
        body: dict[str, object],
    ) -> AuditRecord:
        prev = self.records[-1].content_hash if self.records else _GENESIS
        ts = self.clock()
        hashed = {
            "schema_version": SCHEMA_VERSION,
            "record_type": record_type,
            "session_id": session_id,
            "ts": ts,
            "prev_hash": prev,
            "body": body,
        }
        digest = content_hash(hashed)
        record = AuditRecord(
            id="aud_" + digest[:16],
            record_type=record_type,
            session_id=session_id,
            ts=ts,
            prev_hash=prev,
            content_hash=digest,
            body=body,
        )
        self.records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")
        return record

    def verify(self) -> bool:
        prev = _GENESIS
        for record in self.records:
            if record.prev_hash != prev:
                return False
            hashed = {
                "schema_version": record.schema_version,
                "record_type": record.record_type,
                "session_id": record.session_id,
                "ts": record.ts,
                "prev_hash": record.prev_hash,
                "body": record.body,
            }
            if content_hash(hashed) != record.content_hash:
                return False
            prev = record.content_hash
        return True

    def head(self) -> str | None:
        if not self.records:
            return None
        return self.records[-1].content_hash

    def to_jsonl(self) -> str:
        return "".join(record.model_dump_json() + "\n" for record in self.records)

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.record_type] = counts.get(record.record_type, 0) + 1
        lines = [f"audit records: {len(self.records)} head={self.head() or 'genesis'}"]
        for key in sorted(counts):
            lines.append(f"  {key}: {counts[key]}")
        return "\n".join(lines)

    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            self.records.append(AuditRecord.model_validate(json.loads(line)))
        if not self.verify():
            raise ValueError(f"Audit log failed verification: {self.path}")


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def body_of(model: object) -> dict[str, object]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        dumped = dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    return {"repr": repr(model)}
