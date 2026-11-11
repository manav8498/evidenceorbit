"""Pluggable checkpoints. Default stores are in-memory and local JSON files.

Plug a ProcessFork-style snapshotter or a LangGraph checkpointer behind
`CheckpointAdapter`. EvidenceOrbit only asks for save, load, and
latest_before(seq).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from evidenceorbit.models import Checkpoint, EventKind, TrajectoryEvent


class CheckpointAdapter(Protocol):
    def save(self, checkpoint: Checkpoint) -> str: ...

    def load(self, checkpoint_id: str) -> Checkpoint: ...

    def latest_before(self, session_id: str, seq: int) -> Checkpoint | None: ...


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._items: dict[str, Checkpoint] = {}

    def save(self, checkpoint: Checkpoint) -> str:
        self._items[checkpoint.id] = checkpoint
        return checkpoint.id

    def load(self, checkpoint_id: str) -> Checkpoint:
        try:
            return self._items[checkpoint_id]
        except KeyError as exc:
            raise KeyError(f"Unknown checkpoint {checkpoint_id}") from exc

    def latest_before(self, session_id: str, seq: int) -> Checkpoint | None:
        candidates = [
            item
            for item in self._items.values()
            if item.session_id == session_id and item.seq < seq
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.seq, item.id))

    def __len__(self) -> int:
        return len(self._items)


class FileCheckpointStore:
    """One JSON file per checkpoint plus no external index. Safe for local dev."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, checkpoint_id: str) -> Path:
        safe = checkpoint_id.replace("/", "_")
        return self.root / f"{safe}.json"

    def save(self, checkpoint: Checkpoint) -> str:
        path = self._path(checkpoint.id)
        path.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
        return checkpoint.id

    def load(self, checkpoint_id: str) -> Checkpoint:
        path = self._path(checkpoint_id)
        if not path.exists():
            raise KeyError(f"Unknown checkpoint {checkpoint_id}")
        return Checkpoint.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def latest_before(self, session_id: str, seq: int) -> Checkpoint | None:
        best: Checkpoint | None = None
        for path in self.root.glob("*.json"):
            item = Checkpoint.model_validate(json.loads(path.read_text(encoding="utf-8")))
            if item.session_id != session_id or item.seq >= seq:
                continue
            if best is None or (item.seq, item.id) > (best.seq, best.id):
                best = item
        return best


def checkpoint_from_event(event: TrajectoryEvent) -> Checkpoint | None:
    if event.kind != EventKind.CHECKPOINT:
        return None
    payload = event.payload
    checkpoint_id = str(payload.get("checkpoint_id") or f"cp_{event.id}")
    context = payload.get("agent_context") or {}
    environment = payload.get("environment") or {}
    if not isinstance(context, dict):
        context = {"value": context}
    if not isinstance(environment, dict):
        environment = {"value": environment}
    return Checkpoint(
        id=checkpoint_id,
        session_id=event.session_id,
        seq=event.seq,
        ts=event.ts,
        label=str(payload.get("label") or ""),
        agent_context=dict(context),
        environment=dict(environment),
        event_cursor=event.id,
    )
