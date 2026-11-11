"""EvidenceOrbit: latent-error sidecar for long-horizon agents.

Detect unused contradictory evidence, attribute the origin step, and propose
a surgical rewind. Framework-agnostic. Propose-only by default.

Public API stability: symbols exported here and the JSON schemas written by
``evidenceorbit schemas`` are stable across 0.1.x. Breaking changes require
0.2.0 and a changelog entry.
"""

from evidenceorbit.adapters import LangGraphAdapter, ReActAdapter
from evidenceorbit.audit import AuditLog
from evidenceorbit.checkpoint import (
    CheckpointAdapter,
    FileCheckpointStore,
    InMemoryCheckpointStore,
)
from evidenceorbit.models import (
    AnalysisReport,
    Attribution,
    Checkpoint,
    Detection,
    RecoveryPolicy,
    RecoveryProposal,
    RecoveryResult,
    TrajectoryEvent,
)
from evidenceorbit.orbit import EvidenceOrbit
from evidenceorbit.version import SCHEMA_VERSION, __version__

__all__ = [
    "SCHEMA_VERSION",
    "AnalysisReport",
    "Attribution",
    "AuditLog",
    "Checkpoint",
    "CheckpointAdapter",
    "Detection",
    "EvidenceOrbit",
    "FileCheckpointStore",
    "InMemoryCheckpointStore",
    "LangGraphAdapter",
    "ReActAdapter",
    "RecoveryPolicy",
    "RecoveryProposal",
    "RecoveryResult",
    "TrajectoryEvent",
    "__version__",
]
