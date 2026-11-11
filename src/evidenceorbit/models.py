"""Canonical schemas for events, graph, detections, recovery, and audit.

The JSON shape is the public contract (see docs/schemas.md). A TypeScript port
should mirror these models field-for-field.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evidenceorbit.version import SCHEMA_VERSION


class EventKind(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OBSERVATION = "observation"
    CLAIM = "claim"
    GOAL = "goal"
    CONTRACT = "contract"
    HUMAN = "human"
    CHECKPOINT = "checkpoint"
    SYSTEM = "system"


EVIDENCE_KINDS = frozenset(
    {EventKind.TOOL_RESULT, EventKind.OBSERVATION, EventKind.HUMAN}
)
CONSTRAINT_KINDS = frozenset({EventKind.CONTRACT, EventKind.GOAL})


class Constraint(BaseModel):
    """Explicit task contract / acceptance criterion."""

    model_config = ConfigDict(extra="forbid")

    id: str
    modality: Literal["must", "must_not"]
    text: str
    subject: str | None = None
    predicate: str | None = None
    value: Any = None


class TrajectoryEvent(BaseModel):
    """One ordered, addressable step in an agent trajectory.

    Push mode: the agent emits these directly.
    Pull mode: adapters / MCP proxy synthesize them from runtime messages.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    id: str
    seq: int = Field(ge=0)
    ts: str
    session_id: str
    kind: EventKind
    actor: str = "agent"
    parent_ids: list[str] = Field(default_factory=list)
    text: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    content_hash: str | None = None

    @field_validator("id", "session_id", "ts")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("must be a non-empty string")
        return str(value)


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    value: str
    polarity: Literal["assert", "deny"] = "assert"
    source: Literal["structured", "text"] = "structured"


class NodeKind(str, Enum):
    CLAIM = "claim"
    EVIDENCE = "evidence"
    GOAL = "goal"
    TOOL_RESULT = "tool_result"
    OBSERVATION = "observation"
    CONTRACT = "contract"
    OTHER = "other"


class EdgeKind(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DERIVES_FROM = "derives_from"
    VIOLATES = "violates"


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: NodeKind
    event_id: str
    label: str
    seq: int


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    kind: EdgeKind
    confidence: float = 1.0
    detail: str = ""


class GraphSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[GraphNode]
    edges: list[GraphEdge]


class JudgeTrace(BaseModel):
    """Every non-deterministic (or judge) decision is logged with identity."""

    model_config = ConfigDict(extra="forbid")

    model: str
    prompt_hash: str
    temperature: float
    contradicts: bool
    confidence: float
    explanation: str


DetectionType = Literal[
    "direct_contradiction",
    "unused_counterevidence",
    "contract_violation",
    "internal_inconsistency",
    "compounded_error",
]


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    id: str
    session_id: str
    type: DetectionType
    confidence: float = Field(ge=0.0, le=1.0)
    claim_event_id: str
    evidence_event_ids: list[str]
    explanation: str
    claim_fact: Fact | None = None
    evidence_fact: Fact | None = None
    constraint_id: str | None = None
    judge: JudgeTrace | None = None


class Attribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    id: str
    detection_id: str
    origin_event_id: str
    origin_seq: int
    propagation_path: list[str]
    propagation_summary: str
    unused_evidence_ids: list[str]


class CorrectiveSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str
    evidence_event_ids: list[str]
    quotes: list[str]
    do_not_repeat: str


class RecoveryProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    id: str
    detection_id: str
    session_id: str
    origin_event_id: str
    rewind_seq: int
    checkpoint_id: str | None = None
    restore_scope: Literal["partial_belief", "full_state", "external_blocked"]
    corrective_signal: CorrectiveSignal
    auto_executable: bool
    blocked_reason: str | None = None
    destructive_event_ids: list[str] = Field(default_factory=list)
    mode: Literal["propose_only", "auto"] = "propose_only"


class RecoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    proposal_id: str
    applied: bool
    reason: str
    restored_checkpoint_id: str | None = None
    injected_event_id: str | None = None
    truncated_to_seq: int | None = None


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    id: str
    session_id: str
    seq: int
    ts: str
    label: str = ""
    agent_context: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    event_cursor: str | None = None


class RecoveryPolicy(BaseModel):
    """Default is propose-only. Auto-rewind never fires on destructive effects
    unless allow_destructive is explicitly set.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["propose_only", "auto"] = "propose_only"
    allow_destructive: bool = False


class AuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    id: str
    record_type: Literal[
        "detection",
        "resolution",
        "attribution",
        "recovery_proposal",
        "recovery_applied",
        "recovery_refused",
        "judge",
    ]
    session_id: str
    ts: str
    prev_hash: str
    content_hash: str
    body: dict[str, Any]


class AnalysisReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    session_id: str
    event_count: int
    detections: list[Detection]
    attributions: list[Attribution]
    proposals: list[RecoveryProposal]
    graph: GraphSnapshot
    audit_head: str | None = None

    def attribution_for(self, detection_id: str) -> Attribution | None:
        for item in self.attributions:
            if item.detection_id == detection_id:
                return item
        return None

    def proposal_for(self, detection_id: str) -> RecoveryProposal | None:
        for item in self.proposals:
            if item.detection_id == detection_id:
                return item
        return None
