"""Incremental evidence graph.

Nodes are trajectory events. Structural edges (`derives_from`) come from
parent links and tool call/result pairing. Semantic edges (`supports`,
`contradicts`, `violates`) are written by the detector.
"""

from __future__ import annotations

from evidenceorbit.models import (
    EdgeKind,
    EventKind,
    GraphEdge,
    GraphNode,
    GraphSnapshot,
    NodeKind,
    TrajectoryEvent,
)


def node_kind_for(kind: EventKind) -> NodeKind:
    return {
        EventKind.CLAIM: NodeKind.CLAIM,
        EventKind.TOOL_RESULT: NodeKind.TOOL_RESULT,
        EventKind.OBSERVATION: NodeKind.OBSERVATION,
        EventKind.GOAL: NodeKind.GOAL,
        EventKind.CONTRACT: NodeKind.CONTRACT,
        EventKind.HUMAN: NodeKind.EVIDENCE,
    }.get(kind, NodeKind.OTHER)


class EvidenceGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

    def add_event(self, event: TrajectoryEvent) -> None:
        label = event.text.strip() or event.kind.value
        if len(label) > 140:
            label = label[:137] + "..."
        self.nodes[event.id] = GraphNode(
            id=event.id,
            kind=node_kind_for(event.kind),
            event_id=event.id,
            label=label,
            seq=event.seq,
        )

    def add_structural(self, events: list[TrajectoryEvent]) -> None:
        """Replace structural edges. Semantic edges are left to the caller."""
        semantic = [edge for edge in self.edges if edge.kind != EdgeKind.DERIVES_FROM]
        structural: list[GraphEdge] = []
        known = {event.id for event in events}
        calls: dict[str, str] = {}
        for event in events:
            if event.kind == EventKind.TOOL_CALL:
                call_id = event.payload.get("call_id")
                if isinstance(call_id, str):
                    calls[call_id] = event.id
        seen: set[tuple[str, str]] = set()
        for event in events:
            parents = list(event.parent_ids)
            if event.kind == EventKind.TOOL_RESULT:
                call_id = event.payload.get("call_id")
                if isinstance(call_id, str) and call_id in calls:
                    parents.append(calls[call_id])
            for parent in parents:
                if parent not in known:
                    continue
                key = (event.id, parent)
                if key in seen:
                    continue
                seen.add(key)
                structural.append(
                    GraphEdge(
                        source=event.id,
                        target=parent,
                        kind=EdgeKind.DERIVES_FROM,
                        detail="parent",
                    )
                )
        self.edges = structural + semantic

    def set_semantic(self, edges: list[GraphEdge]) -> None:
        structural = [edge for edge in self.edges if edge.kind == EdgeKind.DERIVES_FROM]
        self.edges = structural + edges

    def snapshot(self) -> GraphSnapshot:
        nodes = sorted(self.nodes.values(), key=lambda node: (node.seq, node.id))
        return GraphSnapshot(nodes=nodes, edges=list(self.edges))

    def ancestors(self, event_id: str) -> list[str]:
        """Parent-chain (derives_from targets), not including event_id."""
        parents: dict[str, list[str]] = {}
        for edge in self.edges:
            if edge.kind == EdgeKind.DERIVES_FROM:
                parents.setdefault(edge.source, []).append(edge.target)
        out: list[str] = []
        stack = list(parents.get(event_id, []))
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            out.append(current)
            stack.extend(parents.get(current, []))
        return out
