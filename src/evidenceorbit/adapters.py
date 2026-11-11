"""Minimal adapters for two agent styles.

`ReActAdapter` wraps a hand-rolled tool-calling loop (push mode).
`LangGraphAdapter` consumes LangGraph-style state dicts with a `messages`
list (pull mode). Neither imports its framework. Both emit the canonical
trajectory schema and return newly active detections.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from evidenceorbit.models import Detection, EventKind, TrajectoryEvent
from evidenceorbit.orbit import EvidenceOrbit


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "\n".join(part for part in parts if part)
    return str(content)


def _payload_from_message(message: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    raw = message.get("payload")
    if isinstance(raw, dict):
        payload.update(raw)
    for key in (
        "subject",
        "predicate",
        "value",
        "polarity",
        "call_id",
        "tool_name",
        "name",
        "side_effect",
        "ok",
        "citations",
        "supersedes",
        "status",
        "constraints",
        "arguments",
        "tool_call_id",
    ):
        if key in message and key not in payload:
            payload[key] = message[key]
    if "tool_name" not in payload and isinstance(message.get("name"), str):
        payload["tool_name"] = message["name"]
    if "call_id" not in payload and isinstance(message.get("tool_call_id"), str):
        payload["call_id"] = message["tool_call_id"]
    return payload


class ReActAdapter:
    """Push-mode adapter for a generic reason/act loop.

    Call `goal`, then alternate `llm` / `tool` / `claim`. Each call ingests
    canonical events and returns detections that just became active.
    """

    def __init__(self, orbit: EvidenceOrbit | None = None, *, session_id: str = "react") -> None:
        self.orbit = orbit or EvidenceOrbit()
        self.session_id = session_id
        self._seq = 0

    def goal(self, text: str, **payload: Any) -> list[Detection]:
        return self._push(EventKind.GOAL, text, payload=payload)

    def contract(self, text: str, constraints: list[dict[str, Any]]) -> list[Detection]:
        return self._push(EventKind.CONTRACT, text, payload={"constraints": constraints})

    def llm(
        self,
        prompt: str,
        response: str,
        *,
        model: str = "unknown",
        temperature: float = 0.0,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> list[Detection]:
        from evidenceorbit.canonical import content_hash

        return self._push(
            EventKind.LLM_CALL,
            response,
            payload={
                "model": model,
                "temperature": temperature,
                "prompt_hash": content_hash({"prompt": prompt}),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "response": response,
            },
        )

    def tool(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str,
        *,
        ok: bool = True,
        call_id: str | None = None,
        side_effect: str = "read",
        payload: dict[str, Any] | None = None,
    ) -> list[Detection]:
        call = call_id or f"call_{self._seq + 1}"
        found: list[Detection] = []
        found.extend(
            self._push(
                EventKind.TOOL_CALL,
                name,
                payload={
                    "tool_name": name,
                    "arguments": arguments,
                    "call_id": call,
                    "side_effect": side_effect,
                },
            )
        )
        body = dict(payload or {})
        body.setdefault("tool_name", name)
        body.setdefault("call_id", call)
        body.setdefault("ok", ok)
        body.setdefault("side_effect", side_effect)
        found.extend(self._push(EventKind.TOOL_RESULT, result, payload=body))
        return found

    def claim(self, text: str, *, parent_ids: list[str] | None = None, **payload: Any) -> list[Detection]:
        return self._push(EventKind.CLAIM, text, parent_ids=parent_ids, payload=payload)

    def observation(self, text: str, **payload: Any) -> list[Detection]:
        return self._push(EventKind.OBSERVATION, text, payload=payload, actor="environment")

    def checkpoint(
        self,
        label: str,
        agent_context: dict[str, Any],
        environment: dict[str, Any] | None = None,
        *,
        checkpoint_id: str | None = None,
    ) -> list[Detection]:
        return self._push(
            EventKind.CHECKPOINT,
            label,
            payload={
                "checkpoint_id": checkpoint_id or f"cp_{self.session_id}_{self._seq + 1}",
                "label": label,
                "agent_context": agent_context,
                "environment": environment or {},
            },
        )

    def last_event_id(self) -> str | None:
        if not self.orbit.events:
            return None
        return self.orbit.events[-1].id

    def _push(
        self,
        kind: EventKind,
        text: str,
        *,
        payload: dict[str, Any] | None = None,
        parent_ids: list[str] | None = None,
        actor: str = "agent",
    ) -> list[Detection]:
        self._seq += 1
        minute, second = divmod(self._seq, 60)
        event = TrajectoryEvent(
            id=f"{self.session_id}_{self._seq:04d}",
            seq=self._seq,
            ts=f"2026-04-21T12:{minute:02d}:{second:02d}Z",
            session_id=self.session_id,
            kind=kind,
            actor=actor,
            parent_ids=list(parent_ids or []),
            text=text,
            payload=dict(payload or {}),
        )
        return self.orbit.ingest(event)


class LangGraphAdapter:
    """Pull-mode adapter for a LangGraph-style node `(state) -> state`.

    Messages are dicts (or objects) with `role` and `content`. Optional
    `payload` / `subject` / `predicate` / `value` become canonical facts.
    Assistant messages become an LLM call plus a claim. Tool messages become
    tool results. The first user message is the goal; later user messages
    are human evidence.
    """

    def __init__(self, orbit: EvidenceOrbit | None = None, *, session_id: str = "langgraph") -> None:
        self.orbit = orbit or EvidenceOrbit()
        self.session_id = session_id
        self._seq = 0
        self._seen: set[str] = set()
        self._user_count = 0
        self._last_llm: str | None = None

    def wrap_node(self, node_fn: Callable[[dict[str, Any]], Any], *, name: str = "node") -> Callable[[dict[str, Any]], Any]:
        del name

        def wrapped(state: dict[str, Any]) -> Any:
            self.consume(list(state.get("messages") or []))
            result = node_fn(state)
            if isinstance(result, dict) and "messages" in result:
                self.consume(list(result.get("messages") or []))
            return result

        return wrapped

    def consume(self, messages: list[Any]) -> list[Detection]:
        found: list[Detection] = []
        for index, message in enumerate(messages):
            identity = _message_id(message, index)
            if identity in self._seen:
                continue
            self._seen.add(identity)
            found.extend(self._ingest_message(message))
        return found

    def _ingest_message(self, message: Any) -> list[Detection]:
        role = str(_field(message, "role", "assistant"))
        content = _text(_field(message, "content", ""))
        as_dict = message if isinstance(message, dict) else {"role": role, "content": content}
        payload = _payload_from_message(as_dict if isinstance(as_dict, dict) else {})
        if not isinstance(message, dict):
            for key in ("subject", "predicate", "value", "name", "tool_call_id"):
                value = _field(message, key, None)
                if value is not None and key not in payload:
                    payload[key] = value
        found: list[Detection] = []
        if role in {"user", "human"}:
            self._user_count += 1
            kind = EventKind.GOAL if self._user_count == 1 else EventKind.HUMAN
            actor = "user"
            found.extend(self._push(kind, content, payload=payload, actor=actor))
        elif role == "system":
            kind = EventKind.CONTRACT if payload.get("constraints") else EventKind.GOAL
            found.extend(self._push(kind, content, payload=payload, actor="system"))
        elif role in {"tool", "tool_result"}:
            found.extend(self._push(EventKind.TOOL_RESULT, content, payload=payload, actor="tool"))
        elif role == "assistant":
            llm_id = self._next_id()
            found.extend(
                self._push_with_id(
                    llm_id,
                    EventKind.LLM_CALL,
                    content,
                    payload={"model": str(payload.get("model") or "unknown"), "response": content},
                )
            )
            self._last_llm = llm_id
            tool_calls = _field(message, "tool_calls", None) or payload.get("tool_calls") or []
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    call_id = str(call.get("id") or call.get("call_id") or self._next_id())
                    name = str(call.get("name") or (call.get("function") or {}).get("name") or "tool")
                    arguments = call.get("arguments") or (call.get("function") or {}).get("arguments") or {}
                    found.extend(
                        self._push(
                            EventKind.TOOL_CALL,
                            name,
                            payload={
                                "tool_name": name,
                                "arguments": arguments if isinstance(arguments, dict) else {"raw": arguments},
                                "call_id": call_id,
                            },
                            parent_ids=[llm_id],
                        )
                    )
            if content.strip():
                claim_payload: dict[str, Any] = {
                    key: payload[key]
                    for key in ("subject", "predicate", "value", "polarity", "citations", "supersedes", "status")
                    if key in payload
                }
                found.extend(
                    self._push(
                        EventKind.CLAIM,
                        content,
                        payload=claim_payload,
                        parent_ids=[llm_id],
                    )
                )
        else:
            found.extend(self._push(EventKind.OBSERVATION, content, payload=payload, actor=role))
        return found

    def _push(
        self,
        kind: EventKind,
        text: str,
        *,
        payload: dict[str, Any] | None = None,
        parent_ids: list[str] | None = None,
        actor: str = "agent",
    ) -> list[Detection]:
        return self._push_with_id(self._next_id(), kind, text, payload=payload, parent_ids=parent_ids, actor=actor)

    def _push_with_id(
        self,
        event_id: str,
        kind: EventKind,
        text: str,
        *,
        payload: dict[str, Any] | None = None,
        parent_ids: list[str] | None = None,
        actor: str = "agent",
    ) -> list[Detection]:
        minute, second = divmod(self._seq, 60)
        event = TrajectoryEvent(
            id=event_id,
            seq=self._seq,
            ts=f"2026-04-21T13:{minute:02d}:{second:02d}Z",
            session_id=self.session_id,
            kind=kind,
            actor=actor,
            parent_ids=list(parent_ids or []),
            text=text,
            payload=dict(payload or {}),
        )
        return self.orbit.ingest(event)

    def _next_id(self) -> str:
        self._seq += 1
        return f"{self.session_id}_{self._seq:04d}"


def _field(message: Any, key: str, default: Any) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _message_id(message: Any, index: int) -> str:
    explicit = _field(message, "id", None)
    if explicit:
        return str(explicit)
    role = str(_field(message, "role", "msg"))
    content = _text(_field(message, "content", ""))
    return f"auto:{index}:{role}:{content[:80]}"
