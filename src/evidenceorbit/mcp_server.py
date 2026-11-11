"""Minimal MCP (stdio, JSON-RPC 2.0) server.

Implements initialize, tools/list, and tools/call well enough for any
MCP-capable host to use EvidenceOrbit as a sidecar. No third-party MCP
SDK is required. Framing is Content-Length, matching the MCP transport.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from evidenceorbit.models import RecoveryPolicy
from evidenceorbit.orbit import EvidenceOrbit
from evidenceorbit.version import __version__

_PROTOCOL = "2024-11-05"


def _tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "check_trajectory",
            "description": "Analyze a trajectory and return latent errors, origins, and recovery proposals.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "events": {"type": "array", "items": {"type": "object"}},
                    "session_id": {"type": "string"},
                },
                "required": ["events"],
            },
        },
        {
            "name": "get_latent_errors",
            "description": "Return active latent errors for a session previously checked.",
            "inputSchema": {
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
            },
        },
        {
            "name": "propose_recovery",
            "description": "Return the surgical recovery proposal for one detection.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "detection_id": {"type": "string"},
                },
                "required": ["session_id", "detection_id"],
            },
        },
        {
            "name": "apply_recovery",
            "description": "Apply a recovery proposal. Gated: confirm must be true. Refuses destructive effects by default.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "proposal_id": {"type": "string"},
                    "confirm": {"type": "boolean"},
                },
                "required": ["session_id", "proposal_id", "confirm"],
            },
        },
    ]


class MCPSidecar:
    def __init__(self, *, policy: RecoveryPolicy | None = None) -> None:
        self.policy = policy or RecoveryPolicy()
        self.sessions: dict[str, EvidenceOrbit] = {}

    def _session(self, session_id: str) -> EvidenceOrbit:
        found = self.sessions.get(session_id)
        if found is None:
            found = EvidenceOrbit(policy=self.policy)
            self.sessions[session_id] = found
        return found

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "check_trajectory":
            events = list(arguments.get("events") or [])
            if not events:
                raise ValueError("events must be a non-empty list")
            session_id = str(arguments.get("session_id") or events[0].get("session_id") or "mcp")
            normalized = []
            for event in events:
                if not isinstance(event, dict):
                    raise ValueError("each event must be an object")
                item = dict(event)
                item.setdefault("session_id", session_id)
                normalized.append(item)
            orbit = EvidenceOrbit(policy=self.policy)
            report = orbit.ingest_many(normalized)
            self.sessions[report.session_id or session_id] = orbit
            return report.model_dump(mode="json")
        if name == "get_latent_errors":
            session_id = str(arguments["session_id"])
            stored = self.sessions.get(session_id)
            if stored is None:
                return {"session_id": session_id, "detections": []}
            return {
                "session_id": session_id,
                "detections": [item.model_dump(mode="json") for item in stored.latent_errors()],
            }
        if name == "propose_recovery":
            orbit = self._session(str(arguments["session_id"]))
            proposal = orbit.propose_recovery(str(arguments["detection_id"]))
            return proposal.model_dump(mode="json")
        if name == "apply_recovery":
            orbit = self._session(str(arguments["session_id"]))
            result = orbit.apply_recovery(
                str(arguments["proposal_id"]),
                confirm=bool(arguments.get("confirm")),
            )
            return result.model_dump(mode="json")
        raise ValueError(f"Unknown tool {name}")

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        msg_id = message.get("id")
        if method == "notifications/initialized" or (method or "").startswith("notifications/"):
            return None
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": _PROTOCOL,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "evidenceorbit", "version": __version__},
                },
            }
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": _tool_specs()}}
        if method == "tools/call":
            params = message.get("params") or {}
            name = str(params.get("name") or "")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            try:
                payload = self.call_tool(name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(payload)}],
                        "isError": False,
                    },
                }
            except Exception as exc:  # MCP tools report errors, they do not crash the host
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": str(exc)}],
                        "isError": True,
                    },
                }
        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
        if msg_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def serve(stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    """Block on stdio and speak MCP. Used by `evidenceorbit mcp`."""
    reader = stdin or sys.stdin
    writer = stdout or sys.stdout
    sidecar = MCPSidecar()
    while True:
        message = _read_message(reader)
        if message is None:
            return
        response = sidecar.handle(message)
        if response is not None:
            _write_message(writer, response)


def _read_message(reader: TextIO) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = reader.readline()
        if line == "":
            return None
        if line in {"\r\n", "\n"}:
            break
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = reader.read(length)
    if not body:
        return None
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("MCP message must be a JSON object")
    return parsed


def _write_message(writer: TextIO, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    writer.write(f"Content-Length: {len(raw)}\r\n\r\n")
    writer.flush()
    # TextIO may not accept bytes; write decoded UTF-8. Content-Length is bytes.
    writer.write(raw.decode("utf-8"))
    writer.flush()
