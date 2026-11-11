"""Core detector, attribution, recovery, audit, adapters, CLI, MCP, benchmark."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from evidenceorbit import EvidenceOrbit, __version__
from evidenceorbit.adapters import LangGraphAdapter, ReActAdapter
from evidenceorbit.audit import AuditLog
from evidenceorbit.benchmarks.corpus import CASES
from evidenceorbit.benchmarks.runner import run_benchmark
from evidenceorbit.canonical import content_hash
from evidenceorbit.checkpoint import FileCheckpointStore, InMemoryCheckpointStore
from evidenceorbit.cli import main
from evidenceorbit.facts import extract_text_facts, facts_contradict, make_fact
from evidenceorbit.judge import HeuristicJudge, LLMJudge
from evidenceorbit.mcp_server import MCPSidecar, _read_message, _write_message
from evidenceorbit.models import Checkpoint, RecoveryPolicy
from evidenceorbit.orbit import coerce_event


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_benchmark_is_thirty_cases() -> None:
    assert len(CASES) == 30
    modes = {case.failure_mode for case in CASES}
    assert {"early_wrong_decision", "ignored_tool_result", "contract_violation", "compounding", "happy"} <= modes


def test_benchmark_meets_bar() -> None:
    result = run_benchmark()
    assert result.ok, result.summary()
    assert result.detection_recall == 1
    assert result.attribution_accuracy == 1
    assert result.recovery_success == 1
    assert result.false_positive_rate == 0


def test_text_numeric_contradiction() -> None:
    left = extract_text_facts("The warehouse quantity is 100")
    right = extract_text_facts("The warehouse quantity is 9")
    assert left and right
    assert facts_contradict(left[0], right[0])


def test_text_antonym_and_non_conflict() -> None:
    present = make_fact("file", "is", "present", source="text")
    missing = make_fact("file", "is", "missing", source="text")
    available = make_fact("file", "is", "available", source="text")
    assert facts_contradict(present, missing)
    assert not facts_contradict(present, available)


def test_polarity_flip() -> None:
    yes = make_fact("refund", "refund", "true", polarity="assert")
    no = make_fact("refund", "refund", "true", polarity="deny")
    assert facts_contradict(yes, no)


def test_soft_judge_path() -> None:
    events = [
        {
            "id": "s1",
            "seq": 1,
            "ts": "2026-04-21T12:00:01Z",
            "session_id": "soft",
            "kind": "claim",
            "text": "backup completed successfully overnight",
            "payload": {},
        },
        {
            "id": "s2",
            "seq": 2,
            "ts": "2026-04-21T12:00:02Z",
            "session_id": "soft",
            "kind": "observation",
            "text": "backup did not complete successfully overnight",
            "payload": {},
            "actor": "environment",
        },
    ]
    report = EvidenceOrbit().analyze(events)
    assert report.detections, "soft path should confirm via the heuristic judge"
    detection = report.detections[0]
    assert detection.judge is not None
    assert detection.judge.model == "evidenceorbit-heuristic-v1"
    assert detection.judge.temperature == 0.0
    assert detection.judge.prompt_hash == content_hash(
        {
            "task": "contradiction_judge",
            "claim": events[0]["text"],
            "evidence": events[1]["text"],
        }
    )
    assert report.audit_head is not None


def test_llm_judge_records_model() -> None:
    def complete(prompt: str) -> str:
        assert "CLAIM" in prompt
        return json.dumps({"contradicts": True, "confidence": 0.7, "explanation": "model says no"})

    judge = LLMJudge(complete, model="stub-model", temperature=0.0)
    verdict = judge.judge("backup completed successfully overnight", "backup did not complete successfully overnight")
    assert verdict.contradicts
    assert verdict.model == "stub-model"
    assert verdict.prompt_hash


def test_propose_only_refuses_without_confirm() -> None:
    case = next(item for item in CASES if item.id == "bm14_checkpoint")
    orbit = EvidenceOrbit()
    orbit.ingest_many(case.events)
    proposal = orbit.report.proposals[0]
    refused = orbit.apply_recovery(proposal.id, confirm=False)
    assert refused.applied is False
    applied = orbit.apply_recovery(proposal.id, confirm=True)
    assert applied.applied is True
    assert applied.restored_checkpoint_id == "cp_014"
    assert applied.injected_event_id is not None
    assert all(event.seq < proposal.rewind_seq or event.id == applied.injected_event_id for event in orbit.events)
    assert orbit.latent_errors() == []


def test_destructive_blocked_even_with_confirm() -> None:
    case = next(item for item in CASES if item.id == "bm09_destructive")
    orbit = EvidenceOrbit()
    orbit.ingest_many(case.events)
    proposal = next(item for item in orbit.report.proposals if item.restore_scope == "external_blocked")
    result = orbit.apply_recovery(proposal.id, confirm=True)
    assert result.applied is False
    armed = EvidenceOrbit(policy=RecoveryPolicy(mode="propose_only", allow_destructive=True))
    armed.ingest_many(case.events)
    proposal = next(item for item in armed.report.proposals if item.restore_scope == "external_blocked")
    applied = armed.apply_recovery(proposal.id, confirm=True)
    assert applied.applied is True


def test_auto_policy_executes_safe_rewind() -> None:
    case = next(item for item in CASES if item.id == "bm01_early_price")
    orbit = EvidenceOrbit(policy=RecoveryPolicy(mode="auto"))
    orbit.ingest_many(case.events)
    proposal = orbit.report.proposals[0]
    assert proposal.auto_executable is True
    result = orbit.apply_recovery(proposal.id, confirm=False)
    assert result.applied is True


def test_audit_chain_and_file_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    clock = iter([f"2026-04-21T00:00:{index:02d}Z" for index in range(30)])

    def now() -> str:
        return next(clock)

    orbit = EvidenceOrbit(audit_path=path, clock=now)
    case = next(item for item in CASES if item.id == "bm01_early_price")
    orbit.ingest_many(case.events)
    assert orbit.audit.verify()
    reloaded = AuditLog(path)
    assert reloaded.verify()
    assert reloaded.head() == orbit.audit.head()
    assert "detection" in reloaded.summary()


def test_file_checkpoint_store(tmp_path: Path) -> None:
    store = FileCheckpointStore(tmp_path)
    store.save(
        Checkpoint(
            id="cp_a",
            session_id="s",
            seq=1,
            ts="2026-04-21T00:00:01Z",
            agent_context={"n": 1},
            environment={"disk": "ok"},
        )
    )
    store.save(
        Checkpoint(
            id="cp_b",
            session_id="s",
            seq=4,
            ts="2026-04-21T00:00:04Z",
            agent_context={"n": 2},
        )
    )
    assert store.latest_before("s", 4).id == "cp_a"  # type: ignore[union-attr]
    assert store.load("cp_b").agent_context["n"] == 2
    assert InMemoryCheckpointStore().latest_before("missing", 3) is None


def test_react_adapter_detects() -> None:
    adapter = ReActAdapter(session_id="loop")
    adapter.goal("Book the flight.")
    adapter.llm("prompt", "The fare is 420.", model="stub")
    llm_id = adapter.last_event_id()
    assert llm_id is not None
    adapter.claim("The fare is 420.", parent_ids=[llm_id], subject="flight", predicate="price", value=420)
    found = adapter.tool(
        "search_flights",
        {"to": "SFO"},
        "The fare is 880.",
        payload={"subject": "flight", "predicate": "price", "value": 880},
    )
    assert found
    assert found[0].type == "unused_counterevidence"


def test_langgraph_adapter_wraps_node() -> None:
    adapter = LangGraphAdapter(session_id="lg")

    def node(state: dict) -> dict:
        messages = list(state["messages"])
        messages.append(
            {
                "id": "m_claim",
                "role": "assistant",
                "content": "The fare is 420.",
                "subject": "flight",
                "predicate": "price",
                "value": 420,
            }
        )
        return {"messages": messages}

    wrapped = adapter.wrap_node(node)
    state = {
        "messages": [
            {"id": "m_goal", "role": "user", "content": "Book a flight."},
            {
                "id": "m_tool",
                "role": "tool",
                "name": "search_flights",
                "content": "The fare is 880.",
                "subject": "flight",
                "predicate": "price",
                "value": 880,
                "side_effect": "read",
            },
        ]
    }
    out = wrapped(state)
    assert any(message["id"] == "m_claim" for message in out["messages"])
    types = {item.type for item in adapter.orbit.latent_errors()}
    assert "unused_counterevidence" in types


def test_correction_is_not_compounded() -> None:
    events = [
        {"id": "a", "seq": 1, "ts": "2026-04-21T12:00:01Z", "session_id": "fix", "kind": "llm_call", "text": "qty 100", "payload": {"model": "stub"}},
        {"id": "b", "seq": 2, "ts": "2026-04-21T12:00:02Z", "session_id": "fix", "kind": "claim", "text": "qty 100", "parent_ids": ["a"], "payload": {"subject": "warehouse", "predicate": "quantity", "value": 100}},
        {"id": "c", "seq": 3, "ts": "2026-04-21T12:00:03Z", "session_id": "fix", "kind": "tool_result", "text": "qty 9", "payload": {"subject": "warehouse", "predicate": "quantity", "value": 9, "tool_name": "read_inventory", "side_effect": "read"}},
        {"id": "d", "seq": 4, "ts": "2026-04-21T12:00:04Z", "session_id": "fix", "kind": "claim", "text": "qty 9", "parent_ids": ["b"], "payload": {"subject": "warehouse", "predicate": "quantity", "value": 9, "citations": ["c"], "supersedes": ["b"]}},
    ]
    report = EvidenceOrbit().analyze(events)
    assert report.detections == []


def test_cli_analyze_and_bench(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    case = next(item for item in CASES if item.id == "bm01_early_price")
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(event) + "\n" for event in case.events), encoding="utf-8")
    out = tmp_path / "out.txt"
    assert main(["analyze", str(trace), "--format", "summary", "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "unused_counterevidence" in text
    assert "origin=evt_002" in text
    assert main(["bench"]) == 0
    captured = capsys.readouterr()
    assert "recall=1.000" in captured.out


def test_mcp_tools() -> None:
    sidecar = MCPSidecar()
    init = sidecar.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init is not None
    assert init["result"]["serverInfo"]["name"] == "evidenceorbit"
    listing = sidecar.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert listing is not None
    names = {tool["name"] for tool in listing["result"]["tools"]}
    assert names == {"check_trajectory", "get_latent_errors", "propose_recovery", "apply_recovery"}
    case = next(item for item in CASES if item.id == "bm02_ignored_tool")
    called = sidecar.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "check_trajectory", "arguments": {"events": case.events}},
        }
    )
    assert called is not None
    assert called["result"]["isError"] is False
    payload = json.loads(called["result"]["content"][0]["text"])
    assert payload["detections"]
    session_id = payload["session_id"]
    detection_id = payload["detections"][0]["id"]
    proposal_id = payload["proposals"][0]["id"]
    proposed = sidecar.handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "propose_recovery",
                "arguments": {"session_id": session_id, "detection_id": detection_id},
            },
        }
    )
    assert proposed is not None
    assert json.loads(proposed["result"]["content"][0]["text"])["rewind_seq"] == payload["proposals"][0]["rewind_seq"]
    refused = sidecar.handle(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "apply_recovery",
                "arguments": {"session_id": session_id, "proposal_id": proposal_id, "confirm": False},
            },
        }
    )
    assert refused is not None
    assert json.loads(refused["result"]["content"][0]["text"])["applied"] is False
    assert sidecar.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_mcp_framing() -> None:
    sidecar = MCPSidecar()
    buffer = io.StringIO()
    _write_message(buffer, {"jsonrpc": "2.0", "id": 1, "method": "ping"})
    buffer.seek(0)
    message = _read_message(buffer)
    assert message is not None
    assert sidecar.handle(message)["result"] == {}


def test_fast_path_latency() -> None:
    import time

    orbit = EvidenceOrbit()
    samples: list[float] = []
    for index in range(1, 61):
        event = {
            "id": f"e{index}",
            "seq": index,
            "ts": f"2026-04-21T12:{index // 60:02d}:{index % 60:02d}Z",
            "session_id": "lat",
            "kind": "observation" if index % 2 == 0 else "claim",
            "text": f"batch quantity is {index}",
            "payload": {"subject": f"batch{index}", "predicate": "quantity", "value": index},
        }
        start = time.perf_counter()
        orbit.ingest(event)
        samples.append((time.perf_counter() - start) * 1000)
    mean = sum(samples) / len(samples)
    p95 = sorted(samples)[int(len(samples) * 0.95) - 1]
    assert mean < 50, f"mean {mean:.2f}ms p95 {p95:.2f}ms"
    assert p95 < 50, f"p95 {p95:.2f}ms"


def test_duplicate_and_schema() -> None:
    orbit = EvidenceOrbit()
    event = {
        "id": "e",
        "seq": 1,
        "ts": "2026-04-21T12:00:01Z",
        "session_id": "s",
        "kind": "goal",
        "text": "go",
    }
    orbit.ingest(event)
    with pytest.raises(ValueError):
        orbit.ingest(event)
    coerced = coerce_event(event)
    assert coerced.content_hash


def test_heuristic_judge_negative() -> None:
    verdict = HeuristicJudge().judge("unrelated apples", "oranges in a bowl")
    assert verdict.contradicts is False


def test_cli_wrap(tmp_path: Path) -> None:
    script = tmp_path / "agent.py"
    script.write_text(
        "import json\n"
        "print(json.dumps({'schema_version':'0.1.0','id':'evt_001','seq':1,'ts':'2026-04-21T12:00:01Z',"
        "'session_id':'wrap','kind':'claim','text':'qty is 3','payload':{'subject':'box','predicate':'quantity','value':3}}))\n"
        "print(json.dumps({'schema_version':'0.1.0','id':'evt_002','seq':2,'ts':'2026-04-21T12:00:02Z',"
        "'session_id':'wrap','kind':'tool_result','text':'qty is 9','payload':{'subject':'box','predicate':'quantity','value':9,"
        "'tool_name':'read_inventory','side_effect':'read','ok':True}}))\n",
        encoding="utf-8",
    )
    code = main(["wrap", "--", "python3", str(script)])
    assert code == 0
