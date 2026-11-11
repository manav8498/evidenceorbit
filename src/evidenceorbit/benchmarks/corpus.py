"""Thirty curated trajectories covering the v0.1 failure modes.

Each case is a finished trajectory plus the final-state expectation.
Negative cases are happy paths, respected contracts, and errors the agent
already repaired. Positive cases leave contradictory evidence unused or a
contract broken.

IDs are per-session (`evt_###`) so a case is readable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchCase:
    id: str
    title: str
    failure_mode: str
    description: str
    events: list[dict[str, Any]]
    expect_detection: bool
    expect_claim: str | None = None
    expect_type: str | None = None
    expect_origin: str | None = None
    expect_evidence: list[str] = field(default_factory=list)
    expect_also: list[tuple[str, str]] = field(default_factory=list)
    expect_checkpoint: str | None = None
    expect_scope: str | None = None


def _ev(
    session: str,
    seq: int,
    kind: str,
    text: str = "",
    *,
    parent: list[str] | None = None,
    payload: dict[str, Any] | None = None,
    actor: str = "agent",
) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "id": f"evt_{seq:03d}",
        "seq": seq,
        "ts": f"2026-04-21T12:00:{seq:02d}Z",
        "session_id": session,
        "kind": kind,
        "actor": actor,
        "parent_ids": list(parent or []),
        "text": text,
        "payload": dict(payload or {}),
    }


def _fact(subject: str, predicate: str, value: Any, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"subject": subject, "predicate": predicate, "value": value}
    body.update(extra)
    return body


def _build() -> list[BenchCase]:
    cases: list[BenchCase] = []

    sid = "bm01_early_price"
    cases.append(
        BenchCase(
            id=sid,
            title="Early fare commitment, later fare unused",
            failure_mode="early_wrong_decision",
            description="The agent commits to a $420 fare before the search tool returns $880 and never cites it.",
            expect_detection=True,
            expect_claim="evt_003",
            expect_type="unused_counterevidence",
            expect_origin="evt_002",
            expect_evidence=["evt_005"],
            events=[
                _ev(sid, 1, "goal", "Book a nonstop SFO flight under 500 dollars."),
                _ev(sid, 2, "llm_call", "The SFO flight costs 420 dollars.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 3, "claim", "The SFO flight costs 420 dollars.", parent=["evt_002"], payload=_fact("flight", "price", 420)),
                _ev(sid, 4, "tool_call", "search_flights", payload={"tool_name": "search_flights", "call_id": "c1", "side_effect": "read", "arguments": {"to": "SFO"}}),
                _ev(sid, 5, "tool_result", "SFO nonstop fare is 880 dollars.", parent=["evt_004"], payload={**_fact("flight", "price", 880), "tool_name": "search_flights", "call_id": "c1", "ok": True, "side_effect": "read"}),
            ],
        )
    )

    sid = "bm02_ignored_tool"
    cases.append(
        BenchCase(
            id=sid,
            title="Inventory tool ignored",
            failure_mode="ignored_tool_result",
            description="Stock lookup returns 12. The next claim still asserts 100 and does not cite the tool.",
            expect_detection=True,
            expect_claim="evt_005",
            expect_type="unused_counterevidence",
            expect_origin="evt_004",
            expect_evidence=["evt_003"],
            events=[
                _ev(sid, 1, "goal", "Confirm we can ship 80 units."),
                _ev(sid, 2, "tool_call", "read_inventory", payload={"tool_name": "read_inventory", "call_id": "c1", "side_effect": "read"}),
                _ev(sid, 3, "tool_result", "Warehouse quantity is 12.", parent=["evt_002"], payload={**_fact("inventory", "quantity", 12), "tool_name": "read_inventory", "call_id": "c1", "ok": True, "side_effect": "read"}),
                _ev(sid, 4, "llm_call", "We have 100 units on hand.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 5, "claim", "Warehouse quantity is 100.", parent=["evt_004"], payload=_fact("inventory", "quantity", 100)),
            ],
        )
    )

    sid = "bm03_contract_refund"
    cases.append(
        BenchCase(
            id=sid,
            title="Refund contract violated",
            failure_mode="contract_violation",
            description="The task forbids refunds. The agent claims it issued one.",
            expect_detection=True,
            expect_claim="evt_003",
            expect_type="contract_violation",
            expect_origin="evt_002",
            expect_evidence=["evt_001"],
            events=[
                _ev(
                    sid,
                    1,
                    "contract",
                    "Do not issue a refund.",
                    actor="user",
                    payload={
                        "constraints": [
                            {"id": "c_refund", "modality": "must_not", "text": "Do not issue a refund.", "subject": "action", "predicate": "refund", "value": True}
                        ]
                    },
                ),
                _ev(sid, 2, "llm_call", "I issued a refund to the customer.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 3, "claim", "I issued a refund to the customer.", parent=["evt_002"], payload=_fact("action", "refund", True)),
            ],
        )
    )

    sid = "bm04_compound"
    cases.append(
        BenchCase(
            id=sid,
            title="Bad stock figure compounds into a fulfill decision",
            failure_mode="compounding",
            description="A wrong quantity claim is the parent of a fulfillable claim. Later stock evidence falsifies the root.",
            expect_detection=True,
            expect_claim="evt_002",
            expect_type="unused_counterevidence",
            expect_origin="evt_001",
            expect_evidence=["evt_004"],
            expect_also=[("evt_003", "compounded_error")],
            events=[
                _ev(sid, 1, "llm_call", "Warehouse quantity is 100.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 2, "claim", "Warehouse quantity is 100.", parent=["evt_001"], payload=_fact("warehouse", "quantity", 100)),
                _ev(sid, 3, "claim", "The order is fulfillable.", parent=["evt_002"], payload=_fact("order", "fulfillable", True)),
                _ev(sid, 4, "tool_result", "Warehouse quantity is 9.", payload={**_fact("warehouse", "quantity", 9), "tool_name": "read_inventory", "call_id": "c1", "ok": True, "side_effect": "read"}),
            ],
        )
    )

    sid = "bm05_happy"
    cases.append(
        BenchCase(
            id=sid,
            title="Cited fare matches the tool",
            failure_mode="happy",
            description="The claim cites the tool result and repeats its value.",
            expect_detection=False,
            events=[
                _ev(sid, 1, "goal", "Find the SFO fare."),
                _ev(sid, 2, "tool_result", "SFO fare is 880 dollars.", payload={**_fact("flight", "price", 880), "tool_name": "search_flights", "side_effect": "read", "ok": True}),
                _ev(sid, 3, "llm_call", "The SFO fare is 880 dollars.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 4, "claim", "The SFO fare is 880 dollars.", parent=["evt_003"], payload={**_fact("flight", "price", 880), "citations": ["evt_002"]}),
            ],
        )
    )

    sid = "bm06_resolved"
    cases.append(
        BenchCase(
            id=sid,
            title="Wrong fare is superseded using the tool",
            failure_mode="resolved",
            description="An early wrong claim is explicitly superseded by a later claim that cites the tool.",
            expect_detection=False,
            events=[
                _ev(sid, 1, "claim", "The fare is 420.", payload=_fact("flight", "price", 420)),
                _ev(sid, 2, "tool_result", "The fare is 880.", payload={**_fact("flight", "price", 880), "tool_name": "search_flights", "side_effect": "read", "ok": True}),
                _ev(sid, 3, "claim", "The fare is 880.", payload={**_fact("flight", "price", 880), "citations": ["evt_002"], "supersedes": ["evt_001"]}),
            ],
        )
    )

    sid = "bm07_status"
    cases.append(
        BenchCase(
            id=sid,
            title="Success claim, failure observation",
            failure_mode="early_wrong_decision",
            description="Payment is marked success. A later observation says failure and is not cited.",
            expect_detection=True,
            expect_claim="evt_003",
            expect_type="unused_counterevidence",
            expect_origin="evt_002",
            expect_evidence=["evt_004"],
            events=[
                _ev(sid, 1, "goal", "Report payment status."),
                _ev(sid, 2, "llm_call", "The payment status is success.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 3, "claim", "The payment status is success.", parent=["evt_002"], payload=_fact("payment", "status", "success")),
                _ev(sid, 4, "observation", "The payment status is failure.", actor="environment", payload=_fact("payment", "status", "failure")),
            ],
        )
    )

    sid = "bm08_must_format"
    cases.append(
        BenchCase(
            id=sid,
            title="Report format must be JSON",
            failure_mode="contract_violation",
            description="Acceptance criterion requires JSON. The agent claims PDF.",
            expect_detection=True,
            expect_claim="evt_003",
            expect_type="contract_violation",
            expect_origin="evt_002",
            expect_evidence=["evt_001"],
            events=[
                _ev(
                    sid,
                    1,
                    "contract",
                    "Report format must be JSON.",
                    actor="user",
                    payload={"constraints": [{"id": "c_fmt", "modality": "must", "text": "Report format must be JSON.", "subject": "report", "predicate": "format", "value": "json"}]},
                ),
                _ev(sid, 2, "llm_call", "I wrote the report as PDF.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 3, "claim", "Report format is PDF.", parent=["evt_002"], payload=_fact("report", "format", "pdf")),
            ],
        )
    )

    sid = "bm09_destructive"
    cases.append(
        BenchCase(
            id=sid,
            title="Wrong fare plus an email already sent",
            failure_mode="early_wrong_decision",
            description="After the bad commitment the agent sends email. Recovery must stay propose-only for the external effect.",
            expect_detection=True,
            expect_claim="evt_003",
            expect_type="unused_counterevidence",
            expect_origin="evt_002",
            expect_evidence=["evt_006"],
            expect_checkpoint="cp_009",
            expect_scope="external_blocked",
            events=[
                _ev(sid, 1, "checkpoint", "before search", payload={"checkpoint_id": "cp_009", "label": "start", "agent_context": {"step": "start"}, "environment": {"inbox": "empty"}}),
                _ev(sid, 2, "llm_call", "The fare is 420.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 3, "claim", "The fare is 420.", parent=["evt_002"], payload=_fact("flight", "price", 420)),
                _ev(sid, 4, "tool_call", "send_email", payload={"tool_name": "send_email", "call_id": "m1", "side_effect": "write", "arguments": {"body": "fare 420"}}),
                _ev(sid, 5, "tool_result", "Email sent.", parent=["evt_004"], payload={"tool_name": "send_email", "call_id": "m1", "ok": True, "side_effect": "write"}),
                _ev(sid, 6, "tool_result", "Actual fare is 880.", payload={**_fact("flight", "price", 880), "tool_name": "search_flights", "call_id": "c2", "ok": True, "side_effect": "read"}),
            ],
        )
    )

    sid = "bm10_internal"
    cases.append(
        BenchCase(
            id=sid,
            title="Two active claims disagree",
            failure_mode="internal_inconsistency",
            description="Nothing external contradicts the agent, but two live claims assign different statuses.",
            expect_detection=True,
            expect_claim="evt_001",
            expect_type="internal_inconsistency",
            expect_origin="evt_001",
            expect_evidence=["evt_002"],
            events=[
                _ev(sid, 1, "claim", "Build status is success.", payload=_fact("build", "status", "success")),
                _ev(sid, 2, "claim", "Build status is failure.", payload=_fact("build", "status", "failure")),
            ],
        )
    )

    sid = "bm11_human"
    cases.append(
        BenchCase(
            id=sid,
            title="Human correction left unused",
            failure_mode="ignored_tool_result",
            description="The user says Thursday. The agent keeps Friday and does not cite the user.",
            expect_detection=True,
            expect_claim="evt_003",
            expect_type="unused_counterevidence",
            expect_origin="evt_002",
            expect_evidence=["evt_004"],
            events=[
                _ev(sid, 1, "goal", "Set the filing day."),
                _ev(sid, 2, "llm_call", "The deadline day is Friday.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 3, "claim", "The deadline day is Friday.", parent=["evt_002"], payload=_fact("deadline", "day", "friday")),
                _ev(sid, 4, "human", "The deadline day is Thursday.", actor="user", payload=_fact("deadline", "day", "thursday")),
            ],
        )
    )

    sid = "bm12_multihop"
    cases.append(
        BenchCase(
            id=sid,
            title="Three-hop compounding",
            failure_mode="compounding",
            description="Quantity error parents fulfillable, which parents ship-today. Stock evidence arrives last.",
            expect_detection=True,
            expect_claim="evt_002",
            expect_type="unused_counterevidence",
            expect_origin="evt_001",
            expect_evidence=["evt_005"],
            expect_also=[("evt_003", "compounded_error"), ("evt_004", "compounded_error")],
            events=[
                _ev(sid, 1, "llm_call", "Quantity is 100.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 2, "claim", "Quantity is 100.", parent=["evt_001"], payload=_fact("warehouse", "quantity", 100)),
                _ev(sid, 3, "claim", "Order is fulfillable.", parent=["evt_002"], payload=_fact("order", "fulfillable", True)),
                _ev(sid, 4, "claim", "We ship today.", parent=["evt_003"], payload=_fact("shipment", "decision", "today")),
                _ev(sid, 5, "tool_result", "Quantity is 4.", payload={**_fact("warehouse", "quantity", 4), "tool_name": "read_inventory", "side_effect": "read", "ok": True}),
            ],
        )
    )

    sid = "bm13_cited_wrong"
    cases.append(
        BenchCase(
            id=sid,
            title="Evidence cited but not adopted",
            failure_mode="early_wrong_decision",
            description="The claim lists the tool in citations and still asserts the contradicted value.",
            expect_detection=True,
            expect_claim="evt_004",
            expect_type="direct_contradiction",
            expect_origin="evt_003",
            expect_evidence=["evt_002"],
            events=[
                _ev(sid, 1, "goal", "Price the fare."),
                _ev(sid, 2, "tool_result", "Fare is 880.", payload={**_fact("flight", "price", 880), "tool_name": "search_flights", "side_effect": "read", "ok": True}),
                _ev(sid, 3, "llm_call", "Fare is 420.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 4, "claim", "Fare is 420.", parent=["evt_003"], payload={**_fact("flight", "price", 420), "citations": ["evt_002"]}),
            ],
        )
    )

    sid = "bm14_checkpoint"
    cases.append(
        BenchCase(
            id=sid,
            title="Consistent checkpoint before the bad claim",
            failure_mode="early_wrong_decision",
            description="A checkpoint captures environment before the wrong commitment. Recovery should name it.",
            expect_detection=True,
            expect_claim="evt_003",
            expect_type="unused_counterevidence",
            expect_origin="evt_002",
            expect_evidence=["evt_004"],
            expect_checkpoint="cp_014",
            expect_scope="full_state",
            events=[
                _ev(sid, 1, "checkpoint", "clean", payload={"checkpoint_id": "cp_014", "label": "clean", "agent_context": {"claims": []}, "environment": {"inventory": 12}}),
                _ev(sid, 2, "llm_call", "Quantity is 100.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 3, "claim", "Quantity is 100.", parent=["evt_002"], payload=_fact("warehouse", "quantity", 100)),
                _ev(sid, 4, "observation", "Quantity is 12.", actor="environment", payload=_fact("warehouse", "quantity", 12)),
            ],
        )
    )

    sid = "bm15_diff_pred"
    cases.append(
        BenchCase(
            id=sid,
            title="Same subject, different predicates",
            failure_mode="happy",
            description="Price 420 and duration 90 share a subject and must not be treated as a contradiction.",
            expect_detection=False,
            events=[
                _ev(sid, 1, "claim", "Fare is 420.", payload=_fact("flight", "price", 420)),
                _ev(sid, 2, "tool_result", "Duration is 90.", payload={**_fact("flight", "duration", 90), "tool_name": "lookup_duration", "side_effect": "read", "ok": True}),
            ],
        )
    )

    sid = "bm16_present"
    cases.append(
        BenchCase(
            id=sid,
            title="Present versus missing",
            failure_mode="ignored_tool_result",
            description="The agent says the file is present. The tool says missing.",
            expect_detection=True,
            expect_claim="evt_002",
            expect_type="unused_counterevidence",
            expect_origin="evt_001",
            expect_evidence=["evt_003"],
            events=[
                _ev(sid, 1, "llm_call", "The file is present.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 2, "claim", "The file is present.", parent=["evt_001"], payload=_fact("file", "status", "present")),
                _ev(sid, 3, "tool_result", "The file is missing.", payload={**_fact("file", "status", "missing"), "tool_name": "stat_file", "side_effect": "read", "ok": True}),
            ],
        )
    )

    sid = "bm17_contract_ok"
    cases.append(
        BenchCase(
            id=sid,
            title="Refund forbidden and not issued",
            failure_mode="happy",
            description="must_not refund=true is respected by a claim that the refund value is false.",
            expect_detection=False,
            events=[
                _ev(
                    sid,
                    1,
                    "contract",
                    "Do not issue a refund.",
                    actor="user",
                    payload={"constraints": [{"id": "c_refund", "modality": "must_not", "text": "Do not issue a refund.", "subject": "action", "predicate": "refund", "value": True}]},
                ),
                _ev(sid, 2, "claim", "No refund was issued.", payload=_fact("action", "refund", False)),
            ],
        )
    )

    sid = "bm18_noise"
    cases.append(
        BenchCase(
            id=sid,
            title="One bad commitment among consistent side facts",
            failure_mode="early_wrong_decision",
            description="Weather is cited correctly. Destination is not. Only the destination claim is a latent error.",
            expect_detection=True,
            expect_claim="evt_003",
            expect_type="unused_counterevidence",
            expect_origin="evt_002",
            expect_evidence=["evt_006"],
            events=[
                _ev(sid, 1, "goal", "Plan the trip."),
                _ev(sid, 2, "llm_call", "Destination is Paris.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 3, "claim", "Destination is Paris.", parent=["evt_002"], payload=_fact("trip", "destination", "paris")),
                _ev(sid, 4, "tool_result", "Weather is clear.", payload={**_fact("weather", "status", "clear"), "tool_name": "lookup_weather", "side_effect": "read", "ok": True}),
                _ev(sid, 5, "claim", "Weather is clear.", parent=["evt_002"], payload={**_fact("weather", "status", "clear"), "citations": ["evt_004"]}),
                _ev(sid, 6, "tool_result", "Destination is London.", payload={**_fact("trip", "destination", "london"), "tool_name": "lookup_itinerary", "side_effect": "read", "ok": True}),
            ],
        )
    )

    sid = "bm19_door"
    cases.append(
        BenchCase(
            id=sid,
            title="Door observation contradicts an open claim",
            failure_mode="early_wrong_decision",
            description="Environment feedback says the door is closed after the agent claimed open.",
            expect_detection=True,
            expect_claim="evt_002",
            expect_type="unused_counterevidence",
            expect_origin="evt_001",
            expect_evidence=["evt_003"],
            events=[
                _ev(sid, 1, "llm_call", "The door is open.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 2, "claim", "The door is open.", parent=["evt_001"], payload=_fact("door", "status", "open")),
                _ev(sid, 3, "observation", "The door is closed.", actor="environment", payload=_fact("door", "status", "closed")),
            ],
        )
    )

    sid = "bm20_tool_failure"
    cases.append(
        BenchCase(
            id=sid,
            title="Deploy status claim ignores a failed lookup",
            failure_mode="ignored_tool_result",
            description="Lookup returns failure. The claim, produced after the tool, still says success.",
            expect_detection=True,
            expect_claim="evt_004",
            expect_type="unused_counterevidence",
            expect_origin="evt_003",
            expect_evidence=["evt_002"],
            events=[
                _ev(sid, 1, "tool_call", "get_status", payload={"tool_name": "get_status", "call_id": "d1", "side_effect": "read"}),
                _ev(sid, 2, "tool_result", "Deploy status is failure.", parent=["evt_001"], payload={**_fact("deploy", "status", "failure"), "tool_name": "get_status", "call_id": "d1", "ok": False, "side_effect": "read"}),
                _ev(sid, 3, "llm_call", "Deploy status is success.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 4, "claim", "Deploy status is success.", parent=["evt_003"], payload=_fact("deploy", "status", "success")),
            ],
        )
    )

    sid = "bm21_earliest"
    cases.append(
        BenchCase(
            id=sid,
            title="Origin is the producing LLM, not an earlier unrelated turn",
            failure_mode="early_wrong_decision",
            description="An earlier LLM turn is only a parent of the producer. The bad value enters at the second LLM call.",
            expect_detection=True,
            expect_claim="evt_004",
            expect_type="unused_counterevidence",
            expect_origin="evt_003",
            expect_evidence=["evt_005"],
            events=[
                _ev(sid, 1, "goal", "Price the item."),
                _ev(sid, 2, "llm_call", "I will look up the price.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 3, "llm_call", "The price is 10.", parent=["evt_002"], payload={"model": "stub", "temperature": 0}),
                _ev(sid, 4, "claim", "The price is 10.", parent=["evt_003"], payload=_fact("item", "price", 10)),
                _ev(sid, 5, "tool_result", "The price is 99.", payload={**_fact("item", "price", 99), "tool_name": "lookup_price", "side_effect": "read", "ok": True}),
            ],
        )
    )

    sid = "bm22_retracted"
    cases.append(
        BenchCase(
            id=sid,
            title="Retracted claim is not a latent error",
            failure_mode="resolved",
            description="The contradicted claim is already marked retracted.",
            expect_detection=False,
            events=[
                _ev(sid, 1, "claim", "The fare is 420.", payload={**_fact("flight", "price", 420), "status": "retracted"}),
                _ev(sid, 2, "tool_result", "The fare is 880.", payload={**_fact("flight", "price", 880), "tool_name": "search_flights", "side_effect": "read", "ok": True}),
            ],
        )
    )

    sid = "bm23_goal"
    cases.append(
        BenchCase(
            id=sid,
            title="A goal is not a claim",
            failure_mode="happy",
            description="The goal mentions a number. A tool result with a different number must not be scored as the goal contradicting itself.",
            expect_detection=False,
            events=[
                _ev(sid, 1, "goal", "Keep the budget at 100 dollars."),
                _ev(sid, 2, "tool_result", "Quoted total is 900.", payload={**_fact("quote", "price", 900), "tool_name": "lookup_quote", "side_effect": "read", "ok": True}),
            ],
        )
    )

    sid = "bm24_equal"
    cases.append(
        BenchCase(
            id=sid,
            title="Equal numeric values",
            failure_mode="happy",
            description="15 and 15.0 are the same fact.",
            expect_detection=False,
            events=[
                _ev(sid, 1, "claim", "Count is 15.", payload=_fact("batch", "quantity", 15)),
                _ev(sid, 2, "tool_result", "Count is 15.", payload={**_fact("batch", "quantity", 15.0), "tool_name": "count_batch", "side_effect": "read", "ok": True}),
            ],
        )
    )

    sid = "bm25_human_agrees"
    cases.append(
        BenchCase(
            id=sid,
            title="Human message agrees",
            failure_mode="happy",
            description="User confirmation of the same status is support, not a contradiction.",
            expect_detection=False,
            events=[
                _ev(sid, 1, "claim", "Door is open.", payload=_fact("door", "status", "open")),
                _ev(sid, 2, "human", "Door is open.", actor="user", payload=_fact("door", "status", "open")),
            ],
        )
    )

    sid = "bm26_route"
    cases.append(
        BenchCase(
            id=sid,
            title="Nonstop claim, connecting itinerary",
            failure_mode="ignored_tool_result",
            description="Route values are mutually exclusive. The tool result is unused.",
            expect_detection=True,
            expect_claim="evt_002",
            expect_type="unused_counterevidence",
            expect_origin="evt_001",
            expect_evidence=["evt_003"],
            events=[
                _ev(sid, 1, "llm_call", "The route is nonstop.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 2, "claim", "The route is nonstop.", parent=["evt_001"], payload=_fact("flight", "route", "nonstop")),
                _ev(sid, 3, "tool_result", "The route is connecting.", payload={**_fact("flight", "route", "connecting"), "tool_name": "search_flights", "side_effect": "read", "ok": True}),
            ],
        )
    )

    sid = "bm27_class_ok"
    cases.append(
        BenchCase(
            id=sid,
            title="Forbidden cabin avoided",
            failure_mode="happy",
            description="must_not class=first is respected by an economy claim.",
            expect_detection=False,
            events=[
                _ev(
                    sid,
                    1,
                    "contract",
                    "Do not book first class.",
                    actor="user",
                    payload={"constraints": [{"id": "c_class", "modality": "must_not", "text": "Do not book first class.", "subject": "ticket", "predicate": "class", "value": "first"}]},
                ),
                _ev(sid, 2, "claim", "Cabin is economy.", payload=_fact("ticket", "class", "economy")),
            ],
        )
    )

    sid = "bm28_one_of_three"
    cases.append(
        BenchCase(
            id=sid,
            title="Two irrelevant results, one contradiction",
            failure_mode="ignored_tool_result",
            description="Duration and weather do not touch the price claim. The price tool does.",
            expect_detection=True,
            expect_claim="evt_002",
            expect_type="unused_counterevidence",
            expect_origin="evt_001",
            expect_evidence=["evt_004"],
            events=[
                _ev(sid, 1, "llm_call", "Price is 50.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 2, "claim", "Price is 50.", parent=["evt_001"], payload=_fact("flight", "price", 50)),
                _ev(sid, 3, "tool_result", "Duration is 80.", payload={**_fact("flight", "duration", 80), "tool_name": "lookup_duration", "side_effect": "read", "ok": True}),
                _ev(sid, 4, "tool_result", "Price is 75.", payload={**_fact("flight", "price", 75), "tool_name": "lookup_price", "side_effect": "read", "ok": True}),
                _ev(sid, 5, "observation", "Weather is clear.", actor="environment", payload=_fact("weather", "status", "clear")),
            ],
        )
    )

    sid = "bm29_supersede"
    cases.append(
        BenchCase(
            id=sid,
            title="Supersede without an explicit citation",
            failure_mode="resolved",
            description="The corrected claim supersedes the bad one and matches the tool, so nothing active is latent.",
            expect_detection=False,
            events=[
                _ev(sid, 1, "claim", "Fare is 420.", payload=_fact("flight", "price", 420)),
                _ev(sid, 2, "tool_result", "Fare is 880.", payload={**_fact("flight", "price", 880), "tool_name": "search_flights", "side_effect": "read", "ok": True}),
                _ev(sid, 3, "claim", "Fare is 880.", payload={**_fact("flight", "price", 880), "supersedes": ["evt_001"]}),
            ],
        )
    )

    sid = "bm30_obs_first"
    cases.append(
        BenchCase(
            id=sid,
            title="Observation available before the commitment",
            failure_mode="ignored_tool_result",
            description="The closed-door observation is already in the trajectory when the agent claims the door is open.",
            expect_detection=True,
            expect_claim="evt_004",
            expect_type="unused_counterevidence",
            expect_origin="evt_003",
            expect_evidence=["evt_002"],
            events=[
                _ev(sid, 1, "goal", "Check the door."),
                _ev(sid, 2, "observation", "The door is closed.", actor="environment", payload=_fact("door", "status", "closed")),
                _ev(sid, 3, "llm_call", "The door is open.", payload={"model": "stub", "temperature": 0}),
                _ev(sid, 4, "claim", "The door is open.", parent=["evt_003"], payload=_fact("door", "status", "open")),
            ],
        )
    )

    return cases


CASES: list[BenchCase] = _build()
