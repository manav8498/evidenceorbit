"""Minimal in-process loop. Run from the repo root with the package installed."""

from evidenceorbit import EvidenceOrbit, ReActAdapter


def main() -> None:
    loop = ReActAdapter(EvidenceOrbit(), session_id="example")
    loop.checkpoint("start", {"goal": "book sfo"}, {"booked": False}, checkpoint_id="cp_example")
    loop.goal("Book a nonstop SFO flight under 500 dollars.")
    loop.llm(
        "What is the fare?",
        "The SFO flight costs 420 dollars.",
        model="example-stub",
        temperature=0.0,
    )
    producer = loop.last_event_id()
    loop.claim(
        "The SFO flight costs 420 dollars.",
        parent_ids=[producer] if producer else None,
        subject="flight",
        predicate="price",
        value=420,
    )
    found = loop.tool(
        "search_flights",
        {"to": "SFO"},
        "SFO nonstop fare is 880 dollars.",
        payload={"subject": "flight", "predicate": "price", "value": 880},
    )
    print(f"new detections: {len(found)}")
    for detection in loop.orbit.latent_errors():
        attr = loop.orbit.report.attribution_for(detection.id)
        proposal = loop.orbit.report.proposal_for(detection.id)
        print(detection.type, detection.explanation)
        if attr:
            print("origin", attr.origin_event_id, attr.propagation_summary)
        if proposal:
            print("rewind", proposal.rewind_seq, proposal.restore_scope)
            print(proposal.corrective_signal.instruction)
            refused = loop.orbit.apply_recovery(proposal.id, confirm=False)
            print("apply without confirm:", refused.applied, refused.reason)


if __name__ == "__main__":
    main()
