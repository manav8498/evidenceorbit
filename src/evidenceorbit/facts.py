"""Fact extraction and contradiction rules.

Structured payload fields win. Free text is a deterministic fallback so the
fast path never needs a model. Exclusive structured predicates contradict on
any value mismatch; free-text mismatches contradict only for numbers or known
antonym groups, which keeps the happy path quiet.
"""

from __future__ import annotations

import re
from typing import Any

from evidenceorbit.models import EventKind, Fact, TrajectoryEvent

_PREDICATE_ALIASES = {
    "price": "price",
    "price_usd": "price",
    "cost": "price",
    "costs": "price",
    "fare": "price",
    "quantity": "quantity",
    "qty": "quantity",
    "count": "quantity",
    "inventory": "quantity",
    "stock": "quantity",
    "total": "quantity",
    "balance": "balance",
    "limit": "limit",
    "status": "status",
    "state": "status",
    "result": "status",
    "refund": "refund",
    "fulfillable": "fulfillable",
    "class": "class",
    "format": "format",
    "day": "day",
    "decision": "decision",
    "destination": "destination",
    "route": "route",
    "owner": "owner",
    "mode": "mode",
    "priority": "priority",
    "eta": "eta",
    "duration": "duration",
    "is": "is",
}

_VALUE_SYNONYMS = {
    "failed": "failure",
    "error": "failure",
    "ok": "success",
    "succeeded": "success",
    "canceled": "cancelled",
}

# Values in the same group are mutually exclusive when asserted for one field.
_ANTONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"nonstop", "connecting", "layover", "direct", "indirect"}),
    frozenset({"success", "failure"}),
    frozenset({"true", "false"}),
    frozenset({"available", "unavailable"}),
    frozenset({"present", "missing", "absent"}),
    frozenset({"yes", "no"}),
    frozenset({"open", "closed"}),
    frozenset({"allowed", "forbidden", "denied"}),
    frozenset({"valid", "invalid"}),
    frozenset({"found", "not_found"}),
    frozenset({"confirmed", "unconfirmed", "cancelled"}),
    frozenset({"pass", "fail"}),
    frozenset({"enabled", "disabled"}),
    frozenset({"safe", "unsafe"}),
    frozenset({"complete", "incomplete"}),
    frozenset({"friday", "thursday", "monday", "tuesday", "wednesday", "saturday", "sunday"}),
    frozenset({"json", "pdf", "xml", "html", "csv"}),
    frozenset({"first", "economy", "business"}),
)

_FACT_RE = re.compile(
    r"\b(?:the\s+|a\s+|an\s+)?(?P<subj>[a-z][\w-]{2,}(?:\s+[a-z][\w-]{2,})?)\s+"
    r"(?P<neg>not\s+|never\s+)?(?P<pred>costs?|priced at|equals?|is|are|was|were)\s+"
    r"(?P<neg2>not\s+)?(?P<val>\$?\d+(?:\.\d+)?|[a-z][\w-]{2,})",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_subject(value: str) -> str:
    text = value.strip().lower().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    for article in ("the ", "a ", "an "):
        if text.startswith(article):
            text = text[len(article) :]
    return text.strip()


def normalize_predicate(value: str) -> str:
    key = value.strip().lower().replace(" ", "_")
    return _PREDICATE_ALIASES.get(key, key)


def normalize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, "g")
    text = str(value).strip().lower()
    text = text.replace(",", "")
    if text.startswith("$"):
        text = text[1:]
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        number = float(text)
        if number.is_integer():
            return str(int(number))
        return format(number, "g")
    text = text.replace(" ", "_")
    return _VALUE_SYNONYMS.get(text, text)


def _is_number(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", value))


def are_antonyms(left: str, right: str) -> bool:
    if left == right:
        return False
    for group in _ANTONYM_GROUPS:
        if left in group and right in group:
            return True
    return False


def values_equal(left: str, right: str) -> bool:
    return left == right


def make_fact(
    subject: str,
    predicate: str,
    value: Any,
    *,
    polarity: str = "assert",
    source: str = "structured",
) -> Fact:
    pol: Any = "deny" if polarity == "deny" else "assert"
    src: Any = "text" if source == "text" else "structured"
    return Fact(
        subject=normalize_subject(subject),
        predicate=normalize_predicate(predicate),
        value=normalize_value(value),
        polarity=pol,
        source=src,
    )


def extract_text_facts(text: str) -> list[Fact]:
    found: list[Fact] = []
    for match in _FACT_RE.finditer(text):
        pred_raw = match.group("pred").lower()
        predicate = "price" if pred_raw.startswith("cost") or pred_raw.startswith("priced") else "is"
        if pred_raw.startswith("equal"):
            predicate = "is"
        value = match.group("val")
        if predicate == "is" and _is_number(normalize_value(value)):
            predicate = "quantity"
        polarity = "deny" if (match.group("neg") or match.group("neg2")) else "assert"
        found.append(
            make_fact(match.group("subj"), predicate, value, polarity=polarity, source="text")
        )
    return _dedupe(found)


def facts_from_event(event: TrajectoryEvent) -> list[Fact]:
    """Structured facts if present; otherwise text. Goals are not claims."""
    if event.kind in {EventKind.GOAL, EventKind.CONTRACT, EventKind.CHECKPOINT, EventKind.TOOL_CALL}:
        return []
    if event.kind == EventKind.SYSTEM:
        return []
    payload = event.payload
    structured: list[Fact] = []
    if payload.get("subject") is not None and "value" in payload:
        structured.append(
            make_fact(
                str(payload.get("subject")),
                str(payload.get("predicate") or "status"),
                payload.get("value"),
                polarity=str(payload.get("polarity") or "assert"),
                source="structured",
            )
        )
    extra = payload.get("facts")
    if isinstance(extra, list):
        for item in extra:
            if isinstance(item, dict) and item.get("subject") is not None and "value" in item:
                structured.append(
                    make_fact(
                        str(item.get("subject")),
                        str(item.get("predicate") or "status"),
                        item.get("value"),
                        polarity=str(item.get("polarity") or "assert"),
                        source="structured",
                    )
                )
    if structured:
        return _dedupe(structured)
    if event.text:
        return extract_text_facts(event.text)
    return []


def facts_contradict(left: Fact, right: Fact) -> bool:
    if left.subject != right.subject or left.predicate != right.predicate:
        return False
    if left.polarity != right.polarity:
        # "is X" vs "is not X" only. Deny(success) vs assert(failure) is consistent.
        return values_equal(left.value, right.value)
    if left.polarity == "deny":
        return False
    if values_equal(left.value, right.value):
        return False
    if left.source == "structured" or right.source == "structured":
        return True
    if _is_number(left.value) and _is_number(right.value):
        return True
    return are_antonyms(left.value, right.value)


def facts_agree(left: list[Fact], right: list[Fact]) -> bool:
    for a in left:
        for b in right:
            if (
                a.subject == b.subject
                and a.predicate == b.predicate
                and a.polarity == b.polarity == "assert"
                and values_equal(a.value, b.value)
            ):
                return True
    return False


def first_contradiction(left: list[Fact], right: list[Fact]) -> tuple[Fact, Fact] | None:
    for a in left:
        for b in right:
            if facts_contradict(a, b):
                return a, b
    return None


def constraint_facts(subject: str | None, predicate: str | None, value: Any) -> Fact | None:
    if subject is None or value is None:
        return None
    return make_fact(subject, predicate or "status", value, polarity="assert", source="structured")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _dedupe(facts: list[Fact]) -> list[Fact]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[Fact] = []
    for fact in facts:
        key = (fact.subject, fact.predicate, fact.value, fact.polarity)
        if key in seen:
            continue
        seen.add(key)
        out.append(fact)
    return out
