"""Contradiction judges.

The fast path accepts high-confidence structural contradictions directly.
The slow path runs only for soft candidates. The default judge is a
deterministic heuristic (temperature 0, model name logged). Inject
`LLMJudge` with a completion callable to use a real model; its prompt hash,
model id, and temperature are stored on the detection.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from evidenceorbit.canonical import content_hash
from evidenceorbit.embed import Embedder, HashingEmbedder, cosine
from evidenceorbit.models import JudgeTrace

_NEGATION = re.compile(
    r"\b(not|never|no|cannot|can't|didn't|did not|without|n't|unable)\b",
    re.IGNORECASE,
)


def text_is_negative(text: str) -> bool:
    return _NEGATION.search(text) is not None


class Judge(Protocol):
    model: str
    temperature: float

    def judge(self, claim_text: str, evidence_text: str) -> JudgeTrace: ...


def _trace(
    *,
    model: str,
    temperature: float,
    claim_text: str,
    evidence_text: str,
    contradicts: bool,
    confidence: float,
    explanation: str,
) -> JudgeTrace:
    prompt = {"task": "contradiction_judge", "claim": claim_text, "evidence": evidence_text}
    return JudgeTrace(
        model=model,
        prompt_hash=content_hash(prompt),
        temperature=temperature,
        contradicts=contradicts,
        confidence=confidence,
        explanation=explanation,
    )


class HeuristicJudge:
    """Deterministic stand-in. Used when no LLM client is configured."""

    model = "evidenceorbit-heuristic-v1"
    temperature = 0.0

    def __init__(self, embedder: Embedder | None = None, cosine_threshold: float = 0.55) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.cosine_threshold = cosine_threshold

    def judge(self, claim_text: str, evidence_text: str) -> JudgeTrace:
        claim_neg = text_is_negative(claim_text)
        evidence_neg = text_is_negative(evidence_text)
        polarity_differs = claim_neg != evidence_neg
        similarity = cosine(self.embedder.embed(claim_text), self.embedder.embed(evidence_text))
        claim_tokens = set(re.findall(r"[a-z0-9]+", claim_text.lower()))
        evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence_text.lower()))
        shared = claim_tokens & evidence_tokens
        stop = {"the", "a", "an", "is", "was", "were", "to", "of", "and", "or", "not", "did", "do"}
        shared_content = shared - stop
        contradicts = polarity_differs and similarity >= self.cosine_threshold and len(shared_content) >= 2
        confidence = 0.66 if contradicts else 0.2
        if contradicts:
            explanation = (
                f"Heuristic judge: polarity differs, cosine {similarity:.2f}, "
                f"shared tokens {sorted(shared_content)}."
            )
        else:
            explanation = (
                f"Heuristic judge: not a contradiction "
                f"(polarity_differs={polarity_differs}, cosine {similarity:.2f})."
            )
        return _trace(
            model=self.model,
            temperature=self.temperature,
            claim_text=claim_text,
            evidence_text=evidence_text,
            contradicts=contradicts,
            confidence=confidence,
            explanation=explanation,
        )


class LLMJudge:
    """Slow path. `complete` must return a JSON object as text.

    Expected keys: contradicts (bool), confidence (float), explanation (str).
    The callable is injected so the library never opens a network connection
    on its own. temperature is recorded, not enforced — callers pass it to
    their provider.
    """

    def __init__(
        self,
        complete: object,
        *,
        model: str,
        temperature: float = 0.0,
    ) -> None:
        if not callable(complete):
            raise TypeError("complete must be callable")
        self._complete = complete
        self.model = model
        self.temperature = temperature

    def judge(self, claim_text: str, evidence_text: str) -> JudgeTrace:
        prompt = (
            "Does the evidence contradict the claim? "
            'Reply JSON {"contradicts": bool, "confidence": float, "explanation": str}.\n'
            f"CLAIM: {claim_text}\nEVIDENCE: {evidence_text}"
        )
        raw = self._complete(prompt)
        parsed = _parse_verdict(str(raw))
        trace = _trace(
            model=self.model,
            temperature=self.temperature,
            claim_text=claim_text,
            evidence_text=evidence_text,
            contradicts=bool(parsed["contradicts"]),
            confidence=float(parsed["confidence"]) if isinstance(parsed["confidence"], (int, float)) else 0.0,
            explanation=str(parsed["explanation"]),
        )
        return trace


def _parse_verdict(raw: str) -> dict[str, object]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        return {"contradicts": False, "confidence": 0.0, "explanation": "Judge returned non-JSON."}
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {"contradicts": False, "confidence": 0.0, "explanation": "Judge JSON failed to parse."}
    if not isinstance(data, dict):
        return {"contradicts": False, "confidence": 0.0, "explanation": "Judge JSON was not an object."}
    confidence = data.get("confidence", 0.0)
    confidence_f = 0.0
    if isinstance(confidence, (int, float, str)) and not isinstance(confidence, bool):
        try:
            confidence_f = max(0.0, min(1.0, float(confidence)))
        except ValueError:
            confidence_f = 0.0
    return {
        "contradicts": bool(data.get("contradicts")),
        "confidence": confidence_f,
        "explanation": str(data.get("explanation") or "LLM judge verdict."),
    }
