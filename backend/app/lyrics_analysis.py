"""Structured, derived lyrical meaning and candidate verification."""

import json
import logging
import os
from typing import Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from .models import LyricsMeaning, Song


logger = logging.getLogger(__name__)

LYRICS_MEANING_VERSION = 1
GPT_4O_MINI_INPUT_PRICE = 0.15
GPT_4O_MINI_OUTPUT_PRICE = 0.60


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def estimated_cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * GPT_4O_MINI_INPUT_PRICE
            + self.output_tokens / 1_000_000 * GPT_4O_MINI_OUTPUT_PRICE
        )


class LyricsMeaningExtractor:
    """Convert temporary raw lyrics into a compact non-quoting analysis."""

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env; never commit or paste it.")
        self.model = model or os.getenv("LYRICS_ANALYSIS_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key, timeout=45.0, max_retries=2)

    def extract(self, title: str, artist: str, lyrics: str) -> tuple[LyricsMeaning, Usage]:
        response = self.client.responses.parse(
            model=self.model,
            instructions=(
                "Analyze only the supplied lyrics. Produce a concise, abstract meaning record for music "
                "retrieval. Do not quote, reproduce, or closely paraphrase lyric lines. Identify the actual "
                "story, relationship dynamics, narrator perspective, and emotional progression. Distinguish "
                "specific situations such as unrequited love, infidelity, forbidden attraction, grief, and "
                "reconciliation instead of collapsing them into generic love or sadness. If the text does "
                "not establish a detail, omit it rather than guessing from the title or artist."
            ),
            input=f"Title: {title}\nArtist: {artist}\n\nLyrics:\n{lyrics}",
            text_format=LyricsMeaning,
            max_output_tokens=350,
            temperature=0,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError("The lyrics analysis model returned no structured result")
        usage = Usage(
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
        )
        return response.output_parsed, usage


class CandidateVerdict(BaseModel):
    song_id: str
    verdict: Literal["match", "partial", "no_match"]
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=3, max_length=220)


class RequiredClaim(BaseModel):
    claim_id: str = Field(min_length=1, max_length=30)
    description: str = Field(min_length=3, max_length=180)


class ClaimCheck(BaseModel):
    claim_id: str
    support: Literal["supported", "not_stated", "contradicted"]
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list, max_length=5)


class CandidateAssessment(BaseModel):
    song_id: str
    checks: list[ClaimCheck]
    reason: str = Field(min_length=3, max_length=220)


class VerificationAssessment(BaseModel):
    required_claims: list[RequiredClaim] = Field(min_length=1, max_length=4)
    candidates: list[CandidateAssessment]


class LyricsVerifier(Protocol):
    def verify(self, request: str, songs: list[Song]) -> dict[str, CandidateVerdict]: ...


class OpenAILyricsVerifier:
    """Verify a small retrieved set against stored derived meaning records."""

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env; never commit or paste it.")
        self.model = model or os.getenv("LYRICS_VERIFIER_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)

    def verify(self, request: str, songs: list[Song]) -> dict[str, CandidateVerdict]:
        candidates = [
            {
                "song_id": song.id,
                "evidence": self._meaning_evidence(song.lyrics_meaning),
            }
            for song in songs
            if song.lyrics_meaning is not None
        ]
        if not candidates:
            return {}
        try:
            response = self.client.responses.parse(
                model=self.model,
                instructions=(
                    "First split the user's request into 1 to 4 independently verifiable REQUIRED claims. Preserve "
                    "every defining qualifier, relationship, perspective, and barrier. For example, 'intense "
                    "forbidden attraction' requires attraction, intensity, and an actual obstacle or prohibition; "
                    "ordinary attraction is insufficient. Then check every required claim against every supplied "
                    "meaning record. Mark supported only when the record explicitly establishes that claim. Use "
                    "not_stated when it is absent and contradicted when incompatible. Do not infer missing facts, "
                    "do not use titles or outside song knowledge, and do not rewrite a candidate record into a more "
                    "favorable story. Each candidate contains evidence entries with stable IDs. For every supported "
                    "check, return one or more evidence_ids copied exactly from that candidate. Use an empty list for "
                    "not_stated or contradicted. The cited evidence must establish the requested situation; mere shared "
                    "emotion does not. Return exactly one check per required claim for every supplied song_id."
                ),
                input=json.dumps({"request": request, "candidates": candidates}, ensure_ascii=False),
                text_format=VerificationAssessment,
                max_output_tokens=3000,
                temperature=0,
                store=False,
            )
        except ValidationError as error:
            raise RuntimeError(
                "The verifier response was incomplete; no candidates were accepted and no retry was made."
            ) from error
        assessment = response.output_parsed
        if assessment is None:
            raise RuntimeError("The lyrics verifier returned no structured result")
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            logger.info(
                "lyrics_verifier model=%s candidates=%s estimated_cost_usd=%.8f",
                self.model,
                len(candidates),
                usage.estimated_cost_usd(),
            )
        evidence_ids_by_song_id = {
            candidate["song_id"]: set(candidate["evidence"])
            for candidate in candidates
        }
        return self._derive_verdicts(
            assessment,
            {candidate["song_id"] for candidate in candidates},
            evidence_ids_by_song_id,
        )

    @staticmethod
    def _meaning_evidence(meaning: LyricsMeaning) -> dict[str, str]:
        """Expose derived meaning as addressable facts without fragile quote matching."""
        evidence = {"summary": meaning.summary}
        evidence.update({f"themes.{index}": value for index, value in enumerate(meaning.themes)})
        evidence.update({
            f"relationship_dynamics.{index}": value
            for index, value in enumerate(meaning.relationship_dynamics)
        })
        if meaning.narrator_perspective:
            evidence["narrator_perspective"] = meaning.narrator_perspective
        if meaning.emotional_arc:
            evidence["emotional_arc"] = meaning.emotional_arc
        return evidence

    @staticmethod
    def _derive_verdicts(
        assessment: VerificationAssessment,
        allowed_ids: set[str],
        evidence_ids_by_song_id: dict[str, set[str]] | None = None,
    ) -> dict[str, CandidateVerdict]:
        """Compute the gate in code so the model cannot waive a required claim."""
        required_ids = {claim.claim_id for claim in assessment.required_claims}
        verdicts: dict[str, CandidateVerdict] = {}
        for candidate in assessment.candidates:
            if candidate.song_id not in allowed_ids:
                continue
            checks = {check.claim_id: check for check in candidate.checks}
            if evidence_ids_by_song_id is not None:
                allowed_evidence_ids = evidence_ids_by_song_id.get(candidate.song_id, set())
                for check in checks.values():
                    if check.support != "supported":
                        continue
                    cited_ids = set(check.evidence_ids)
                    if not cited_ids or not cited_ids.issubset(allowed_evidence_ids):
                        check.support = "not_stated"
                        check.confidence = 1.0
                        check.evidence_ids = []
            required_checks = [checks.get(claim_id) for claim_id in required_ids]
            supported = [
                check
                for check in required_checks
                if check is not None and check.support == "supported"
            ]
            all_supported = len(supported) == len(required_ids)
            if all_supported:
                verdict = "match"
                confidence = min(check.confidence for check in supported)
            elif supported:
                verdict = "partial"
                confidence = min(check.confidence for check in supported)
            else:
                verdict = "no_match"
                present = [check.confidence for check in required_checks if check is not None]
                confidence = max(present, default=0.0)
            verdicts[candidate.song_id] = CandidateVerdict(
                song_id=candidate.song_id,
                verdict=verdict,
                confidence=confidence,
                reason=candidate.reason,
            )
        return verdicts
