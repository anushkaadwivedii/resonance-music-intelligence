"""LLM-backed structured music-intent extraction."""

import logging
import os
from pathlib import Path
from typing import Literal, Protocol

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from .models import Intent, SignalWeights


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

LLM_PRICES_PER_MILLION_TOKENS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


class ExtractedIntent(BaseModel):
    search_description: str = Field(min_length=3, max_length=500)
    lyrics_search_description: str | None = Field(default=None, min_length=3, max_length=500)
    desired_lyrical_themes: list[str]
    avoid_lyrical_themes: list[str]
    lyrics_required: bool
    avoid_sound: list[str]
    signal_weights: SignalWeights
    moods: list[Literal[
        "balanced", "calm", "energetic", "intimate", "joyful",
        "melancholic", "playful", "romantic", "dreamy", "reflective",
    ]]
    genres: list[str]
    contexts: list[str]
    excluded_genres: list[str]
    title_contains: str | None
    artist_reference: str | None
    bpm_min: int | None = Field(ge=30, le=250)
    bpm_max: int | None = Field(ge=30, le=250)
    bpm_is_explicit: bool
    valence_target: float | None = Field(ge=0, le=1)
    energy_target: float | None = Field(ge=0, le=1)
    danceability_target: float | None = Field(ge=0, le=1)
    acousticness_target: float | None = Field(ge=0, le=1)
    instrumentalness_target: float | None = Field(ge=0, le=1)


class IntentParser(Protocol):
    def parse(self, query: str) -> Intent: ...


class OpenAIIntentParser:
    """Use Structured Outputs so free-form model text never enters retrieval."""

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env; never commit or paste it.")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)

    def parse(self, query: str) -> Intent:
        response = self.client.responses.parse(
            model=self.model,
            instructions=(
                "Extract music-search intent. Do not recommend songs. Return a vivid, meaning-based "
                "search_description that expresses the desired SOUND without relying on matching words "
                "in song titles or artist names. Separately extract desired_lyrical_themes and "
                "avoid_lyrical_themes; these describe what the words of the song should or should not be about. "
                "When lyrical content is explicitly requested, write lyrics_search_description as a detailed, "
                "meaning-preserving description of the requested lyrical story. Preserve relationships and "
                "perspective—for example, 'longing for someone who is now with somebody else'—rather than "
                "reducing it to generic tags. Exclude production, genre, tempo, and other sound instructions. "
                "Otherwise set lyrics_search_description to null. "
                "Set lyrics_required true only when lyrical content is an explicit requirement, such as "
                "'songs about forgiveness', 'lyrics about home', or 'without childish lyrics'. Do not require "
                "lyrics for ordinary sound or mood requests such as 'happy music'. "
                "Put unwanted sonic qualities in avoid_sound. Set title_contains ONLY when the user explicitly "
                "asks for a word or phrase in the song title (for example, 'songs with happy in the title'). "
                "Do not infer title_contains from an ordinary mood request like 'happy songs'. Set "
                "artist_reference only when the user explicitly names or refers to an artist. Use only the "
                "allowed mood labels. Normalize settings such as studying to "
                "study and gym to workout. Infer 0-to-1 audio-feature targets when useful. Preserve explicit "
                "Set signal_weights from 0 to 1 based on how important each signal is in THIS request; use 0 "
                "when the signal is not requested. Semantic should normally be nonzero. Lyrics means lyrical "
                "subject matter, not vocal sound. popularity_tiebreak must stay subtle: normally 0.02 to 0.05, "
                "up to 0.15 only when the user explicitly asks for popular or familiar music. "
                "genre exclusions. Only set bpm_is_explicit true when the user gives a numeric BPM constraint; "
                "do not invent numeric BPM bounds. Use null for unsupported or unspecified targets."
            ),
            input=query,
            text_format=ExtractedIntent,
            max_output_tokens=500,
            temperature=0,
            store=False,
        )
        extracted = response.output_parsed
        if extracted is None:
            raise RuntimeError("The intent model returned no structured intent")

        usage = response.usage
        price = LLM_PRICES_PER_MILLION_TOKENS.get(self.model)
        if usage and price:
            cost = (
                usage.input_tokens / 1_000_000 * price["input"]
                + usage.output_tokens / 1_000_000 * price["output"]
            )
            logger.info(
                "intent_parser model=%s input_tokens=%s output_tokens=%s estimated_cost_usd=%.8f",
                self.model,
                usage.input_tokens,
                usage.output_tokens,
                cost,
            )
        return Intent(**extracted.model_dump())
