from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Song(BaseModel):
    id: str
    title: str
    artist: str
    album: str | None = None
    genre: str
    genres: list[str] = []
    moods: list[str]
    contexts: list[str]
    bpm: float
    perceived_bpm: float | None = None
    year: int | None = None
    description: str
    accent: str
    popularity: int | None = None
    energy: float | None = None
    danceability: float | None = None
    valence: float | None = None
    acousticness: float | None = None
    instrumentalness: float | None = None
    lyrics_evidence: Literal["analyzed", "unavailable", "not_analyzed"] = "not_analyzed"


class SignalWeights(BaseModel):
    """Query-specific priorities. Retrieval normalizes applicable signals."""

    semantic: float = Field(default=1.0, ge=0, le=1)
    mood: float = Field(default=1.0, ge=0, le=1)
    audio: float = Field(default=1.0, ge=0, le=1)
    context: float = Field(default=1.0, ge=0, le=1)
    tempo: float = Field(default=1.0, ge=0, le=1)
    genre: float = Field(default=1.0, ge=0, le=1)
    artist: float = Field(default=1.0, ge=0, le=1)
    lyrics: float = Field(default=1.0, ge=0, le=1)
    popularity_tiebreak: float = Field(default=0.04, ge=0, le=0.15)


class Intent(BaseModel):
    search_description: str | None = None
    lyrics_search_description: str | None = None
    desired_lyrical_themes: list[str] = []
    avoid_lyrical_themes: list[str] = []
    lyrics_required: bool = False
    avoid_sound: list[str] = []
    signal_weights: SignalWeights = Field(default_factory=SignalWeights)
    moods: list[str] = []
    genres: list[str] = []
    contexts: list[str] = []
    excluded_genres: list[str] = []
    title_contains: str | None = None
    artist_reference: str | None = None
    bpm_min: int | None = None
    bpm_max: int | None = None
    bpm_is_explicit: bool = False
    valence_target: float | None = Field(default=None, ge=0, le=1)
    energy_target: float | None = Field(default=None, ge=0, le=1)
    danceability_target: float | None = Field(default=None, ge=0, le=1)
    acousticness_target: float | None = Field(default=None, ge=0, le=1)
    instrumentalness_target: float | None = Field(default=None, ge=0, le=1)


class ScoreBreakdown(BaseModel):
    semantic: int
    mood: int
    context: int
    tempo: int
    genre: int
    audio: int = 0
    popularity: int = 0
    lyrics: int = 0


class Recommendation(BaseModel):
    song: Song
    score: int = Field(ge=0, le=100)
    explanation: str
    matched_on: list[str]
    breakdown: ScoreBreakdown


class RecommendationRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    limit: int = Field(default=20, ge=1, le=30)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        return " ".join(value.split())


class RecommendationResponse(BaseModel):
    query: str
    summary: str
    intent: Intent
    recommendations: list[Recommendation]
