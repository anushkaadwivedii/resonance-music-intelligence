from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class SongRecord(Base):
    __tablename__ = "songs"
    __table_args__ = (UniqueConstraint("source", "source_id", name="uq_songs_source_identity"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(80), nullable=False)
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    artist: Mapped[str] = mapped_column(Text, nullable=False)
    album: Mapped[str | None] = mapped_column(Text)
    genres: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    bpm: Mapped[float | None] = mapped_column(Float)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    explicit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    danceability: Mapped[float | None] = mapped_column(Float)
    energy: Mapped[float | None] = mapped_column(Float)
    acousticness: Mapped[float | None] = mapped_column(Float)
    instrumentalness: Mapped[float | None] = mapped_column(Float)
    speechiness: Mapped[float | None] = mapped_column(Float)
    valence: Mapped[float | None] = mapped_column(Float)
    popularity: Mapped[int | None] = mapped_column(Integer)
    moods: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    contexts: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536))
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    embedding_version: Mapped[int | None] = mapped_column(Integer)
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sound_embedding: Mapped[list[float] | None] = mapped_column(Vector(1536))
    sound_embedding_model: Mapped[str | None] = mapped_column(String(100))
    sound_embedding_version: Mapped[int | None] = mapped_column(Integer)
    sound_embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lyrics_embedding: Mapped[list[float] | None] = mapped_column(Vector(1536))
    lyrics_embedding_model: Mapped[str | None] = mapped_column(String(100))
    lyrics_embedding_version: Mapped[int | None] = mapped_column(Integer)
    lyrics_embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lyrics_provider: Mapped[str | None] = mapped_column(String(80))
    lyrics_source_id: Mapped[str | None] = mapped_column(String(255))
    lyrics_lookup_status: Mapped[str | None] = mapped_column(String(40))
    lyrics_match_confidence: Mapped[float | None] = mapped_column(Float)
    lyrics_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lyrics_meaning: Mapped[dict | None] = mapped_column(JSONB)
    lyrics_meaning_model: Mapped[str | None] = mapped_column(String(100))
    lyrics_meaning_version: Mapped[int | None] = mapped_column(Integer)
    lyrics_meaning_analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class LyricsChunkRecord(Base):
    """A derived passage vector. Raw lyric text is deliberately not retained."""

    __tablename__ = "lyrics_chunks"
    __table_args__ = (
        UniqueConstraint(
            "song_id", "chunk_index", "embedding_model", "embedding_version",
            name="uq_lyrics_chunks_recipe_position",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    song_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("songs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
