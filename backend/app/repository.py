"""Song storage abstractions.

The retrieval layer depends on SongRepository, not on SQLite directly. A later
Postgres/pgvector implementation can satisfy the same contract.
"""

import hashlib
import os
from typing import Protocol

from sqlalchemy import and_, func, or_, select, text

from .database import database_session
from .db_models import SongRecord
from .embeddings import EMBEDDING_TEXT_VERSION, SOUND_EMBEDDING_TEXT_VERSION, configured_embedding_model
from .models import Intent, Song


class SongRepository(Protocol):
    def list_songs(self, limit: int | None = None) -> list[Song]: ...
    def count(self) -> int: ...
    def search_by_vector(self, query_vector: list[float], limit: int) -> list[tuple[Song, float]]: ...
    def search_by_metadata(
        self, query_vector: list[float], intent: Intent, limit: int
    ) -> list[tuple[Song, float]]: ...
    def search_by_lyrics(
        self, lyrics_vector: list[float], sound_vector: list[float], limit: int
    ) -> list[tuple[Song, float, float]]: ...


ACCENTS = ["#8b7cff", "#d09a6a", "#61af93", "#e7a1a4", "#4c79d8", "#b98a5f", "#977ac0"]


def accent_for(source_id: str) -> str:
    """Give album placeholders a stable color without storing UI data."""
    digest = hashlib.blake2b(source_id.encode(), digest_size=2).digest()
    return ACCENTS[int.from_bytes(digest, "big") % len(ACCENTS)]


class PostgresSongRepository:
    @staticmethod
    def _embedding_columns():
        if os.getenv("USE_SOUND_EMBEDDINGS", "false").lower() == "true":
            return (
                SongRecord.sound_embedding,
                SongRecord.sound_embedding_model,
                SongRecord.sound_embedding_version,
                SOUND_EMBEDDING_TEXT_VERSION,
            )
        return SongRecord.embedding, SongRecord.embedding_model, SongRecord.embedding_version, EMBEDDING_TEXT_VERSION

    def list_songs(self, limit: int | None = None) -> list[Song]:
        statement = select(SongRecord).order_by(SongRecord.popularity.desc().nullslast(), SongRecord.id)
        if limit is not None:
            statement = statement.limit(limit)
        with database_session() as session:
            records = list(session.scalars(statement))
        return [self._to_song(record) for record in records]

    def count(self) -> int:
        with database_session() as session:
            return session.scalar(select(func.count()).select_from(SongRecord)) or 0

    def search_by_vector(self, query_vector: list[float], limit: int) -> list[tuple[Song, float]]:
        """Return embedded songs ordered by cosine similarity."""
        vector_column, model_column, version_column, version = self._embedding_columns()
        distance = vector_column.cosine_distance(query_vector).label("distance")
        statement = (
            select(SongRecord, distance)
            .where(
                vector_column.is_not(None),
                model_column == configured_embedding_model(),
                version_column == version,
            )
            .order_by(distance)
            .limit(limit)
        )
        with database_session() as session:
            session.execute(text("SET LOCAL hnsw.ef_search = 200"))
            rows = session.execute(statement).all()
            return [(self._to_song(record), 1.0 - float(cosine_distance)) for record, cosine_distance in rows]

    def search_by_metadata(
        self, query_vector: list[float], intent: Intent, limit: int
    ) -> list[tuple[Song, float]]:
        """Recall candidates through explicit signals, independent of vector top-k."""
        signals = []
        if intent.moods:
            signals.append(SongRecord.moods.overlap(intent.moods))
        if intent.contexts:
            signals.append(SongRecord.contexts.overlap(intent.contexts))
        if intent.genres:
            signals.append(SongRecord.genres.overlap(intent.genres))
        if intent.title_contains:
            signals.append(SongRecord.title.ilike(f"%{intent.title_contains}%"))
        if intent.artist_reference:
            signals.append(SongRecord.artist.ilike(f"%{intent.artist_reference}%"))
        feature_targets = [
            (SongRecord.valence, intent.valence_target),
            (SongRecord.energy, intent.energy_target),
            (SongRecord.danceability, intent.danceability_target),
            (SongRecord.acousticness, intent.acousticness_target),
            (SongRecord.instrumentalness, intent.instrumentalness_target),
        ]
        for column, target in feature_targets:
            if target is not None:
                signals.append(column.between(max(0.0, target - 0.20), min(1.0, target + 0.20)))
        if intent.bpm_min is not None or intent.bpm_max is not None:
            tempo_conditions = []
            if intent.bpm_min is not None:
                tempo_conditions.append(SongRecord.bpm >= intent.bpm_min)
            if intent.bpm_max is not None:
                tempo_conditions.append(SongRecord.bpm <= intent.bpm_max)
            signals.append(and_(*tempo_conditions))
        if not signals:
            return []

        vector_column, model_column, version_column, version = self._embedding_columns()
        distance = vector_column.cosine_distance(query_vector).label("distance")
        conditions = [
            vector_column.is_not(None),
            model_column == configured_embedding_model(),
            version_column == version,
            or_(*signals),
        ]
        if intent.excluded_genres:
            conditions.append(~SongRecord.genres.overlap(intent.excluded_genres))

        statement = (
            select(SongRecord, distance)
            .where(*conditions)
            .order_by(SongRecord.popularity.desc().nullslast(), SongRecord.id)
            .limit(limit)
        )
        with database_session() as session:
            rows = session.execute(statement).all()
            return [(self._to_song(record), 1.0 - float(cosine_distance)) for record, cosine_distance in rows]

    def search_by_lyrics(
        self, lyrics_vector: list[float], sound_vector: list[float], limit: int
    ) -> list[tuple[Song, float, float]]:
        """Retrieve experimental lyrical meaning while retaining sound evidence."""
        from .embeddings import LYRICS_EMBEDDING_TEXT_VERSION

        sound_column, sound_model_column, sound_version_column, sound_version = self._embedding_columns()
        lyrics_distance = SongRecord.lyrics_embedding.cosine_distance(lyrics_vector).label("lyrics_distance")
        sound_distance = sound_column.cosine_distance(sound_vector).label("sound_distance")
        statement = (
            select(SongRecord, lyrics_distance, sound_distance)
            .where(
                SongRecord.lyrics_embedding.is_not(None),
                SongRecord.lyrics_embedding_model == configured_embedding_model(),
                SongRecord.lyrics_embedding_version == LYRICS_EMBEDDING_TEXT_VERSION,
                sound_column.is_not(None),
                sound_model_column == configured_embedding_model(),
                sound_version_column == sound_version,
            )
            .order_by(lyrics_distance)
            .limit(limit)
        )
        with database_session() as session:
            session.execute(text("SET LOCAL hnsw.ef_search = 200"))
            rows = session.execute(statement).all()
            return [
                (
                    self._to_song(record),
                    1.0 - float(row_lyrics_distance),
                    1.0 - float(row_sound_distance),
                )
                for record, row_lyrics_distance, row_sound_distance in rows
            ]

    @staticmethod
    def _to_song(record: SongRecord) -> Song:
        return Song(
            id=record.source_id,
            title=record.title,
            artist=record.artist,
            album=record.album,
            genre=record.genres[0] if record.genres else "uncategorized",
            genres=record.genres,
            moods=record.moods,
            contexts=record.contexts,
            bpm=record.bpm or 0,
            perceived_bpm=record.bpm / 2 if record.bpm is not None and record.bpm >= 160 else None,
            year=None,
            description=record.description,
            accent=accent_for(record.source_id),
            popularity=record.popularity,
            energy=record.energy,
            danceability=record.danceability,
            valence=record.valence,
            acousticness=record.acousticness,
            instrumentalness=record.instrumentalness,
            lyrics_evidence=(
                "analyzed"
                if record.lyrics_lookup_status == "embedded"
                else "unavailable"
                if record.lyrics_lookup_status in {"not_found", "no_lyrics", "ambiguous"}
                else "not_analyzed"
            ),
        )


class InMemorySongRepository:
    """Small deterministic repository for unit tests and isolated experiments."""

    def __init__(self, songs: list[Song]):
        self.songs = songs

    def list_songs(self, limit: int | None = None) -> list[Song]:
        return self.songs if limit is None else self.songs[:limit]

    def count(self) -> int:
        return len(self.songs)

    def search_by_vector(self, query_vector: list[float], limit: int) -> list[tuple[Song, float]]:
        raise NotImplementedError("In-memory repositories use the free hashing retriever")

    def search_by_metadata(
        self, query_vector: list[float], intent: Intent, limit: int
    ) -> list[tuple[Song, float]]:
        raise NotImplementedError("In-memory repositories use the free hashing retriever")

    def search_by_lyrics(self, lyrics_vector, sound_vector, limit):
        raise NotImplementedError("In-memory repositories do not contain lyrics vectors")


repository = PostgresSongRepository()
