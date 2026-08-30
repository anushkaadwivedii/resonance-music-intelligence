"""Populate shadow sound-only vectors without title, artist, album, or lyrics.

Free preview:
    python -m backend.scripts.embed_sound_songs --limit 10

Execute the displayed batch:
    python -m backend.scripts.embed_sound_songs --limit 10 --execute
"""

import argparse
import math
from datetime import datetime, timezone

from sqlalchemy import or_, select

from backend.app.database import database_session
from backend.app.db_models import SongRecord
from backend.app.embeddings import (
    SOUND_EMBEDDING_TEXT_VERSION,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    configured_embedding_model,
)


EMBEDDING_PRICE_PER_MILLION_TOKENS = 0.02


def sound_embedding_text(song: SongRecord) -> str:
    """Describe sound only; identity and lyrical meaning are separate channels."""
    def percent(value: float | None) -> str:
        return f"{value:.0%}" if value is not None else "unknown"

    bpm = f"{song.bpm:.0f}" if song.bpm is not None else "unknown"
    return "\n".join([
        f"Genres: {', '.join(song.genres) if song.genres else 'uncategorized'}",
        f"Audio-derived moods: {', '.join(song.moods) if song.moods else 'balanced'}",
        f"Listening contexts: {', '.join(song.contexts) if song.contexts else 'general listening'}",
        f"Tempo: {bpm} BPM",
        f"Audio features: {percent(song.energy)} energy, {percent(song.danceability)} danceability, "
        f"{percent(song.valence)} valence, {percent(song.acousticness)} acousticness, "
        f"{percent(song.instrumentalness)} instrumentalness, {percent(song.speechiness)} speechiness",
    ])


def conservative_token_estimate(texts: list[str]) -> int:
    return math.ceil(sum(len(text) for text in texts) / 4 * 1.25)


def estimated_cost(tokens: int) -> float:
    return tokens / 1_000_000 * EMBEDDING_PRICE_PER_MILLION_TOKENS


def selected_song_ids(limit: int, model: str, version: int) -> list[int]:
    with database_session() as session:
        statement = (
            select(SongRecord.id)
            .where(or_(
                SongRecord.sound_embedding.is_(None),
                SongRecord.sound_embedding_model.is_distinct_from(model),
                SongRecord.sound_embedding_version.is_distinct_from(version),
            ))
            .order_by(SongRecord.id)
            .limit(limit)
        )
        return list(session.scalars(statement))


def estimated_tokens_for_ids(ids: list[int], read_batch_size: int = 1000) -> int:
    total_characters = 0
    for start in range(0, len(ids), read_batch_size):
        with database_session() as session:
            records = list(session.scalars(
                select(SongRecord)
                .where(SongRecord.id.in_(ids[start : start + read_batch_size]))
                .order_by(SongRecord.id)
            ))
            total_characters += sum(len(sound_embedding_text(record)) for record in records)
    return math.ceil(total_characters / 4 * 1.25)


def run_embedding_job(
    ids: list[int], provider: EmbeddingProvider, batch_size: int, model: str, version: int
) -> tuple[int, int]:
    embedded = 0
    actual_tokens = 0
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        with database_session() as session:
            records = list(session.scalars(
                select(SongRecord).where(SongRecord.id.in_(batch_ids)).order_by(SongRecord.id)
            ))
            result = provider.embed_many([sound_embedding_text(record) for record in records])
            now = datetime.now(timezone.utc)
            for record, vector in zip(records, result.vectors):
                record.sound_embedding = vector
                record.sound_embedding_model = model
                record.sound_embedding_version = version
                record.sound_embedded_at = now
            embedded += len(records)
            actual_tokens += result.input_tokens
        print(f"Embedded sound {embedded}/{len(ids)} songs")
    return embedded, actual_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed sound-only evidence into the shadow vector")
    parser.add_argument("--limit", type=int, required=True, help="Maximum songs; required as a cost guard")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--execute", action="store_true", help="Call the API; without this flag the command is free")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100_000:
        parser.error("--limit must be between 1 and 100000")
    if not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be between 1 and 100")

    model = configured_embedding_model()
    version = SOUND_EMBEDDING_TEXT_VERSION
    ids = selected_song_ids(args.limit, model, version)
    estimated_tokens = estimated_tokens_for_ids(ids)
    print({
        "mode": "pending_sound_only",
        "embedding_model": model,
        "sound_embedding_version": version,
        "songs_selected": len(ids),
        "batch_size": args.batch_size,
        "estimated_input_tokens_upper_bound": estimated_tokens,
        "estimated_cost_usd": round(estimated_cost(estimated_tokens), 6),
        "execute": args.execute,
    })
    if not ids:
        print("Nothing to do: every song already has the current sound embedding.")
        return
    if not args.execute:
        print("Dry run only. Add --execute to authorize the displayed batch.")
        return

    embedded, actual_tokens = run_embedding_job(ids, OpenAIEmbeddingProvider(), args.batch_size, model, version)
    print({
        "embedded": embedded,
        "actual_input_tokens": actual_tokens,
        "actual_estimated_cost_usd": round(estimated_cost(actual_tokens), 6),
    })


if __name__ == "__main__":
    main()
