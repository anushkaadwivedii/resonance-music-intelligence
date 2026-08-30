"""Generate and persist embeddings for songs that do not have one.

Dry run (free):
    python -m backend.scripts.embed_songs --limit 10

Refresh already embedded songs (preview first):
    python -m backend.scripts.embed_songs --limit 10 --refresh-existing

Execute API calls:
    python -m backend.scripts.embed_songs --limit 10 --execute
"""

import argparse
import math
from datetime import datetime, timezone

from sqlalchemy import or_, select

from backend.app.database import database_session
from backend.app.db_models import SongRecord
from backend.app.embeddings import (
    EMBEDDING_TEXT_VERSION,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    configured_embedding_model,
)


EMBEDDING_PRICE_PER_MILLION_TOKENS = 0.02


def embedding_text(song: SongRecord) -> str:
    bpm = f"{song.bpm:.0f}" if song.bpm is not None else "unknown"
    energy = f"{song.energy:.0%}" if song.energy is not None else "unknown"
    danceability = f"{song.danceability:.0%}" if song.danceability is not None else "unknown"
    acousticness = f"{song.acousticness:.0%}" if song.acousticness is not None else "unknown"
    instrumentalness = f"{song.instrumentalness:.0%}" if song.instrumentalness is not None else "unknown"
    return "\n".join([
        f"Title: {song.title}",
        f"Artist: {song.artist}",
        f"Genres: {', '.join(song.genres)}",
        f"Moods: {', '.join(song.moods)}",
        f"Contexts: {', '.join(song.contexts) if song.contexts else 'general listening'}",
        f"Tempo: {bpm} BPM",
        f"Audio features: {energy} energy, {danceability} danceability, "
        f"{acousticness} acousticness, {instrumentalness} instrumentalness",
        f"Description: {song.description}",
    ])


def conservative_token_estimate(texts: list[str]) -> int:
    # English text often averages about four characters per token. Multiplying
    # by 1.25 intentionally overestimates so the preview is budget-conservative.
    return math.ceil(sum(len(text) for text in texts) / 4 * 1.25)


def estimated_cost(tokens: int) -> float:
    return tokens / 1_000_000 * EMBEDDING_PRICE_PER_MILLION_TOKENS


def estimated_tokens_for_ids(ids: list[int], read_batch_size: int = 1000) -> int:
    """Estimate a large selection without creating an oversized SQL IN clause."""
    total_characters = 0
    for start in range(0, len(ids), read_batch_size):
        batch_ids = ids[start : start + read_batch_size]
        with database_session() as session:
            records = list(session.scalars(
                select(SongRecord).where(SongRecord.id.in_(batch_ids)).order_by(SongRecord.id)
            ))
            total_characters += sum(len(embedding_text(record)) for record in records)
    return math.ceil(total_characters / 4 * 1.25)


def selected_song_ids(
    limit: int,
    model: str,
    version: int,
    refresh_existing: bool = False,
) -> list[int]:
    with database_session() as session:
        embedding_filter = SongRecord.embedding.is_not(None) if refresh_existing else or_(
            SongRecord.embedding.is_(None),
            SongRecord.embedding_model.is_distinct_from(model),
            SongRecord.embedding_version.is_distinct_from(version),
        )
        statement = (
            select(SongRecord.id)
            .where(embedding_filter)
            .order_by(SongRecord.id)
            .limit(limit)
        )
        return list(session.scalars(statement))


def run_embedding_job(
    ids: list[int],
    provider: EmbeddingProvider,
    batch_size: int,
    model: str,
    version: int,
) -> tuple[int, int]:
    embedded = 0
    actual_tokens = 0
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        with database_session() as session:
            records = list(session.scalars(select(SongRecord).where(SongRecord.id.in_(batch_ids)).order_by(SongRecord.id)))
            texts = [embedding_text(record) for record in records]
            result = provider.embed_many(texts)
            for record, vector in zip(records, result.vectors):
                record.embedding = vector
                record.embedding_model = model
                record.embedding_version = version
                record.embedded_at = datetime.now(timezone.utc)
            embedded += len(records)
            actual_tokens += result.input_tokens
        print(f"Embedded {embedded}/{len(ids)} songs")
    return embedded, actual_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed unprocessed PostgreSQL songs")
    parser.add_argument("--limit", type=int, required=True, help="Maximum songs to embed; required as a cost guard")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Re-embed songs that already have vectors; useful after changing embedding text",
    )
    parser.add_argument("--execute", action="store_true", help="Actually call the API; without this flag the command is free")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100_000:
        parser.error("--limit must be between 1 and 100000")
    if not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be between 1 and 100")

    model = configured_embedding_model()
    version = EMBEDDING_TEXT_VERSION
    ids = selected_song_ids(args.limit, model, version, args.refresh_existing)
    estimated_tokens = estimated_tokens_for_ids(ids)
    print({
        "mode": "refresh_existing" if args.refresh_existing else "pending_only",
        "embedding_model": model,
        "embedding_text_version": version,
        "songs_selected": len(ids),
        "batch_size": args.batch_size,
        "estimated_input_tokens_upper_bound": estimated_tokens,
        "estimated_cost_usd": round(estimated_cost(estimated_tokens), 6),
        "execute": args.execute,
    })

    if not ids:
        print("Nothing to do: no songs matched the selected mode.")
        return
    if not args.execute:
        print("Dry run only. Add --execute to authorize the displayed batch.")
        return

    provider = OpenAIEmbeddingProvider()
    embedded, actual_tokens = run_embedding_job(ids, provider, args.batch_size, model, version)
    print({
        "embedded": embedded,
        "actual_input_tokens": actual_tokens,
        "actual_estimated_cost_usd": round(estimated_cost(actual_tokens), 6),
    })


if __name__ == "__main__":
    main()
