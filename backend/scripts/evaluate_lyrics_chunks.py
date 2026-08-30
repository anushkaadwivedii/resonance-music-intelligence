"""Inspect raw passage and whole-song similarities before calibrating cutoffs.

Free preview:
    python -m backend.scripts.evaluate_lyrics_chunks "heartbreak and unable to love again"

One explicitly authorized embedding batch:
    python -m backend.scripts.evaluate_lyrics_chunks "heartbreak" "forbidden attraction" --execute
"""

import argparse
import math

from sqlalchemy import select, text

from backend.app.database import database_session
from backend.app.db_models import LyricsChunkRecord, SongRecord
from backend.app.embeddings import (
    LYRICS_CHUNK_EMBEDDING_TEXT_VERSION,
    LYRICS_EMBEDDING_TEXT_VERSION,
    OpenAIEmbeddingProvider,
    configured_embedding_model,
)
from backend.app.repository import blended_lyrics_similarity


EMBEDDING_PRICE_PER_MILLION_TOKENS = 0.02


def embedding_text(theme: str) -> str:
    return f"Lyrical themes, story, and emotional meaning: {theme}"


def conservative_cost(themes: list[str]) -> float:
    characters = sum(len(embedding_text(theme)) for theme in themes)
    tokens = math.ceil(characters / 4 * 1.25)
    return tokens / 1_000_000 * EMBEDDING_PRICE_PER_MILLION_TOKENS


def top_scores(vector: list[float], limit: int) -> list[dict[str, str | float]]:
    chunk_distance = LyricsChunkRecord.embedding.cosine_distance(vector).label("chunk_distance")
    whole_distance = SongRecord.lyrics_embedding.cosine_distance(vector).label("whole_distance")
    statement = (
        select(SongRecord, chunk_distance, whole_distance)
        .join(LyricsChunkRecord, LyricsChunkRecord.song_id == SongRecord.id)
        .where(
            LyricsChunkRecord.embedding_model == configured_embedding_model(),
            LyricsChunkRecord.embedding_version == LYRICS_CHUNK_EMBEDDING_TEXT_VERSION,
            SongRecord.lyrics_embedding.is_not(None),
            SongRecord.lyrics_embedding_model == configured_embedding_model(),
            SongRecord.lyrics_embedding_version == LYRICS_EMBEDDING_TEXT_VERSION,
        )
        .order_by(chunk_distance)
        .limit(1_000)
    )
    with database_session() as session:
        session.execute(text("SET LOCAL hnsw.ef_search = 200"))
        rows = session.execute(statement).all()

    strongest_by_song: dict[int, dict[str, str | float]] = {}
    for song, row_chunk_distance, row_whole_distance in rows:
        if song.id in strongest_by_song:
            continue
        chunk = 1.0 - float(row_chunk_distance)
        whole = 1.0 - float(row_whole_distance)
        strongest_by_song[song.id] = {
            "title": song.title,
            "artist": song.artist,
            "passage": round(chunk, 4),
            "whole_song": round(whole, 4),
            "blended": round(blended_lyrics_similarity(chunk, whole), 4),
        }
    return sorted(
        strongest_by_song.values(), key=lambda row: float(row["blended"]), reverse=True
    )[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the local passage-lyrics experiment")
    parser.add_argument("themes", nargs="+", help="Quoted lyrical themes to evaluate in one batch")
    parser.add_argument("--limit", type=int, default=8, choices=range(1, 21), metavar="1-20")
    parser.add_argument("--execute", action="store_true", help="Authorize one billable embedding batch")
    args = parser.parse_args()

    print({
        "mode": "lyrics_chunk_calibration",
        "queries": len(args.themes),
        "embedding_model": configured_embedding_model(),
        "estimated_cost_usd_upper_bound": round(conservative_cost(args.themes), 8),
        "execute": args.execute,
    })
    if not args.execute:
        print("Dry run only. Add --execute to authorize the displayed query batch.")
        return

    batch = OpenAIEmbeddingProvider().embed_many([embedding_text(theme) for theme in args.themes])
    for theme, vector in zip(args.themes, batch.vectors):
        print(f"\nQuery: {theme}")
        for position, row in enumerate(top_scores(vector, args.limit), 1):
            print(f"{position}. {row}")
    print({
        "actual_input_tokens": batch.input_tokens,
        "actual_estimated_cost_usd": round(
            batch.input_tokens / 1_000_000 * EMBEDDING_PRICE_PER_MILLION_TOKENS, 8
        ),
    })


if __name__ == "__main__":
    main()
