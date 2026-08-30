"""Search embedded songs with a natural-language query.

Free preview:
    python -m backend.scripts.search_songs "calm acoustic music for studying"

Execute one embedding API call:
    python -m backend.scripts.search_songs "calm acoustic music for studying" --execute
"""

import argparse
from dataclasses import dataclass

from sqlalchemy import func, select

from backend.app.database import database_session
from backend.app.db_models import SongRecord
from backend.app.embeddings import (
    EMBEDDING_TEXT_VERSION,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    configured_embedding_model,
)
from backend.scripts.embed_songs import conservative_token_estimate, estimated_cost


@dataclass(frozen=True)
class SemanticSearchResult:
    title: str
    artist: str
    genres: list[str]
    bpm: float | None
    similarity: float


def embedded_song_count() -> int:
    """Return the number of songs currently available to semantic search."""
    with database_session() as session:
        statement = select(func.count()).select_from(SongRecord).where(
            SongRecord.embedding.is_not(None),
            SongRecord.embedding_model == configured_embedding_model(),
            SongRecord.embedding_version == EMBEDDING_TEXT_VERSION,
        )
        return int(session.scalar(statement) or 0)


def search_by_vector(query_vector: list[float], limit: int) -> list[SemanticSearchResult]:
    """Run exact cosine-distance search inside PostgreSQL."""
    distance = SongRecord.embedding.cosine_distance(query_vector).label("distance")
    statement = (
        select(SongRecord, distance)
        .where(
            SongRecord.embedding.is_not(None),
            SongRecord.embedding_model == configured_embedding_model(),
            SongRecord.embedding_version == EMBEDDING_TEXT_VERSION,
        )
        .order_by(distance)
        .limit(limit)
    )

    with database_session() as session:
        rows = session.execute(statement).all()
        return [
            SemanticSearchResult(
                title=song.title,
                artist=song.artist,
                genres=song.genres,
                bpm=song.bpm,
                similarity=1.0 - float(cosine_distance),
            )
            for song, cosine_distance in rows
        ]


def run_search(query: str, limit: int, provider: EmbeddingProvider) -> tuple[list[SemanticSearchResult], int]:
    """Embed one query and use its vector to rank songs."""
    batch = provider.embed_many([query])
    return search_by_vector(batch.vectors[0], limit), batch.input_tokens


def format_result(rank: int, result: SemanticSearchResult) -> str:
    genres = ", ".join(result.genres) if result.genres else "unknown genre"
    bpm = f"{result.bpm:.0f} BPM" if result.bpm is not None else "BPM unknown"
    return (
        f"{rank}. {result.title} — {result.artist}\n"
        f"   {genres} | {bpm} | similarity {result.similarity:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantically search embedded PostgreSQL songs")
    parser.add_argument("query", help="A natural-language description of the music you want")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the embedding API; without this flag the command is free",
    )
    args = parser.parse_args()
    query = args.query.strip()
    if not query:
        parser.error("query cannot be empty")
    if not 1 <= args.limit <= 20:
        parser.error("--limit must be between 1 and 20")

    searchable_songs = embedded_song_count()
    estimated_tokens = conservative_token_estimate([query])
    print({
        "query": query,
        "searchable_songs": searchable_songs,
        "result_limit": args.limit,
        "estimated_input_tokens_upper_bound": estimated_tokens,
        "estimated_cost_usd": round(estimated_cost(estimated_tokens), 8),
        "execute": args.execute,
    })

    if searchable_songs == 0:
        print("No songs have embeddings yet. Run embed_songs first.")
        return
    if not args.execute:
        print("Dry run only. Add --execute to authorize one query-embedding API call.")
        return

    results, actual_tokens = run_search(query, args.limit, OpenAIEmbeddingProvider())
    print({
        "actual_input_tokens": actual_tokens,
        "actual_estimated_cost_usd": round(estimated_cost(actual_tokens), 8),
    })
    print("\nResults:")
    for rank, result in enumerate(results, start=1):
        print(format_result(rank, result))


if __name__ == "__main__":
    main()
