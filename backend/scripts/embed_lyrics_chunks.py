"""Create a cost-guarded passage-vector pilot without storing raw lyrics.

Preview only (no network or OpenAI calls):
    python -m backend.scripts.embed_lyrics_chunks --limit 100

Explicitly authorize the local experiment:
    python -m backend.scripts.embed_lyrics_chunks --limit 100 --execute
"""

import argparse
import math
import time

import httpx
from sqlalchemy import exists, select

from backend.app.database import database_session
from backend.app.db_models import LyricsChunkRecord, SongRecord
from backend.app.embeddings import (
    LYRICS_CHUNK_EMBEDDING_TEXT_VERSION,
    OpenAIEmbeddingProvider,
    configured_embedding_model,
)
from backend.scripts.embed_lyrics import (
    EMBEDDING_PRICE_PER_MILLION_TOKENS,
    MAX_LYRIC_CHARACTERS,
    MIN_MATCH_CONFIDENCE,
    USER_AGENT,
    lookup,
    lyric_text,
    match_confidence,
)


MAX_CHUNK_CHARACTERS = 1_200
OVERLAP_LINES = 2


def chunk_lyrics(
    text: str, max_characters: int = MAX_CHUNK_CHARACTERS, overlap_lines: int = OVERLAP_LINES
) -> list[str]:
    """Split lyrics into compact, overlapping passages entirely in memory."""
    if max_characters < 100:
        raise ValueError("max_characters must be at least 100")
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        lines.extend(
            line[start : start + max_characters]
            for start in range(0, len(line), max_characters)
        )

    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        proposed = "\n".join([*current, line])
        if current and len(proposed) > max_characters:
            chunks.append("\n".join(current))
            current = current[-overlap_lines:] if overlap_lines else []
            while current and len("\n".join([*current, line])) > max_characters:
                current.pop(0)
        current.append(line)
    if current:
        chunks.append("\n".join(current))

    # Repeated choruses can otherwise create identical vectors for one song.
    return list(dict.fromkeys(chunk for chunk in chunks if chunk.strip()))


def conservative_max_cost(song_count: int) -> float:
    # Line overlap is conservatively budgeted as 30% extra input.
    max_tokens = math.ceil(song_count * MAX_LYRIC_CHARACTERS / 4 * 1.3)
    return max_tokens / 1_000_000 * EMBEDDING_PRICE_PER_MILLION_TOKENS


def selected_song_ids(limit: int, model: str, version: int) -> list[int]:
    already_chunked = exists(
        select(LyricsChunkRecord.id).where(
            LyricsChunkRecord.song_id == SongRecord.id,
            LyricsChunkRecord.embedding_model == model,
            LyricsChunkRecord.embedding_version == version,
        )
    )
    with database_session() as session:
        statement = (
            select(SongRecord.id)
            .where(SongRecord.lyrics_lookup_status == "embedded", ~already_chunked)
            .order_by(SongRecord.popularity.desc().nullslast(), SongRecord.id)
            .limit(limit)
        )
        return list(session.scalars(statement))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype passage-level lyrics vectors")
    parser.add_argument("--limit", type=int, required=True, help="Maximum songs; required as a cost guard")
    parser.add_argument("--execute", action="store_true", help="Perform lookups and billable embedding calls")
    args = parser.parse_args()
    if not 1 <= args.limit <= 1_000:
        parser.error("--limit must be between 1 and 1000 for the local experiment")

    model = configured_embedding_model()
    version = LYRICS_CHUNK_EMBEDDING_TEXT_VERSION
    ids = selected_song_ids(args.limit, model, version)
    print({
        "mode": "local_lyrics_chunk_experiment",
        "embedding_model": model,
        "embedding_version": version,
        "songs_selected": len(ids),
        "chunk_characters": MAX_CHUNK_CHARACTERS,
        "overlap_lines": OVERLAP_LINES,
        "raw_lyrics_retained": False,
        "maximum_embedding_cost_usd": round(conservative_max_cost(len(ids)), 6),
        "execute": args.execute,
    })
    if not ids:
        print("Nothing to do: no eligible analyzed songs remain for this recipe.")
        return
    if not args.execute:
        print("Dry run only. Add --execute to authorize provider lookups and the displayed maximum cost.")
        return

    provider = OpenAIEmbeddingProvider()
    counts = {"embedded_songs": 0, "embedded_chunks": 0, "lookup_skipped": 0, "error": 0}
    actual_tokens = 0
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20.0) as client:
        for position, song_id in enumerate(ids, 1):
            with database_session() as session:
                song = session.get(SongRecord, song_id)
                if song is None:
                    continue
                try:
                    result = lookup(client, song)
                except httpx.HTTPError as error:
                    counts["error"] += 1
                    print(f"Lyrics lookup failed for {song.title!r}: {error}")
                    result = None

                if result is None or match_confidence(song, result) < MIN_MATCH_CONFIDENCE:
                    counts["lookup_skipped"] += 1
                else:
                    text = lyric_text(result)
                    chunks = chunk_lyrics(text) if text else []
                    if not chunks:
                        counts["lookup_skipped"] += 1
                    else:
                        batch = provider.embed_many(chunks)
                        for chunk_index, vector in enumerate(batch.vectors):
                            session.add(LyricsChunkRecord(
                                song_id=song.id,
                                chunk_index=chunk_index,
                                embedding=vector,
                                embedding_model=model,
                                embedding_version=version,
                            ))
                        counts["embedded_songs"] += 1
                        counts["embedded_chunks"] += len(chunks)
                        actual_tokens += batch.input_tokens
            print(f"Lyrics chunks processed {position}/{len(ids)}")
            if position < len(ids):
                time.sleep(0.3)

    print({
        **counts,
        "actual_input_tokens": actual_tokens,
        "actual_estimated_cost_usd": round(
            actual_tokens / 1_000_000 * EMBEDDING_PRICE_PER_MILLION_TOKENS, 6
        ),
    })


if __name__ == "__main__":
    main()
