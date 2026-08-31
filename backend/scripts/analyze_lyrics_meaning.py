"""Create cost-guarded, non-quoting lyrical meaning records.

Free preview (no network or OpenAI calls):
    python -m backend.scripts.analyze_lyrics_meaning --limit 20

Explicitly authorize LRCLIB lookups and billable analysis:
    python -m backend.scripts.analyze_lyrics_meaning --limit 20 --execute
"""

import argparse
import math
import os
import time
from datetime import datetime, timezone

import httpx
from sqlalchemy import or_, select

from backend.app.database import database_session
from backend.app.db_models import SongRecord
from backend.app.lyrics_analysis import (
    GPT_4O_MINI_INPUT_PRICE,
    GPT_4O_MINI_OUTPUT_PRICE,
    LYRICS_MEANING_VERSION,
    LyricsMeaningExtractor,
)
from backend.scripts.embed_lyrics import (
    MAX_LYRIC_CHARACTERS,
    MIN_MATCH_CONFIDENCE,
    USER_AGENT,
    lookup,
    lyric_text,
    match_confidence,
)


MAX_OUTPUT_TOKENS_PER_SONG = 350


def conservative_max_cost(song_count: int) -> float:
    input_tokens = math.ceil(song_count * (MAX_LYRIC_CHARACTERS / 4 + 150))
    output_tokens = song_count * MAX_OUTPUT_TOKENS_PER_SONG
    return (
        input_tokens / 1_000_000 * GPT_4O_MINI_INPUT_PRICE
        + output_tokens / 1_000_000 * GPT_4O_MINI_OUTPUT_PRICE
    )


def selected_song_ids(limit: int, model: str) -> list[int]:
    with database_session() as session:
        statement = (
            select(SongRecord.id)
            .where(
                SongRecord.lyrics_lookup_status == "embedded",
                or_(
                    SongRecord.lyrics_meaning.is_(None),
                    SongRecord.lyrics_meaning_model.is_distinct_from(model),
                    SongRecord.lyrics_meaning_version.is_distinct_from(LYRICS_MEANING_VERSION),
                ),
            )
            .order_by(SongRecord.popularity.desc().nullslast(), SongRecord.id)
            .limit(limit)
        )
        return list(session.scalars(statement))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build derived lyrical meaning for verification")
    parser.add_argument("--limit", type=int, required=True, help="Maximum songs; required as a cost guard")
    parser.add_argument("--execute", action="store_true", help="Perform lookups and billable LLM calls")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100; use small reviewed pilots")

    model = os.getenv("LYRICS_ANALYSIS_MODEL", "gpt-4o-mini")
    ids = selected_song_ids(args.limit, model)
    print({
        "mode": "derived_lyrics_meaning_pilot",
        "analysis_model": model,
        "analysis_version": LYRICS_MEANING_VERSION,
        "songs_selected": len(ids),
        "raw_lyrics_retained": False,
        "maximum_llm_cost_usd": round(conservative_max_cost(len(ids)), 6),
        "execute": args.execute,
    })
    if not ids:
        print("Nothing to do: no eligible analyzed songs remain for this recipe.")
        return
    if not args.execute:
        print("Dry run only. Add --execute to authorize provider lookups and the displayed maximum cost.")
        return

    extractor = LyricsMeaningExtractor(model=model)
    counts = {"analyzed": 0, "lookup_skipped": 0, "error": 0}
    input_tokens = output_tokens = 0
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20.0) as client:
        for position, song_id in enumerate(ids, 1):
            with database_session() as session:
                song = session.get(SongRecord, song_id)
                if song is None:
                    continue
                try:
                    result = lookup(client, song)
                    text = lyric_text(result) if result else None
                    if (
                        result is None
                        or match_confidence(song, result) < MIN_MATCH_CONFIDENCE
                        or not text
                    ):
                        counts["lookup_skipped"] += 1
                    else:
                        meaning, usage = extractor.extract(song.title, song.artist, text)
                        song.lyrics_meaning = meaning.model_dump()
                        song.lyrics_meaning_model = model
                        song.lyrics_meaning_version = LYRICS_MEANING_VERSION
                        song.lyrics_meaning_analyzed_at = datetime.now(timezone.utc)
                        counts["analyzed"] += 1
                        input_tokens += usage.input_tokens
                        output_tokens += usage.output_tokens
                except (httpx.HTTPError, RuntimeError) as error:
                    counts["error"] += 1
                    print(f"Meaning analysis failed for {song.title!r}: {error}")
            print(f"Lyrics meanings processed {position}/{len(ids)}")
            if position < len(ids):
                time.sleep(0.3)

    actual_cost = (
        input_tokens / 1_000_000 * GPT_4O_MINI_INPUT_PRICE
        + output_tokens / 1_000_000 * GPT_4O_MINI_OUTPUT_PRICE
    )
    print({
        **counts,
        "actual_input_tokens": input_tokens,
        "actual_output_tokens": output_tokens,
        "actual_estimated_cost_usd": round(actual_cost, 6),
    })


if __name__ == "__main__":
    main()
