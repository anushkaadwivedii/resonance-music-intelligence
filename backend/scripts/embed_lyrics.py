"""Prototype lyrics lookup and embeddings without storing raw lyric text.

Free preview (no network or OpenAI calls):
    python -m backend.scripts.embed_lyrics --limit 100

Local experiment (LRCLIB lookups plus billable embeddings):
    python -m backend.scripts.embed_lyrics --limit 100 --execute

This is for local evaluation only until the lyrics source grants appropriate
production and derived-use rights.
"""

import argparse
import hashlib
import math
import re
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher

import httpx
from sqlalchemy import and_, or_, select

from backend.app.database import database_session
from backend.app.db_models import SongRecord
from backend.app.embeddings import (
    LYRICS_EMBEDDING_TEXT_VERSION,
    OpenAIEmbeddingProvider,
    configured_embedding_model,
)


PROVIDER = "lrclib"
LRCLIB_URL = "https://lrclib.net/api/get"
USER_AGENT = "Resonance/0.1 (local educational lyrics-retrieval experiment)"
EMBEDDING_PRICE_PER_MILLION_TOKENS = 0.02
MAX_LYRIC_CHARACTERS = 20_000
MIN_MATCH_CONFIDENCE = 0.82
MAX_LOOKUP_ATTEMPTS = 5


def normalize_identity(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def match_confidence(song: SongRecord, result: dict) -> float:
    title = SequenceMatcher(
        None, normalize_identity(song.title), normalize_identity(result.get("trackName", ""))
    ).ratio()
    primary_artist = song.artist.split(";", 1)[0]
    artist = SequenceMatcher(
        None, normalize_identity(primary_artist), normalize_identity(result.get("artistName", ""))
    ).ratio()
    returned_duration = result.get("duration")
    expected_duration = song.duration_ms / 1000 if song.duration_ms else None
    if returned_duration is None or expected_duration is None:
        duration = 0.5
    else:
        duration = max(0.0, 1 - abs(float(returned_duration) - expected_duration) / 10)
    return 0.5 * title + 0.35 * artist + 0.15 * duration


def lyric_text(result: dict) -> str | None:
    plain = result.get("plainLyrics")
    if plain:
        return str(plain)[:MAX_LYRIC_CHARACTERS]
    synced = result.get("syncedLyrics")
    if not synced:
        return None
    without_timestamps = re.sub(r"^\[[^]]+\]\s*", "", str(synced), flags=re.MULTILINE)
    return without_timestamps[:MAX_LYRIC_CHARACTERS]


def pending_filter(model: str, version: int):
    return or_(
        SongRecord.lyrics_lookup_status.is_(None),
        SongRecord.lyrics_lookup_status == "error",
        and_(
            SongRecord.lyrics_lookup_status == "embedded",
            or_(
                SongRecord.lyrics_embedding_model.is_distinct_from(model),
                SongRecord.lyrics_embedding_version.is_distinct_from(version),
            ),
        ),
    )


def representative_ids(
    rows: list[tuple[int, int | None, list[str], str]], limit: int
) -> list[int]:
    """Sample popularity quartiles and genres, deterministically and repeatably."""
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: (row[1] if row[1] is not None else -1), reverse=True)
    quartiles: list[list[tuple[int, int | None, list[str], str]]] = [[], [], [], []]
    for index, row in enumerate(ordered):
        quartile = min(3, index * 4 // len(ordered))
        quartiles[quartile].append(row)

    selected: list[int] = []
    base_quota, remainder = divmod(min(limit, len(rows)), 4)
    for quartile_index, quartile_rows in enumerate(quartiles):
        quota = base_quota + (1 if quartile_index < remainder else 0)
        by_genre: dict[str, list[tuple[int, int | None, list[str], str]]] = {}
        for row in quartile_rows:
            genre = row[2][0].casefold() if row[2] else "uncategorized"
            by_genre.setdefault(genre, []).append(row)
        for genre_rows in by_genre.values():
            genre_rows.sort(key=lambda row: hashlib.blake2b(row[3].encode(), digest_size=8).digest())

        genre_names = sorted(by_genre)
        while quota > 0 and genre_names:
            next_round = []
            for genre in genre_names:
                if quota == 0:
                    break
                if by_genre[genre]:
                    row = by_genre[genre].pop()
                    selected.append(row[0])
                    quota -= 1
                if by_genre[genre]:
                    next_round.append(genre)
            genre_names = next_round
    return selected


def selected_song_ids(limit: int, model: str, version: int, sampling: str = "popular") -> list[int]:
    with database_session() as session:
        if sampling == "popular":
            statement = (
                select(SongRecord.id)
                .where(pending_filter(model, version))
                .order_by(SongRecord.popularity.desc().nullslast(), SongRecord.id)
                .limit(limit)
            )
            return list(session.scalars(statement))

        rows = session.execute(
            select(SongRecord.id, SongRecord.popularity, SongRecord.genres, SongRecord.source_id)
            .where(pending_filter(model, version))
        ).all()
        return representative_ids(
            [(row.id, row.popularity, row.genres, row.source_id) for row in rows], limit
        )


def conservative_max_cost(song_count: int) -> float:
    max_tokens = math.ceil(song_count * MAX_LYRIC_CHARACTERS / 4 * 1.25)
    return max_tokens / 1_000_000 * EMBEDDING_PRICE_PER_MILLION_TOKENS


def lookup(client: httpx.Client, song: SongRecord) -> dict | None:
    params = {
        "track_name": song.title,
        "artist_name": song.artist.split(";", 1)[0],
    }
    if song.album:
        params["album_name"] = song.album
    if song.duration_ms:
        params["duration"] = round(song.duration_ms / 1000)
    for attempt in range(MAX_LOOKUP_ATTEMPTS):
        response = client.get(LRCLIB_URL, params=params)
        if response.status_code == 404:
            return None
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == MAX_LOOKUP_ATTEMPTS - 1:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(2 ** attempt, 15)
            except ValueError:
                delay = min(2 ** attempt, 15)
            print(
                f"Lyrics provider returned {response.status_code}; "
                f"retrying in {delay:g}s ({attempt + 1}/{MAX_LOOKUP_ATTEMPTS})"
            )
            time.sleep(delay)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("Lyrics lookup retry loop ended unexpectedly")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype a non-publishing lyrics embedding channel")
    parser.add_argument("--limit", type=int, required=True, help="Maximum songs; required as a cost guard")
    parser.add_argument("--execute", action="store_true", help="Perform lookups and billable embedding calls")
    parser.add_argument(
        "--sampling", choices=["popular", "representative"], default="popular",
        help="popular continues sequentially; representative spans popularity quartiles and genres",
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 1000:
        parser.error("--limit must be between 1 and 1000 for the local experiment")

    model = configured_embedding_model()
    version = LYRICS_EMBEDDING_TEXT_VERSION
    ids = selected_song_ids(args.limit, model, version, args.sampling)
    print({
        "mode": "local_lyrics_experiment",
        "provider": PROVIDER,
        "sampling": args.sampling,
        "songs_selected": len(ids),
        "raw_lyrics_retained": False,
        "maximum_embedding_cost_usd": round(conservative_max_cost(len(ids)), 6),
        "execute": args.execute,
    })
    if not ids:
        print("Nothing to do: no songs matched the selected mode.")
        return
    if not args.execute:
        print("Dry run only. Add --execute to authorize provider lookups and the displayed maximum cost.")
        return

    provider = OpenAIEmbeddingProvider()
    counts = {"embedded": 0, "not_found": 0, "ambiguous": 0, "no_lyrics": 0, "error": 0}
    actual_tokens = 0
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20.0) as client:
        for position, song_id in enumerate(ids, 1):
            with database_session() as session:
                song = session.get(SongRecord, song_id)
                if song is None:
                    continue
                now = datetime.now(timezone.utc)
                song.lyrics_provider = PROVIDER
                song.lyrics_checked_at = now
                try:
                    result = lookup(client, song)
                except httpx.HTTPError as error:
                    song.lyrics_lookup_status = "error"
                    counts["error"] += 1
                    print(f"Lyrics lookup failed after retries for {song.title!r}: {error}")
                    result = None
                if song.lyrics_lookup_status == "error":
                    pass
                elif result is None:
                    song.lyrics_lookup_status = "not_found"
                    counts["not_found"] += 1
                else:
                    confidence = match_confidence(song, result)
                    song.lyrics_source_id = str(result.get("id", "")) or None
                    song.lyrics_match_confidence = confidence
                    text = lyric_text(result)
                    if confidence < MIN_MATCH_CONFIDENCE:
                        song.lyrics_lookup_status = "ambiguous"
                        counts["ambiguous"] += 1
                    elif not text:
                        song.lyrics_lookup_status = "no_lyrics"
                        counts["no_lyrics"] += 1
                    else:
                        batch = provider.embed_many([text])
                        song.lyrics_embedding = batch.vectors[0]
                        song.lyrics_embedding_model = model
                        song.lyrics_embedding_version = version
                        song.lyrics_embedded_at = now
                        song.lyrics_lookup_status = "embedded"
                        counts["embedded"] += 1
                        actual_tokens += batch.input_tokens
            print(f"Lyrics processed {position}/{len(ids)}")
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
