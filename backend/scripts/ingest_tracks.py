"""Import and enrich the Kaggle track CSV.

Run from the repository root:
    .venv/bin/python -m backend.scripts.ingest_tracks --limit 100
"""

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import case, func, or_, select
from sqlalchemy.dialects.postgresql import insert

from backend.app.database import database_session
from backend.app.db_models import SongRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "spotify-tracks-dataset.csv"


@dataclass
class ImportedSong:
    source_id: str
    title: str
    artist: str
    album: str
    genres: set[str] = field(default_factory=set)
    bpm: float = 0
    duration_ms: int = 0
    explicit: bool = False
    danceability: float = 0
    energy: float = 0
    acousticness: float = 0
    instrumentalness: float = 0
    speechiness: float = 0
    valence: float = 0
    popularity: int = 0


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def as_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def derive_moods(song: ImportedSong) -> list[str]:
    """Translate continuous audio features into explainable coarse labels."""
    moods: set[str] = set()
    if song.energy <= 0.35:
        moods.add("calm")
    elif song.energy >= 0.75:
        moods.add("energetic")

    if song.valence <= 0.30:
        moods.add("melancholic")
    elif song.valence >= 0.70:
        moods.add("joyful")

    if song.acousticness >= 0.65:
        moods.add("intimate")
    if song.danceability >= 0.72 and song.valence >= 0.55:
        moods.add("playful")
    return sorted(moods or {"balanced"})


def derive_contexts(song: ImportedSong) -> list[str]:
    contexts: set[str] = set()
    if song.energy >= 0.72 and song.bpm >= 110:
        contexts.add("workout")
    if song.danceability >= 0.72 and song.energy >= 0.60:
        contexts.add("party")
    if song.instrumentalness >= 0.50 or (song.energy <= 0.55 and song.speechiness <= 0.08):
        contexts.add("study")
    if song.acousticness >= 0.65 and song.energy <= 0.50:
        contexts.add("quiet morning")
    return sorted(contexts)


def create_description(song: ImportedSong, moods: list[str]) -> str:
    genre_text = ", ".join(sorted(song.genres)) or "uncategorized"
    vocal_style = "mostly instrumental" if song.instrumentalness >= 0.50 else "vocal-led"
    production = "acoustic-leaning" if song.acousticness >= 0.60 else "produced"
    mood_text = ", ".join(moods)
    return (
        f"A {mood_text} {genre_text} track by {song.artist}; "
        f"{production} and {vocal_style}, with {song.energy:.0%} energy, "
        f"{song.danceability:.0%} danceability, and a tempo of {song.bpm:.0f} BPM."
    )


def load_unique_songs(csv_path: Path) -> tuple[dict[str, ImportedSong], int]:
    songs: dict[str, ImportedSong] = {}
    skipped = 0
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            source_id = row["track_id"].strip()
            title = row["track_name"].strip()
            artist = row["artists"].strip()
            if not source_id or not title or not artist:
                skipped += 1
                continue

            if source_id not in songs:
                songs[source_id] = ImportedSong(
                    source_id=source_id,
                    title=title,
                    artist=artist,
                    album=row["album_name"].strip(),
                    bpm=as_float(row["tempo"]),
                    duration_ms=as_int(row["duration_ms"]),
                    explicit=row["explicit"].strip().lower() == "true",
                    danceability=as_float(row["danceability"]),
                    energy=as_float(row["energy"]),
                    acousticness=as_float(row["acousticness"]),
                    instrumentalness=as_float(row["instrumentalness"]),
                    speechiness=as_float(row["speechiness"]),
                    valence=as_float(row["valence"]),
                    popularity=as_int(row["popularity"]),
                )
            songs[source_id].genres.add(row["track_genre"].strip())
    return songs, skipped


def ingest(csv_path: Path, limit: int | None, batch_size: int = 1000) -> dict[str, int]:
    unique_songs, skipped = load_unique_songs(csv_path)
    selected = list(unique_songs.values())[:limit]
    payloads = []
    for song in selected:
        moods = derive_moods(song)
        contexts = derive_contexts(song)
        payloads.append({
            "source": "kaggle-spotify-tracks",
            "source_id": song.source_id,
            "title": song.title,
            "artist": song.artist,
            "album": song.album,
            "genres": sorted(song.genres),
            "bpm": song.bpm,
            "duration_ms": song.duration_ms,
            "explicit": song.explicit,
            "danceability": song.danceability,
            "energy": song.energy,
            "acousticness": song.acousticness,
            "instrumentalness": song.instrumentalness,
            "speechiness": song.speechiness,
            "valence": song.valence,
            "popularity": song.popularity,
            "moods": moods,
            "contexts": contexts,
            "description": create_description(song, moods),
        })

    base_insert = insert(SongRecord)
    embedding_inputs_changed = or_(
        SongRecord.title.is_distinct_from(base_insert.excluded.title),
        SongRecord.artist.is_distinct_from(base_insert.excluded.artist),
        SongRecord.genres.is_distinct_from(base_insert.excluded.genres),
        SongRecord.bpm.is_distinct_from(base_insert.excluded.bpm),
        SongRecord.danceability.is_distinct_from(base_insert.excluded.danceability),
        SongRecord.energy.is_distinct_from(base_insert.excluded.energy),
        SongRecord.acousticness.is_distinct_from(base_insert.excluded.acousticness),
        SongRecord.instrumentalness.is_distinct_from(base_insert.excluded.instrumentalness),
        SongRecord.moods.is_distinct_from(base_insert.excluded.moods),
        SongRecord.contexts.is_distinct_from(base_insert.excluded.contexts),
        SongRecord.description.is_distinct_from(base_insert.excluded.description),
    )
    sound_inputs_changed = or_(
        SongRecord.genres.is_distinct_from(base_insert.excluded.genres),
        SongRecord.bpm.is_distinct_from(base_insert.excluded.bpm),
        SongRecord.danceability.is_distinct_from(base_insert.excluded.danceability),
        SongRecord.energy.is_distinct_from(base_insert.excluded.energy),
        SongRecord.acousticness.is_distinct_from(base_insert.excluded.acousticness),
        SongRecord.instrumentalness.is_distinct_from(base_insert.excluded.instrumentalness),
        SongRecord.speechiness.is_distinct_from(base_insert.excluded.speechiness),
        SongRecord.valence.is_distinct_from(base_insert.excluded.valence),
        SongRecord.moods.is_distinct_from(base_insert.excluded.moods),
        SongRecord.contexts.is_distinct_from(base_insert.excluded.contexts),
    )
    upsert = base_insert.on_conflict_do_update(
        constraint="uq_songs_source_identity",
        set_={
            "title": base_insert.excluded.title,
            "artist": base_insert.excluded.artist,
            "album": base_insert.excluded.album,
            "genres": base_insert.excluded.genres,
            "bpm": base_insert.excluded.bpm,
            "duration_ms": base_insert.excluded.duration_ms,
            "explicit": base_insert.excluded.explicit,
            "danceability": base_insert.excluded.danceability,
            "energy": base_insert.excluded.energy,
            "acousticness": base_insert.excluded.acousticness,
            "instrumentalness": base_insert.excluded.instrumentalness,
            "speechiness": base_insert.excluded.speechiness,
            "valence": base_insert.excluded.valence,
            "popularity": base_insert.excluded.popularity,
            "moods": base_insert.excluded.moods,
            "contexts": base_insert.excluded.contexts,
            "description": base_insert.excluded.description,
            "embedding": case((embedding_inputs_changed, None), else_=SongRecord.embedding),
            "embedding_model": case((embedding_inputs_changed, None), else_=SongRecord.embedding_model),
            "embedding_version": case((embedding_inputs_changed, None), else_=SongRecord.embedding_version),
            "embedded_at": case((embedding_inputs_changed, None), else_=SongRecord.embedded_at),
            "sound_embedding": case((sound_inputs_changed, None), else_=SongRecord.sound_embedding),
            "sound_embedding_model": case((sound_inputs_changed, None), else_=SongRecord.sound_embedding_model),
            "sound_embedding_version": case((sound_inputs_changed, None), else_=SongRecord.sound_embedding_version),
            "sound_embedded_at": case((sound_inputs_changed, None), else_=SongRecord.sound_embedded_at),
        },
    )
    for start in range(0, len(payloads), batch_size):
        with database_session() as session:
            session.execute(upsert, payloads[start : start + batch_size])
        if len(payloads) > batch_size:
            print(f"Imported {min(start + batch_size, len(payloads))}/{len(payloads)} songs")
    with database_session() as session:
        total = session.scalar(select(func.count()).select_from(SongRecord)) or 0
    return {"unique_in_csv": len(unique_songs), "skipped": skipped, "processed": len(selected), "rows_in_database": total}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import the Kaggle music catalog")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Import only the first N unique songs")
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows committed per resumable batch")
    arguments = parser.parse_args()
    if not arguments.csv.exists():
        parser.error(f"CSV not found: {arguments.csv}")
    if not 1 <= arguments.batch_size <= 5000:
        parser.error("--batch-size must be between 1 and 5000")
    print(ingest(arguments.csv, arguments.limit, arguments.batch_size))


if __name__ == "__main__":
    main()
