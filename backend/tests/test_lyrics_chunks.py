import pytest

from backend.scripts.embed_lyrics_chunks import chunk_lyrics, conservative_max_cost
from backend.scripts.evaluate_lyrics_chunks import conservative_cost, embedding_text


def test_chunk_lyrics_keeps_passages_small_and_overlapping():
    lyrics = "\n".join(f"line {index} with a few words" for index in range(20))

    chunks = chunk_lyrics(lyrics, max_characters=120, overlap_lines=2)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    first_lines = chunks[0].splitlines()
    second_lines = chunks[1].splitlines()
    assert second_lines[:2] == first_lines[-2:]


def test_chunk_lyrics_deduplicates_repeated_passages():
    repeated = "same chorus line\nanother chorus line"

    chunks = chunk_lyrics(repeated, max_characters=100)

    assert chunks == [repeated]


def test_chunk_cost_guard_is_bounded_for_small_pilot():
    assert conservative_max_cost(100) == pytest.approx(0.013)


def test_calibration_preview_cost_is_tiny_for_three_short_queries():
    themes = ["longing and heartbreak", "unable to love again", "forbidden attraction"]

    assert "Lyrical themes" in embedding_text(themes[0])
    assert conservative_cost(themes) < 0.00001
