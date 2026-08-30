from backend.app.db_models import SongRecord
from backend.app.embeddings import EMBEDDING_TEXT_VERSION
from backend.scripts.embed_songs import conservative_token_estimate, embedding_text, estimated_cost
from backend.scripts.embed_sound_songs import sound_embedding_text


def test_embedding_text_contains_retrieval_evidence():
    song = SongRecord(
        source="test", source_id="one", title="Night Signal", artist="Test Artist",
        genres=["ambient"], moods=["calm"], contexts=["study"],
        bpm=82, energy=0.25, danceability=0.4, acousticness=0.8,
        instrumentalness=0.9, description="A calm ambient track.",
    )
    text = embedding_text(song)
    assert "Title: Night Signal" in text
    assert "Genres: ambient" in text
    assert "Contexts: study" in text
    assert "Tempo: 82 BPM" in text
    assert "25% energy" in text
    assert "80% acousticness" in text


def test_cost_estimate_is_small_and_conservative():
    tokens = conservative_token_estimate(["a" * 400])
    assert tokens == 125
    assert estimated_cost(1_000_000) == 0.02


def test_embedding_recipe_has_an_explicit_version():
    assert EMBEDDING_TEXT_VERSION >= 2


def test_sound_embedding_excludes_song_identity():
    song = SongRecord(
        source="test", source_id="one", title="Happy Together", artist="The Turtles",
        album="Identity Album", genres=["pop"], moods=["joyful"], contexts=["party"],
        bpm=120, energy=0.8, danceability=0.7, valence=0.9,
        acousticness=0.1, instrumentalness=0.0, speechiness=0.05,
        description="A description that names The Turtles.",
    )

    text = sound_embedding_text(song)

    assert "Happy Together" not in text
    assert "The Turtles" not in text
    assert "Identity Album" not in text
    assert "Genres: pop" in text
    assert "90% valence" in text
