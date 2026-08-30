from backend.scripts.ingest_tracks import ImportedSong, create_description, derive_contexts, derive_moods


def song(**overrides) -> ImportedSong:
    values = {
        "source_id": "track-1", "title": "Test Track", "artist": "Test Artist",
        "album": "Test Album", "genres": {"indie"}, "bpm": 120,
        "energy": 0.8, "valence": 0.8, "danceability": 0.8,
    }
    values.update(overrides)
    return ImportedSong(**values)


def test_audio_features_create_moods():
    assert derive_moods(song()) == ["energetic", "joyful", "playful"]


def test_audio_features_create_contexts():
    assert derive_contexts(song()) == ["party", "workout"]


def test_description_uses_factual_features():
    result = create_description(song(), ["energetic", "joyful"])
    assert "Test Artist" in result
    assert "120 BPM" in result
    assert "80% energy" in result
