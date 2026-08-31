import pytest

from backend.app.embeddings import EmbeddingBatch
from backend.app.models import LyricsMeaning, Song
from backend.app.lyrics_analysis import CandidateVerdict
from backend.app.repository import blended_lyrics_similarity
from backend.app.retrieval import HybridRetriever, calibrated_lyrics_scores, lyrics_query_text
from backend.scripts.search_songs import SemanticSearchResult, format_result


class FakeEmbeddingProvider:
    dimensions = 3

    def embed_many(self, texts: list[str]) -> EmbeddingBatch:
        return EmbeddingBatch(vectors=[[0.1, 0.2, 0.3] for _ in texts], input_tokens=5)


class FakeIntentParser:
    def parse(self, query: str):
        from backend.app.models import Intent
        return Intent(
            search_description="bright uplifting high-valence music",
            moods=["joyful"],
            valence_target=0.85,
        )


class LyricsRequiredIntentParser:
    def parse(self, query: str):
        from backend.app.models import Intent, SignalWeights
        return Intent(
            search_description="gentle reflective vocal music",
            lyrics_search_description="forgiving someone and offering them a second chance",
            desired_lyrical_themes=["forgiveness"],
            lyrics_required=True,
            signal_weights=SignalWeights(semantic=0.5, lyrics=1.0),
        )


class IncidentalLyricsIntentParser:
    def parse(self, query: str):
        from backend.app.models import Intent, SignalWeights
        return Intent(
            search_description="bright joyful music",
            desired_lyrical_themes=["celebration"],
            lyrics_required=False,
            signal_weights=SignalWeights(semantic=1.0, lyrics=0.4),
        )


class FakeVectorRepository:
    def __init__(self) -> None:
        self.song = Song(
            id="one",
            title="Night Signal",
            artist="Test Artist",
            genre="ambient",
            genres=["ambient"],
            moods=["calm"],
            contexts=["study"],
            bpm=82,
            valence=0.8,
            description="calm atmospheric instrumental",
            accent="#000000",
        )
        self.received_vector: list[float] | None = None
        self.metadata_intent = None

    def list_songs(self, limit: int | None = None) -> list[Song]:
        return [self.song]

    def count(self) -> int:
        return 1

    def search_by_vector(self, query_vector: list[float], limit: int) -> list[tuple[Song, float]]:
        self.received_vector = query_vector
        return [(self.song, 0.81)]

    def search_by_metadata(self, query_vector, intent, limit):
        self.metadata_intent = intent
        return []

    def search_by_lyrics(self, lyrics_vector, sound_vector, limit):
        return []


class FakeLyricsRepository(FakeVectorRepository):
    def __init__(self) -> None:
        super().__init__()
        self.lyrics_search_called = False
        self.lyric_song = self.song.model_copy(update={
            "id": "lyrics", "title": "Second Chances", "artist": "Words Artist",
            "lyrics_evidence": "analyzed",
        })

    def search_by_vector(self, query_vector, limit):
        return [(self.song, 0.85), (self.lyric_song, 0.75)]

    def search_by_lyrics(self, lyrics_vector, sound_vector, limit):
        self.lyrics_search_called = True
        return [(self.lyric_song, 0.8, 0.75)]


class WeakLyricsRepository(FakeLyricsRepository):
    def search_by_lyrics(self, lyrics_vector, sound_vector, limit):
        self.lyrics_search_called = True
        return [(self.lyric_song, 0.33, 0.75)]


class ManyLyricsRepository(FakeLyricsRepository):
    def __init__(self) -> None:
        super().__init__()
        self.lyric_songs = [
            self.lyric_song.model_copy(update={
                "id": f"lyrics-{index}",
                "title": f"Candidate {index}",
                "artist": f"Artist {index}",
            })
            for index in range(5)
        ]

    def search_by_vector(self, query_vector, limit):
        return []

    def search_by_lyrics(self, lyrics_vector, sound_vector, limit):
        self.lyrics_search_called = True
        return [
            (song, 0.80 - index * 0.01, 0.70)
            for index, song in enumerate(self.lyric_songs)
        ]


class SparseMeaningRepository(ManyLyricsRepository):
    def __init__(self) -> None:
        super().__init__()
        meaning = LyricsMeaning(summary="The narrator offers forgiveness and another chance.")
        self.lyric_songs = [
            self.lyric_song.model_copy(update={
                "id": f"candidate-{index}",
                "title": f"Candidate {index}",
                "lyrics_meaning": meaning if index >= 12 else None,
            })
            for index in range(15)
        ]

    def search_by_lyrics(self, lyrics_vector, sound_vector, limit):
        return [
            (song, 0.80 - index * 0.001, 0.70)
            for index, song in enumerate(self.lyric_songs)
        ]


class CapturingLyricsVerifier:
    def __init__(self) -> None:
        self.song_ids: list[str] = []

    def verify(self, request, songs):
        self.song_ids = [song.id for song in songs]
        return {
            song.id: CandidateVerdict(
                song_id=song.id,
                verdict="match",
                confidence=0.9,
                reason="The stored meaning supports the request.",
            )
            for song in songs
        }


def test_format_result_shows_explanation_fields():
    result = SemanticSearchResult(
        title="Night Signal",
        artist="Test Artist",
        genres=["ambient", "acoustic"],
        bpm=82.4,
        similarity=0.81234,
    )

    rendered = format_result(1, result)

    assert "1. Night Signal — Test Artist" in rendered
    assert "ambient, acoustic" in rendered
    assert "82 BPM" in rendered
    assert "similarity 0.812" in rendered


def test_format_result_handles_missing_metadata():
    result = SemanticSearchResult(
        title="Unknown",
        artist="Someone",
        genres=[],
        bpm=None,
        similarity=0.5,
    )

    rendered = format_result(2, result)

    assert "unknown genre" in rendered
    assert "BPM unknown" in rendered


def test_live_retriever_uses_provider_vector_and_database_similarity():
    repository = FakeVectorRepository()
    retriever = HybridRetriever(repository, embedding_provider_factory=FakeEmbeddingProvider)

    intent, results = retriever.recommend("calm music for studying", limit=1)

    assert repository.received_vector == [0.1, 0.2, 0.3]
    assert repository.metadata_intent.contexts == ["study"]
    assert intent.contexts == ["study"]
    assert results[0].song.title == "Night Signal"
    assert results[0].breakdown.semantic == 81


def test_happy_intent_is_used_for_metadata_candidate_recall():
    repository = FakeVectorRepository()
    retriever = HybridRetriever(repository, embedding_provider_factory=FakeEmbeddingProvider)

    intent, _ = retriever.recommend("happy", limit=1)

    assert intent.moods == ["joyful"]
    assert repository.metadata_intent.moods == ["joyful"]


def test_diversity_limits_repeated_artist_when_alternatives_exist():
    base = FakeVectorRepository().song
    recommendations = []
    from backend.app.models import Recommendation, ScoreBreakdown
    for index, artist in enumerate(["Same Artist", "Same Artist", "Same Artist", "Different Artist"]):
        song = base.model_copy(update={"id": str(index), "title": f"Track {index}", "artist": artist})
        recommendation = Recommendation(
            song=song,
            score=80,
            explanation="test",
            matched_on=["joyful"],
            breakdown=ScoreBreakdown(semantic=70, mood=100, context=0, tempo=0, genre=0),
        )
        recommendations.append((1.0 - index * 0.1, recommendation))

    selected = HybridRetriever._select_diverse(recommendations, 3)

    assert [item.song.artist for item in selected].count("Same Artist") == 2
    assert any(item.song.artist == "Different Artist" for item in selected)


def test_llm_intent_controls_search_description_and_audio_profile():
    repository = FakeVectorRepository()
    retriever = HybridRetriever(
        repository,
        embedding_provider_factory=FakeEmbeddingProvider,
        intent_parser_factory=FakeIntentParser,
    )

    intent, results = retriever.recommend("happy", limit=1)

    assert intent.search_description == "bright uplifting high-valence music"
    assert intent.valence_target == 0.85
    assert results[0].breakdown.audio > 0


def test_literal_title_overlap_is_not_treated_as_vibe_evidence():
    assert HybridRetriever._accidental_title_overlap("happy", "Happy Together") is True
    assert HybridRetriever._accidental_title_overlap("happy", "Brighter Days") is False


def test_popularity_is_a_bounded_tie_breaker():
    base = FakeVectorRepository().song
    assert HybridRetriever._popularity_score(base.model_copy(update={"popularity": 80})) == 0.8
    assert HybridRetriever._popularity_score(base.model_copy(update={"popularity": 150})) == 1.0
    assert HybridRetriever._popularity_score(base.model_copy(update={"popularity": None})) == 0.0


def test_fit_normalizes_only_signals_that_apply():
    signals = {"semantic": (0.5, 0.6), "mood": (0.3, 1.0), "genre": (0.2, 0.0)}

    semantic_only = HybridRetriever._normalized_weighted_score(
        signals, {"semantic": True, "mood": False, "genre": False}
    )
    semantic_and_mood = HybridRetriever._normalized_weighted_score(
        signals, {"semantic": True, "mood": True, "genre": False}
    )

    assert semantic_only == 0.6
    assert round(semantic_and_mood, 6) == 0.75


def test_lyrics_query_is_separate_from_sound_query():
    from backend.app.models import Intent

    assert lyrics_query_text(Intent()) is None
    text = lyrics_query_text(Intent(desired_lyrical_themes=["forgiveness", "second chances"]))
    assert text is not None
    assert "forgiveness" in text
    assert "second chances" in text


def test_lyrics_query_preserves_detailed_narrative_over_generic_tags():
    from backend.app.models import Intent

    text = lyrics_query_text(Intent(
        lyrics_search_description="longing for someone who is now with somebody else",
        desired_lyrical_themes=["longing", "heartbreak"],
        lyrics_required=True,
    ))

    assert text is not None
    assert "now with somebody else" in text


def test_explicit_lyrics_request_excludes_unknown_lyrics(monkeypatch):
    monkeypatch.setenv("USE_LYRICS_EMBEDDINGS", "true")
    retriever = HybridRetriever(
        FakeLyricsRepository(),
        embedding_provider_factory=FakeEmbeddingProvider,
        intent_parser_factory=LyricsRequiredIntentParser,
    )

    intent, results = retriever.recommend("songs about forgiveness", limit=5)

    assert intent.lyrics_required is True
    assert [item.song.id for item in results] == ["lyrics"]
    assert results[0].breakdown.lyrics == 80
    assert "lyrical similarity" in results[0].matched_on


def test_incidental_lyrics_theme_does_not_activate_beta_channel(monkeypatch):
    monkeypatch.setenv("USE_LYRICS_EMBEDDINGS", "true")
    repository = FakeLyricsRepository()
    retriever = HybridRetriever(
        repository,
        embedding_provider_factory=FakeEmbeddingProvider,
        intent_parser_factory=IncidentalLyricsIntentParser,
    )

    _, results = retriever.recommend("happy music", limit=5)

    assert repository.lyrics_search_called is False
    assert all(item.breakdown.lyrics == 0 for item in results)


def test_explicit_lyrics_request_rejects_weak_nearest_neighbors(monkeypatch):
    monkeypatch.setenv("USE_LYRICS_EMBEDDINGS", "true")
    monkeypatch.setenv("LYRICS_MIN_SIMILARITY", "0.45")
    retriever = HybridRetriever(
        WeakLyricsRepository(),
        embedding_provider_factory=FakeEmbeddingProvider,
        intent_parser_factory=LyricsRequiredIntentParser,
    )

    _, results = retriever.recommend("songs about forgiveness", limit=5)

    assert results == []


def test_lyrics_candidates_must_stay_close_to_the_querys_best_match(monkeypatch):
    monkeypatch.setenv("LYRICS_MIN_SIMILARITY", "0.45")
    monkeypatch.setenv("LYRICS_RELATIVE_MARGIN", "0.05")

    retained = calibrated_lyrics_scores({"best": 0.52, "near": 0.48, "weak": 0.46})

    assert retained == {"best": 0.52, "near": 0.48}


def test_lyrics_focus_does_not_pad_the_playlist_to_the_requested_limit(monkeypatch):
    monkeypatch.setenv("USE_LYRICS_EMBEDDINGS", "true")
    monkeypatch.setenv("LYRICS_RESULT_LIMIT", "3")
    retriever = HybridRetriever(
        ManyLyricsRepository(),
        embedding_provider_factory=FakeEmbeddingProvider,
        intent_parser_factory=LyricsRequiredIntentParser,
    )

    _, results = retriever.recommend("songs about forgiveness", limit=5, focus="lyrics")

    assert len(results) == 3


def test_chunk_similarity_is_tempered_by_whole_song_meaning(monkeypatch):
    monkeypatch.setenv("LYRICS_CHUNK_WEIGHT", "0.65")

    score = blended_lyrics_similarity(chunk_similarity=0.70, whole_song_similarity=0.30)

    assert score == pytest.approx(0.56)


def test_sound_focus_never_activates_lyrics_channel(monkeypatch):
    monkeypatch.setenv("USE_LYRICS_EMBEDDINGS", "true")
    repository = FakeLyricsRepository()
    retriever = HybridRetriever(
        repository,
        embedding_provider_factory=FakeEmbeddingProvider,
        intent_parser_factory=LyricsRequiredIntentParser,
    )

    _, results = retriever.recommend("songs about forgiveness", limit=5, focus="sound")

    assert repository.lyrics_search_called is False
    assert all(item.breakdown.lyrics == 0 for item in results)


def test_lyrics_focus_forces_lyrics_for_an_ordinary_prompt(monkeypatch):
    monkeypatch.setenv("USE_LYRICS_EMBEDDINGS", "true")
    repository = FakeLyricsRepository()
    retriever = HybridRetriever(
        repository,
        embedding_provider_factory=FakeEmbeddingProvider,
        intent_parser_factory=FakeIntentParser,
    )

    intent, results = retriever.recommend("happy", limit=5, focus="lyrics")

    assert intent.lyrics_required is True
    assert intent.lyrics_search_description == "happy"
    assert repository.lyrics_search_called is True
    assert [item.song.id for item in results] == ["lyrics"]
    assert results[0].score == 80


def test_lyrics_focus_preserves_exact_user_narrative_over_parser_summary():
    from backend.app.models import Intent

    query = "intense forbidden attraction and desire"
    intent = Intent(
        lyrics_search_description="forbidden love",
        desired_lyrical_themes=["forbidden love"],
    )

    focused = HybridRetriever._apply_focus(intent, query, "lyrics")

    assert focused.lyrics_search_description == query


def test_balanced_focus_averages_sound_and_lyrics_fit(monkeypatch):
    monkeypatch.setenv("USE_LYRICS_EMBEDDINGS", "true")
    retriever = HybridRetriever(
        FakeLyricsRepository(),
        embedding_provider_factory=FakeEmbeddingProvider,
        intent_parser_factory=LyricsRequiredIntentParser,
    )

    _, results = retriever.recommend("songs about forgiveness", limit=5, focus="balanced")

    # The fake lyrical song has 0.75 sound similarity and 0.80 lyric similarity.
    assert results[0].score == 78


def test_narrative_verification_rejects_adjacent_themes_and_reranks_matches():
    scores, reasons = HybridRetriever._verified_lyrics_scores(
        {"exact": 0.48, "adjacent": 0.55, "partial": 0.50},
        {
            "exact": CandidateVerdict(
                song_id="exact", verdict="match", confidence=0.90,
                reason="The relationship is explicitly constrained and mutually desired.",
            ),
            "adjacent": CandidateVerdict(
                song_id="adjacent", verdict="no_match", confidence=0.95,
                reason="The song describes attraction but no forbidden relationship.",
            ),
            "partial": CandidateVerdict(
                song_id="partial", verdict="partial", confidence=0.70,
                reason="Desire is present, but the barrier is not established.",
            ),
        },
    )

    assert "adjacent" not in scores
    assert "partial" not in scores
    assert reasons["exact"].startswith("The relationship")


def test_narrative_verification_requires_a_verdict_for_every_retained_song():
    scores, _ = HybridRetriever._verified_lyrics_scores(
        {"judged": 0.50, "unjudged": 0.60},
        {
            "judged": CandidateVerdict(
                song_id="judged", verdict="match", confidence=0.80,
                reason="The requested situation is present.",
            ),
        },
    )

    assert set(scores) == {"judged"}


def test_verifier_limit_is_applied_after_filtering_for_meaning_records(monkeypatch):
    monkeypatch.setenv("USE_LYRICS_EMBEDDINGS", "true")
    monkeypatch.setenv("LYRICS_VERIFIER_CANDIDATES", "2")
    verifier = CapturingLyricsVerifier()
    retriever = HybridRetriever(
        SparseMeaningRepository(),
        embedding_provider_factory=FakeEmbeddingProvider,
        intent_parser_factory=LyricsRequiredIntentParser,
        lyrics_verifier_factory=lambda: verifier,
    )

    _, results = retriever.recommend("songs about forgiveness", limit=5, focus="lyrics")

    assert verifier.song_ids == ["candidate-12", "candidate-13"]
    assert {result.song.id for result in results} == {"candidate-12", "candidate-13"}
