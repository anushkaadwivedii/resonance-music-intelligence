from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models import Song
from backend.app.repository import InMemorySongRepository, repository as postgres_repository
from backend.app.retrieval import HybridRetriever, parse_intent, retriever, semantic_query_text


# Tests must be deterministic and must never make billable API requests.
retriever.embedding_provider_factory = None
retriever.intent_parser_factory = None
# Retrieval unit tests use a bounded catalog even when production has ~90k rows.
retriever.repository = InMemorySongRepository(postgres_repository.list_songs(100))


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["catalog_size"] >= 15


def test_intent_understands_context_and_tempo():
    intent = parse_intent("dreamy songs for a rainy late night, but not too slow")
    assert "dreamy" in intent.moods
    assert "rainy day" in intent.contexts
    assert "late night" in intent.contexts
    assert intent.bpm_min == 80


def test_numeric_bpm_limit_is_an_explicit_constraint():
    intent = parse_intent("calm acoustic songs under 100 BPM")
    assert intent.bpm_max == 100
    assert intent.bpm_is_explicit is True


def test_single_mood_word_expands_into_meaning_not_literal_title_search():
    intent = parse_intent("happy")
    expanded = semantic_query_text("happy", intent)
    assert expanded != "happy"
    assert "uplifting" in expanded
    assert "high-valence" in expanded
    assert intent.title_contains is None


def test_title_filter_is_only_created_for_an_explicit_title_request():
    intent = parse_intent("songs with the word happy in the title")
    assert intent.title_contains == "happy"


def test_explicit_title_request_filters_results_by_title():
    base = Song(
        id="one", title="Happy Together", artist="One", genre="pop", genres=["pop"],
        moods=["joyful"], contexts=[], bpm=120, description="bright upbeat pop",
        accent="#000000",
    )
    other = base.model_copy(update={"id": "two", "title": "Brighter Days", "artist": "Two"})
    local = HybridRetriever(InMemorySongRepository([base, other]))

    _, results = local.recommend("songs with the word happy in the title", 20)

    assert [item.song.title for item in results] == ["Happy Together"]


def test_calm_context_softly_prefers_suitable_tempo_and_understands_double_time():
    intent = parse_intent("calm music for studying")
    assert retriever._tempo_score(intent, 90) == 1.0
    assert retriever._tempo_score(intent, 140) < 1.0
    assert retriever._tempo_score(intent, 182) == 0.75


def test_genre_exclusion_is_enforced():
    intent, results = retriever.recommend("calm music for studying, not pop", 12)
    assert "pop" in intent.excluded_genres
    assert all("pop" not in result.song.genre for result in results)


def test_recommendation_response_has_explanations():
    response = client.post("/api/recommendations", json={"query": "cinematic music for a night drive", "limit": 5})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["recommendations"]) == 5
    assert all(item["explanation"] for item in payload["recommendations"])
    assert all(0 <= item["score"] <= 100 for item in payload["recommendations"])


def test_results_do_not_repeat_same_title_and_artist():
    _, results = retriever.recommend("calm acoustic music for studying", 12)
    identities = [(item.song.title.casefold(), item.song.artist.casefold()) for item in results]
    assert len(identities) == len(set(identities))
