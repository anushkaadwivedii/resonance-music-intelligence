from backend.app.db_models import SongRecord
from backend.app.repository import PostgresSongRepository
import httpx

from backend.scripts.embed_lyrics import lookup, lyric_text, match_confidence, representative_ids
from backend.scripts.analyze_lyrics_meaning import conservative_max_cost as meaning_max_cost
from backend.app.lyrics_analysis import (
    CandidateAssessment,
    ClaimCheck,
    OpenAILyricsVerifier,
    RequiredClaim,
    VerificationAssessment,
)


def test_lyrics_match_uses_identity_and_duration():
    song = SongRecord(
        source="test", source_id="one", title="Example Song", artist="Main Artist;Guest",
        album="Example Album", genres=[], moods=[], contexts=[], bpm=100,
        duration_ms=180_000, description="test",
    )
    result = {
        "trackName": "Example Song", "artistName": "Main Artist",
        "albumName": "Example Album", "duration": 180,
    }

    assert match_confidence(song, result) == 1.0


def test_synced_lyrics_are_cleaned_in_memory():
    result = {"plainLyrics": None, "syncedLyrics": "[00:01.00]First line\n[00:02.00]Second line"}

    assert lyric_text(result) == "First line\nSecond line"


def test_temporary_provider_failure_is_retried(monkeypatch):
    song = SongRecord(
        source="test", source_id="one", title="Example Song", artist="Main Artist",
        album="Example Album", genres=[], moods=[], contexts=[], bpm=100,
        duration_ms=180_000, description="test",
    )
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, request=request, json={"trackName": "Example Song"})

    monkeypatch.setattr("backend.scripts.embed_lyrics.time.sleep", lambda _: None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = lookup(client, song)

    assert attempts == 2
    assert result == {"trackName": "Example Song"}


def test_representative_sample_spans_popularity_and_genres():
    rows = [
        (index, 100 - index, ["pop" if index % 2 else "rock"], f"song-{index}")
        for index in range(40)
    ]

    selected = representative_ids(rows, 8)
    selected_popularities = [100 - index for index in selected]
    selected_genres = {"pop" if index % 2 else "rock" for index in selected}

    assert len(selected) == 8
    assert max(selected_popularities) >= 91
    assert min(selected_popularities) <= 70
    assert selected_genres == {"pop", "rock"}


def test_repository_exposes_lyrics_evidence_without_exposing_lyrics():
    def record(status: str | None) -> SongRecord:
        return SongRecord(
            source="test", source_id=f"song-{status}", title="Example Song", artist="Artist",
            album=None, genres=[], moods=[], contexts=[], bpm=100, description="test",
            lyrics_lookup_status=status,
        )

    assert PostgresSongRepository._to_song(record("embedded")).lyrics_evidence == "analyzed"
    assert PostgresSongRepository._to_song(record("not_found")).lyrics_evidence == "unavailable"
    assert PostgresSongRepository._to_song(record("no_lyrics")).lyrics_evidence == "unavailable"
    assert PostgresSongRepository._to_song(record(None)).lyrics_evidence == "not_analyzed"


def test_meaning_analysis_cost_guard_includes_input_and_output_tokens():
    assert meaning_max_cost(0) == 0
    assert meaning_max_cost(20) > 0
    assert meaning_max_cost(20) < meaning_max_cost(100)


def test_verifier_cannot_match_when_a_required_claim_is_missing():
    assessment = VerificationAssessment(
        required_claims=[
            RequiredClaim(claim_id="desire", description="Strong mutual attraction"),
            RequiredClaim(claim_id="barrier", description="A real prohibition or obstacle"),
        ],
        candidates=[
            CandidateAssessment(
                song_id="ordinary-attraction",
                checks=[
                    ClaimCheck(claim_id="desire", support="supported", confidence=0.9),
                    ClaimCheck(claim_id="barrier", support="not_stated", confidence=0.95),
                ],
                reason="Attraction is present, but no forbidden barrier is established.",
            )
        ],
    )

    verdict = OpenAILyricsVerifier._derive_verdicts(
        assessment, {"ordinary-attraction"}
    )["ordinary-attraction"]

    assert verdict.verdict == "partial"


def test_verifier_rejects_supported_claim_with_fabricated_evidence():
    assessment = VerificationAssessment(
        required_claims=[
            RequiredClaim(claim_id="heartbreak", description="Pain from a past relationship"),
        ],
        candidates=[
            CandidateAssessment(
                song_id="attraction-song",
                checks=[
                    ClaimCheck(
                        claim_id="heartbreak",
                        support="supported",
                        confidence=0.95,
                        evidence_ids=["fabricated"],
                    ),
                ],
                reason="The record supposedly describes heartbreak.",
            )
        ],
    )

    verdict = OpenAILyricsVerifier._derive_verdicts(
        assessment,
        {"attraction-song"},
        {"attraction-song": {"summary"}},
    )["attraction-song"]

    assert verdict.verdict == "no_match"


def test_verifier_accepts_supported_claim_with_valid_evidence_id():
    assessment = VerificationAssessment(
        required_claims=[
            RequiredClaim(claim_id="rival", description="The former partner is with somebody else"),
        ],
        candidates=[
            CandidateAssessment(
                song_id="drivers-license",
                checks=[
                    ClaimCheck(
                        claim_id="rival",
                        support="supported",
                        confidence=0.9,
                        evidence_ids=["summary"],
                    ),
                ],
                reason="The ex has moved on with somebody else.",
            )
        ],
    )

    verdict = OpenAILyricsVerifier._derive_verdicts(
        assessment,
        {"drivers-license"},
        {"drivers-license": {"summary", "emotional_arc"}},
    )["drivers-license"]

    assert verdict.verdict == "match"
