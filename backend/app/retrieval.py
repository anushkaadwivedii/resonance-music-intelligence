import hashlib
import math
import os
import re
from collections import Counter
from collections.abc import Callable

from .embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from .intent_parser import IntentParser, OpenAIIntentParser
from .models import Intent, Recommendation, ScoreBreakdown, Song
from .repository import SongRepository, repository


MOODS = {"balanced", "calm", "energetic", "intimate", "joyful", "melancholic", "playful", "sad", "happy", "chill", "romantic", "dreamy", "reflective"}
COMMON_GENRES = {"acoustic", "ambient", "blues", "classical", "country", "electronic", "folk", "hip-hop", "indie", "jazz", "metal", "pop", "r&b", "rap", "reggae", "rock", "soul"}
KNOWN_CONTEXTS = {"coding", "late night", "meditation", "morning", "night drive", "party", "quiet morning", "rainy day", "reading", "road trip", "running", "sleep", "study", "travel", "walking", "workout"}
CONTEXT_ALIASES = {
    "studying": "study", "focus": "study", "working": "coding",
    "drive": "night drive", "driving": "night drive", "rain": "rainy day",
    "rainy": "rainy day", "gym": "workout", "exercise": "workout",
    "bed": "sleep", "relaxing": "meditation", "trip": "road trip",
}
MOOD_ALIASES = {
    "sad": "melancholic", "happy": "joyful", "chill": "calm", "peaceful": "calm",
    "relaxed": "calm", "upbeat": "energetic", "positive": "hopeful", "emotional": "bittersweet",
}
STOP_WORDS = {"a", "an", "and", "are", "but", "for", "i", "in", "is", "it", "like", "me", "music", "of", "on", "or", "songs", "some", "that", "the", "to", "with"}
MOOD_SEARCH_DESCRIPTIONS = {
    "joyful": "bright uplifting optimistic celebratory warm positive high-valence feel-good music",
    "calm": "peaceful soothing gentle relaxed low-energy comforting music",
    "energetic": "driving lively powerful exciting high-energy motivating music",
    "melancholic": "wistful reflective bittersweet somber emotionally heavy music",
    "intimate": "close warm tender personal vulnerable acoustic music",
    "playful": "fun bouncy lighthearted cheeky colorful music",
    "romantic": "tender affectionate warm devoted romantic music",
    "dreamy": "ethereal hazy atmospheric floating immersive music",
    "reflective": "thoughtful introspective contemplative emotionally nuanced music",
}


def lyrics_min_similarity() -> float:
    """Configurable beta gate: nearest does not necessarily mean relevant."""
    try:
        configured = float(os.getenv("LYRICS_MIN_SIMILARITY", "0.45"))
    except ValueError:
        configured = 0.45
    return max(0.0, min(1.0, configured))


def tokenize(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOP_WORDS]


def is_sparse_mood_query(query: str, intent: Intent) -> bool:
    return bool(
        intent.moods
        and len(tokenize(query)) <= 2
        and not intent.genres
        and not intent.contexts
        and intent.bpm_min is None
        and intent.bpm_max is None
        and intent.artist_reference is None
    )


def semantic_query_text(query: str, intent: Intent) -> str:
    """Expand underspecified mood words into acoustic/emotional meaning."""
    if intent.search_description:
        return intent.search_description
    if not is_sparse_mood_query(query, intent):
        return query
    descriptions = [MOOD_SEARCH_DESCRIPTIONS.get(mood, f"{mood} emotional music") for mood in intent.moods]
    return ". ".join(descriptions)


def lyrics_query_text(intent: Intent) -> str | None:
    """Represent requested lyrical subject matter separately from sound."""
    if not intent.desired_lyrical_themes:
        return None
    return "Lyrical themes, story, and emotional meaning: " + ", ".join(intent.desired_lyrical_themes)


def explicit_title_request(query: str) -> str | None:
    """Extract a title constraint only when the user names the title field."""
    patterns = [
        r"(?:songs?|tracks?)\s+with\s+(?:the\s+)?(?:word|phrase)\s+[\"']?(.+?)[\"']?\s+in\s+(?:the|their)\s+titles?\b",
        r"(?:songs?|tracks?)\s+(?:with|containing|that contain)\s+[\"']?(.+?)[\"']?\s+in\s+(?:the|their)\s+titles?\b",
        r"(?:title|song title)\s+(?:contains?|includes?|has)\s+[\"']?(.+?)[\"']?(?:$|[,.])",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" \t\"'")
    return None


class HashingEmbedder:
    """Small dependency-free semantic-ish encoder suitable for the demo catalog."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        terms = tokenize(text)
        features = terms + [f"{a}_{b}" for a, b in zip(terms, terms[1:])]
        vector = [0.0] * self.dimensions
        for feature, count in Counter(features).items():
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            sign = 1 if digest[0] & 1 else -1
            vector[index] += sign * (1 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector)) or 1
        return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def parse_intent(query: str, songs: list[Song] | None = None) -> Intent:
    lowered = query.lower()
    songs = songs or []
    moods = set()
    for mood in MOODS:
        if mood in lowered:
            moods.add(MOOD_ALIASES.get(mood, mood))
    for word, canonical in MOOD_ALIASES.items():
        if word in lowered:
            moods.add(canonical)

    excluded_genres = set()
    genres = set()
    catalog_genres = {genre for song in songs for genre in (song.genres or [song.genre])}
    genres_to_check = COMMON_GENRES | catalog_genres
    for genre in sorted(genres_to_check, key=len, reverse=True):
        if re.search(rf"\b(?:not|no|less)\s+(?:too\s+)?{re.escape(genre)}\b", lowered):
            excluded_genres.add(genre)
        elif re.search(rf"\b{re.escape(genre)}\b", lowered):
            genres.add(genre)

    contexts = set()
    for alias, canonical in CONTEXT_ALIASES.items():
        if alias in lowered:
            contexts.add(canonical)
    for context in KNOWN_CONTEXTS | {item for song in songs for item in song.contexts}:
        if context in lowered:
            contexts.add(context)

    bpm_min = bpm_max = None
    bpm_is_explicit = False
    range_match = re.search(r"\bbetween\s+(\d{2,3})\s+(?:and|to|-)\s+(\d{2,3})\s*bpm\b", lowered)
    max_match = re.search(r"\b(?:under|below|less than|up to|maximum|max)\s+(\d{2,3})\s*bpm\b", lowered)
    min_match = re.search(r"\b(?:over|above|more than|at least|minimum|min)\s+(\d{2,3})\s*bpm\b", lowered)
    if range_match:
        bpm_min, bpm_max = sorted((int(range_match.group(1)), int(range_match.group(2))))
        bpm_is_explicit = True
    elif max_match:
        bpm_max = int(max_match.group(1))
        bpm_is_explicit = True
    elif min_match:
        bpm_min = int(min_match.group(1))
        bpm_is_explicit = True
    else:
        if any(phrase in lowered for phrase in ["not too slow", "mid tempo", "mid-tempo"]):
            bpm_min = 80
        elif any(phrase in lowered for phrase in ["slow", "gentle", "quiet"]):
            bpm_max = 90
        if any(phrase in lowered for phrase in ["fast", "high energy", "upbeat"]):
            bpm_min = max(bpm_min or 0, 110)

    artist_reference = None
    for song in songs:
        if song.artist.lower() in lowered:
            artist_reference = song.artist
            break

    return Intent(
        moods=sorted(moods), genres=sorted(genres), contexts=sorted(contexts),
        excluded_genres=sorted(excluded_genres), title_contains=explicit_title_request(query),
        artist_reference=artist_reference,
        bpm_min=bpm_min, bpm_max=bpm_max, bpm_is_explicit=bpm_is_explicit,
    )


def song_text(song: Song) -> str:
    # Identity is deliberately excluded. Title and artist are filters, not
    # evidence for how a recording sounds or feels.
    return " ".join([song.genre, *song.moods, *song.contexts, song.description])


class HybridRetriever:
    def __init__(
        self,
        song_repository: SongRepository = repository,
        embedding_provider_factory: Callable[[], EmbeddingProvider] | None = None,
        intent_parser_factory: Callable[[], IntentParser] | None = None,
    ) -> None:
        self.repository = song_repository
        self.embedding_provider_factory = embedding_provider_factory
        self.intent_parser_factory = intent_parser_factory
        self.embedder = HashingEmbedder()
        self.song_vectors: dict[str, list[float]] = {}

    def recommend(self, query: str, limit: int = 7) -> tuple[Intent, list[Recommendation]]:
        if self.embedding_provider_factory is not None:
            # This is the live, billable path: one short query embedding per search.
            intent = self._understand(query)
            embedding_query = semantic_query_text(query, intent)
            requested_lyrics_text = lyrics_query_text(intent)
            lyrics_channel_active = bool(
                intent.lyrics_required
                and requested_lyrics_text
                and os.getenv("USE_LYRICS_EMBEDDINGS", "false").lower() == "true"
            )
            embedding_inputs = [embedding_query]
            if lyrics_channel_active and requested_lyrics_text:
                embedding_inputs.append(requested_lyrics_text)
            query_batch = self.embedding_provider_factory().embed_many(embedding_inputs)
            query_vector = query_batch.vectors[0]
            lyrics_vector = query_batch.vectors[1] if lyrics_channel_active else None
            semantic_rows = self.repository.search_by_vector(query_vector, max(100, limit * 20))
            metadata_rows = self.repository.search_by_metadata(query_vector, intent, max(200, limit * 30))
            lyrics_rows = (
                self.repository.search_by_lyrics(lyrics_vector, query_vector, max(100, limit * 20))
                if lyrics_vector is not None
                else []
            )

            # Union both retrieval channels. If a song appears in both, retain
            # its strongest measured vector similarity and only rank it once.
            merged: dict[str, Song] = {}
            semantic_scores: dict[str, float] = {}
            lyrics_scores: dict[str, float] = {}
            for song, similarity in [*semantic_rows, *metadata_rows]:
                merged[song.id] = song
                semantic_scores[song.id] = max(similarity, semantic_scores.get(song.id, 0.0))
            for song, lyrics_similarity, sound_similarity in lyrics_rows:
                if lyrics_similarity < lyrics_min_similarity():
                    continue
                merged[song.id] = song
                lyrics_scores[song.id] = lyrics_similarity
                semantic_scores[song.id] = max(sound_similarity, semantic_scores.get(song.id, 0.0))
            songs = list(merged.values())
        else:
            # Deterministic and free: used by unit tests and offline experiments.
            catalog = self.repository.list_songs()
            intent = parse_intent(query, catalog)
            query_vector = self.embedder.embed(query)
            songs = catalog
            semantic_scores = {}
            lyrics_scores = {}
            lyrics_channel_active = False

        candidates = []
        seen_recordings: set[tuple[str, str]] = set()

        for song in songs:
            recording_key = (song.title.casefold().strip(), song.artist.casefold().strip())
            if recording_key in seen_recordings:
                continue
            seen_recordings.add(recording_key)
            if any(excluded in song.genre for excluded in intent.excluded_genres):
                continue
            if intent.title_contains and intent.title_contains.casefold() not in song.title.casefold():
                continue
            if intent.bpm_is_explicit and self._outside_explicit_tempo(intent, song.bpm):
                continue
            if self.embedding_provider_factory is not None:
                semantic = max(0.0, min(1.0, semantic_scores[song.id]))
            else:
                if song.id not in self.song_vectors:
                    self.song_vectors[song.id] = self.embedder.embed(song_text(song))
                raw_semantic = max(0.0, cosine(query_vector, self.song_vectors[song.id]))
                semantic = min(1.0, raw_semantic * 2.5)
            mood = self._overlap(intent.moods, song.moods)
            context = self._overlap(intent.contexts, song.contexts)
            genre = max((1.0 if wanted in song.genre else 0.0 for wanted in intent.genres), default=0.0)
            tempo = self._tempo_score(intent, song.bpm)
            artist = 1.0 if intent.artist_reference and song.artist == intent.artist_reference else 0.0
            audio = self._audio_profile_score(intent, song)
            popularity = self._popularity_score(song)
            lyrics_available = song.id in lyrics_scores
            if lyrics_channel_active and intent.lyrics_required and not lyrics_available:
                continue
            lyrics = max(0.0, min(1.0, lyrics_scores.get(song.id, 0.0)))

            has_audio_targets = any(target is not None for target in [
                intent.valence_target, intent.energy_target, intent.danceability_target,
                intent.acousticness_target, intent.instrumentalness_target,
            ])
            priorities = intent.signal_weights
            active = {
                "semantic": (priorities.semantic, semantic),
                "mood": (priorities.mood, mood),
                "audio": (priorities.audio, audio),
                "context": (priorities.context, context),
                "tempo": (priorities.tempo, tempo),
                "genre": (priorities.genre, genre),
                "artist": (priorities.artist, artist),
                "lyrics": (priorities.lyrics, lyrics),
            }

            tempo_applies = bool(
                intent.bpm_min is not None
                or intent.bpm_max is not None
                or {"calm", "energetic"} & set(intent.moods)
                or {"study", "reading", "sleep", "meditation", "workout", "running", "party"}
                & set(intent.contexts)
            )
            applicable = {
                "semantic": True,
                "mood": bool(intent.moods),
                "audio": has_audio_targets,
                "context": bool(intent.contexts),
                "tempo": tempo_applies,
                "genre": bool(intent.genres),
                "artist": intent.artist_reference is not None,
                # Missing lyrics are unknown, not a zero-quality match. For an
                # explicit lyrical request they were filtered above; otherwise
                # this evidence participates only when it actually exists.
                "lyrics": lyrics_channel_active and lyrics_available,
            }
            fit_score = self._normalized_weighted_score(active, applicable)
            # Existing v2 catalog vectors contain titles. Until the sound-only
            # vectors are rebuilt, cancel their most visible failure mode:
            # literal query/title overlap is not positive evidence unless the
            # user explicitly asked to search titles.
            if not intent.title_contains and self._accidental_title_overlap(query, song.title):
                fit_score -= 0.08
            fit_score = max(0.0, min(1.0, fit_score))
            # Popularity helps order near-ties, but is deliberately excluded
            # from the displayed fit because familiarity is not relevance.
            popularity_weight = priorities.popularity_tiebreak
            ranking_score = (1 - popularity_weight) * fit_score + popularity_weight * popularity
            score_pct = round(fit_score * 100)
            matched = self._matched_evidence(intent, song, semantic, lyrics if lyrics_available else None)
            candidates.append((ranking_score, Recommendation(
                song=song,
                score=score_pct,
                explanation=self._explain(song, intent, matched),
                matched_on=matched,
                breakdown=ScoreBreakdown(
                    semantic=round(semantic * 100), mood=round(mood * 100),
                    context=round(context * 100), tempo=round(tempo * 100),
                    genre=round(genre * 100), audio=round(audio * 100),
                    popularity=round(popularity * 100),
                    lyrics=round(lyrics * 100),
                ),
            )))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return intent, self._select_diverse(candidates, limit)

    @staticmethod
    def _select_diverse(candidates: list[tuple[float, Recommendation]], limit: int) -> list[Recommendation]:
        """Prefer variety while still filling the requested playlist."""
        selected: list[Recommendation] = []
        overflow: list[Recommendation] = []
        artist_counts: Counter[str] = Counter()
        for _, recommendation in candidates:
            # Credits often append orchestras or featured performers. The first
            # credit is the primary artist/composer and is the useful diversity key.
            artist_key = recommendation.song.artist.split(";", 1)[0].casefold().strip()
            if artist_counts[artist_key] >= 2:
                overflow.append(recommendation)
                continue
            selected.append(recommendation)
            artist_counts[artist_key] += 1
            if len(selected) == limit:
                return selected
        if len(selected) < limit:
            selected.extend(overflow[: limit - len(selected)])
        return selected

    def _understand(self, query: str) -> Intent:
        if self.intent_parser_factory is None:
            return parse_intent(query)
        try:
            return self.intent_parser_factory().parse(query)
        except Exception:
            # Search remains usable during model/network/schema failures.
            import logging
            logging.getLogger(__name__).exception("LLM intent parsing failed; using deterministic fallback")
            return parse_intent(query)

    @staticmethod
    def _overlap(wanted: list[str], actual: list[str]) -> float:
        if not wanted:
            return 0.0
        return len(set(wanted) & set(actual)) / len(set(wanted))

    @staticmethod
    def _normalized_weighted_score(
        signals: dict[str, tuple[float, float]], applicable: dict[str, bool]
    ) -> float:
        selected = [
            (weight, value)
            for name, (weight, value) in signals.items()
            if applicable.get(name, False)
        ]
        total_weight = sum(weight for weight, _ in selected)
        if total_weight == 0:
            return 0.0
        return sum(weight * value for weight, value in selected) / total_weight

    @staticmethod
    def _accidental_title_overlap(query: str, title: str) -> bool:
        query_terms = {term for term in tokenize(query) if len(term) >= 4}
        title_terms = set(tokenize(title))
        return bool(query_terms & title_terms)

    @staticmethod
    def _audio_profile_score(intent: Intent, song: Song) -> float:
        pairs = [
            (intent.valence_target, song.valence),
            (intent.energy_target, song.energy),
            (intent.danceability_target, song.danceability),
            (intent.acousticness_target, song.acousticness),
            (intent.instrumentalness_target, song.instrumentalness),
        ]
        similarities = [1 - abs(target - actual) for target, actual in pairs if target is not None and actual is not None]
        return sum(similarities) / len(similarities) if similarities else 0.0

    @staticmethod
    def _popularity_score(song: Song) -> float:
        if song.popularity is None:
            return 0.0
        return max(0.0, min(1.0, song.popularity / 100))

    @staticmethod
    def _range_tempo_score(bpm: float, bpm_min: int | None, bpm_max: int | None) -> float:
        if bpm_min is not None and bpm < bpm_min:
            return max(0.0, 1 - (bpm_min - bpm) / 30)
        if bpm_max is not None and bpm > bpm_max:
            return max(0.0, 1 - (bpm - bpm_max) / 40)
        return 1.0

    @classmethod
    def _tempo_score(cls, intent: Intent, bpm: float) -> float:
        bpm_min, bpm_max = intent.bpm_min, intent.bpm_max
        if bpm_min is None and bpm_max is None:
            if "calm" in intent.moods or {"study", "reading", "sleep", "meditation"} & set(intent.contexts):
                bpm_min, bpm_max = 60, 110
            elif "energetic" in intent.moods or {"workout", "running", "party"} & set(intent.contexts):
                bpm_min = 110
            else:
                return 0.0

        raw_score = cls._range_tempo_score(bpm, bpm_min, bpm_max)
        if intent.bpm_is_explicit or bpm < 160:
            return raw_score

        # Some catalogs encode the same pulse in double time. Treat half-time
        # as a weaker possibility for soft mood/context preferences only.
        half_time_score = cls._range_tempo_score(bpm / 2, bpm_min, bpm_max) * 0.75
        return max(raw_score, half_time_score)

    @staticmethod
    def _outside_explicit_tempo(intent: Intent, bpm: float) -> bool:
        return bool(
            (intent.bpm_min is not None and bpm < intent.bpm_min)
            or (intent.bpm_max is not None and bpm > intent.bpm_max)
        )

    @staticmethod
    def _matched_evidence(
        intent: Intent, song: Song, semantic: float, lyrics: float | None = None
    ) -> list[str]:
        matches = [mood for mood in intent.moods if mood in song.moods]
        matches += [context for context in intent.contexts if context in song.contexts]
        matches += [genre for genre in intent.genres if genre in song.genre]
        if lyrics is not None and lyrics > 0 and intent.desired_lyrical_themes:
            matches.append(f"lyrics: {intent.desired_lyrical_themes[0]}")
        if intent.bpm_min and song.bpm >= intent.bpm_min:
            matches.append(f"{round(song.bpm)} BPM")
        if intent.bpm_max and song.bpm <= intent.bpm_max:
            matches.append(f"{round(song.bpm)} BPM")
        if not matches and semantic > 0:
            matches.append("semantic match")
        return matches[:4]

    @staticmethod
    def _explain(song: Song, intent: Intent, matched: list[str]) -> str:
        lead = f"Its {song.description}."
        if matched:
            return f"{lead} It connects to your request through {', '.join(matched)}."
        if intent.excluded_genres:
            return f"{lead} It also stays outside the excluded {', '.join(intent.excluded_genres)} sound."
        return f"{lead} The overall tone is a strong semantic fit for the feeling you described."


provider_factory = OpenAIEmbeddingProvider if os.getenv("AI_PROVIDER", "openai").lower() == "openai" else None
intent_factory = OpenAIIntentParser if os.getenv("INTENT_PROVIDER", "rules").lower() == "openai" else None
retriever = HybridRetriever(embedding_provider_factory=provider_factory, intent_parser_factory=intent_factory)
