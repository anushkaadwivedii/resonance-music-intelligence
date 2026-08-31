from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAIError

from .models import RecommendationRequest, RecommendationResponse, Song
from .repository import repository
from .retrieval import retriever


app = FastAPI(title="Resonance API", version="0.1.0", description="Explainable hybrid music retrieval")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "catalog_size": repository.count()}


@app.get("/api/songs", response_model=list[Song])
def songs(limit: int = 100) -> list[Song]:
    safe_limit = min(max(limit, 1), 500)
    return repository.list_songs(safe_limit)


@app.post("/api/recommendations", response_model=RecommendationResponse)
def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    try:
        intent, results = retriever.recommend(request.query, request.limit, request.focus)
    except (OpenAIError, RuntimeError) as error:
        raise HTTPException(
            status_code=503,
            detail="The semantic search provider is temporarily unavailable.",
        ) from error
    signals = [*intent.moods, *intent.contexts, *intent.genres]
    focus = ", ".join(signals[:3]) if signals else "the atmosphere you described"
    if not results and intent.lyrics_required:
        themes = ", ".join(intent.desired_lyrical_themes[:2]) or "that lyrical theme"
        summary = (
            f"I couldn't find a confident match about {themes} in the current lyrics beta. "
            "The catalog is still limited, so I left the playlist empty instead of guessing."
        )
    elif request.focus == "lyrics" and any(item.lyrics_verified for item in results):
        summary = (
            f"Found {len(results)} lyric matches. Vector search retrieved the candidates, "
            "and a second-stage check confirmed the requested narrative."
        )
    elif request.focus == "lyrics":
        summary = (
            f"Found {len(results)} passage-level lyric candidates. "
            "They are ranked by semantic similarity, not a verified interpretation."
        )
    else:
        summary = (
            f"Found {len(results)} tracks matching {focus}. "
            "The ranking combines semantic similarity with relevant tempo, mood, and context signals."
        )
    return RecommendationResponse(query=request.query, summary=summary, intent=intent, recommendations=results)
