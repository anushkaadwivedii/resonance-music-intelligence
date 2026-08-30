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
        intent, results = retriever.recommend(request.query, request.limit)
    except OpenAIError as error:
        raise HTTPException(
            status_code=503,
            detail="The semantic search provider is temporarily unavailable.",
        ) from error
    signals = [*intent.moods, *intent.contexts, *intent.genres]
    focus = ", ".join(signals[:3]) if signals else "the atmosphere you described"
    summary = f"I found {len(results)} tracks shaped around {focus}. I balanced meaning with tempo, mood, and context so the set feels cohesive without sounding repetitive."
    return RecommendationResponse(query=request.query, summary=summary, intent=intent, recommendations=results)
