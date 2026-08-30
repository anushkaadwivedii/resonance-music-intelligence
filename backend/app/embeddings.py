"""Embedding providers and the single boundary where billable calls occur."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# Increment this whenever embedding_text() changes meaningfully.
EMBEDDING_TEXT_VERSION = 2
SOUND_EMBEDDING_TEXT_VERSION = 1
LYRICS_EMBEDDING_TEXT_VERSION = 1


def configured_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


@dataclass
class EmbeddingBatch:
    vectors: list[list[float]]
    input_tokens: int


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed_many(self, texts: list[str]) -> EmbeddingBatch: ...


class OpenAIEmbeddingProvider:
    """Thin adapter around the OpenAI embeddings endpoint."""

    dimensions = 1536

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env; never commit or paste it.")
        self.model = model or configured_embedding_model()
        self.client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)

    def embed_many(self, texts: list[str]) -> EmbeddingBatch:
        # This is the only billable operation in the embedding ingestion path.
        response = self.client.embeddings.create(model=self.model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [item.embedding for item in ordered]
        if len(vectors) != len(texts):
            raise RuntimeError(f"Expected {len(texts)} embeddings, received {len(vectors)}")
        if any(len(vector) != self.dimensions for vector in vectors):
            raise RuntimeError(f"Embedding model must return {self.dimensions} dimensions")
        return EmbeddingBatch(vectors=vectors, input_tokens=response.usage.total_tokens)
