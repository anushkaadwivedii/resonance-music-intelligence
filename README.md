# Resonance — Conversational Music Intelligence

Resonance is a full-stack MVP for explainable, natural-language music discovery. A user describes a moment (for example, “warm indie songs for a rainy train ride, but not too slow”), and the app interprets the intent, performs hybrid semantic/metadata retrieval, and builds an editable playlist with a reason for every recommendation.

## What is implemented

- Conversational React interface with suggestion prompts, loading/error states, playlist editing, and “Why this song?” details
- FastAPI API with typed request/response models and generated API docs
- PostgreSQL song catalog with pgvector embedding storage and cosine search
- Versioned sound-only embeddings that deliberately exclude titles and artists
- Shadow lyrics-embedding experiment that never stores raw lyric text
- HNSW cosine indexes for sound and experimental lyrics search
- OpenAI query embeddings combined with raw audio features and explicit constraints
- Hybrid candidate recall that unions semantic HNSW neighbors with explicit mood/context/genre matches
- Meaning expansion for short mood prompts and artist-diverse 20-track playlists
- Structured `gpt-4o-mini` intent parsing with query-specific signal priorities and a free rule fallback
- Natural-language intent extraction, including tempo phrases and exclusions such as “not pop”
- Normalized fit scores using only signals applicable to each request
- Curated demo catalog and backend unit tests
- Free deterministic fallback for tests; live semantic searches use one small OpenAI embedding request

## Run locally

Requirements: Node 18+, Python 3.10+, PostgreSQL 17, and pgvector.

```bash
brew install postgresql@17 pgvector
brew services start postgresql@17
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"
createdb resonance
psql -d resonance -c 'CREATE EXTENSION IF NOT EXISTS vector;'

python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
alembic upgrade head
python -m backend.scripts.ingest_tracks --limit 100
# Copy the environment template and add your own key locally
cp .env.example .env
# Free cost preview; does not contact OpenAI
python -m backend.scripts.embed_sound_songs --limit 10
# After configuring OPENAI_API_KEY in .env
python -m backend.scripts.embed_sound_songs --limit 10 --execute
python -m uvicorn backend.app.main:app --reload --port 8000
```

In another terminal:

```bash
npm install
npm run dev
```

Open http://localhost:5173. API documentation is at http://localhost:8000/docs.

## Test

```bash
python -m pytest backend/tests
npm run build
```

## Architecture

```text
Kaggle CSV ──► cleaning/enrichment ──► PostgreSQL + sound vectors
                                                    │
Natural-language prompt                    │
        │
        ▼
LLM intent parser ──► sound, lyrics, constraints, dynamic priorities
        │
        ▼
Sound query ──► sound HNSW search ─────────┐
Lyrics query ──► lyrics HNSW experiment ───┤
        │
        ▼
Normalized hybrid ranker ◄─────────────────┘
        │
        ├──► fit + evidence
        │
        ▼
FastAPI response ──► React playlist + explanations
```

The database uses 1,536-dimensional pgvector columns. The active `sound_embedding`
contains genre and audio evidence but excludes title, artist, album, description,
and lyrics. Title matching is activated only by explicit requests such as
“songs with home in the title.” The experimental `lyrics_embedding` stores a
derived vector and provider audit metadata without retaining raw lyric text.

Each submitted website search currently makes one `gpt-4o-mini` intent call and
one `text-embedding-3-small` request. PostgreSQL similarity search is local and
does not consume OpenAI tokens. Unit tests disable paid providers.

## API example

```bash
curl -X POST http://localhost:8000/api/recommendations \
  -H 'Content-Type: application/json' \
  -d '{"query":"dreamy electronic music for late-night focus, not too slow","limit":6}'
```

## Current safety and evaluation status

- The full local catalog contains 89,740 sound vectors.
- Lyrics retrieval remains experiment-only and is disabled by default.
- LRCLIB-derived vectors must not be treated as cleared for publication without confirming appropriate rights.
- Embedding scripts require `--limit` and are dry-run-only unless `--execute` is supplied.
- Interrupted jobs are resumable because each processed song is committed independently.

Next milestones are a representative lyrics-coverage evaluation, a judged query
set with ranking metrics, exact-query caching, and a licensed production lyrics source.
### Try semantic song search

Preview the query and estimated API cost without calling OpenAI:

```bash
python -m backend.scripts.search_songs "calm acoustic music for studying"
```

Authorize one query-embedding call and search the embedded songs:

```bash
python -m backend.scripts.search_songs "calm acoustic music for studying" --execute
```

When `AI_PROVIDER=openai`, `INTENT_PROVIDER=openai`, and `OPENAI_API_KEY` are
present in `.env`, each `POST /api/recommendations` request uses one intent call
and one embedding request. The test suite disables those providers.

### Full-catalog workflow

Import all unique CSV tracks locally (free):

```bash
python -m backend.scripts.ingest_tracks --batch-size 1000
```

Preview every missing or stale embedding (free):

```bash
python -m backend.scripts.embed_songs --limit 100000 --batch-size 100
```

Only after reviewing the preview, authorize the displayed work:

```bash
python -m backend.scripts.embed_songs --limit 100000 --batch-size 100 --execute
```

Each batch commits independently. If the command is interrupted, rerunning it
selects only missing, mismatched-model, or outdated-text-version vectors.
