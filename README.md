# Resonance — Explainable Music Discovery

Resonance is a full-stack music-discovery prototype that turns natural-language requests into explainable playlists. A listener can describe a mood, activity, sound, tempo, or lyrical idea—for example, “warm indie songs for a rainy train ride, but not too slow”—and choose whether the result should match by **Sound**, **Both**, or **Lyrics**.

The app keeps track identity separate from evidence about sound and lyrical meaning. Every result includes a fit score and a breakdown of the signals used in the ranking.

## Current product

- React and TypeScript interface with responsive playlist results
- Sound, balanced, and experimental lyrics matching modes
- FastAPI backend with typed request and response models
- PostgreSQL 17 catalog with pgvector and HNSW cosine indexes
- Structured `gpt-4o-mini` intent parsing with a free rule-based fallback
- `text-embedding-3-small` query and catalog embeddings
- Hybrid retrieval across vector similarity, metadata, audio features, and explicit constraints
- Query-specific scoring that normalizes only the signals relevant to the request
- Artist diversity and conservative lyrics-result limits
- “Why this song?” explanations with per-signal score breakdowns
- Play buttons that open a title-and-artist search on Spotify
- Cost-guarded, resumable ingestion and embedding scripts
- An optional second-stage lyrical verifier, implemented but disabled by default

## Local data snapshot

The local development database currently contains:

| Data | Coverage |
| --- | ---: |
| Catalog songs | 89,740 |
| Sound embeddings | 89,740 songs (100%) |
| Whole-song lyrics embeddings | 1,611 songs |
| Passage-level lyrics embeddings | 1,611 songs / 2,983 passages |
| Structured lyrical-meaning records | 120 songs |

Sound search covers the complete catalog. Lyrics search is a beta and covers only songs with a sufficiently confident lyrics lookup.

## How a search works

```text
Natural-language request
          │
          ▼
Structured intent parser
  mood · sound · context · tempo · exclusions · lyrical subject
          │
          ├──────────── Sound query embedding ────────┐
          │                                            │
          └──────────── Lyrics query embedding ───┐    │
                                                   ▼    ▼
                                      pgvector HNSW candidate recall
                                                   │
                              metadata and audio-feature candidate recall
                                                   │
                                                   ▼
                                normalized hybrid scoring + diversity
                                                   │
                                                   ▼
                                  FastAPI response with evidence
                                                   │
                                                   ▼
                                      React explainable playlist
```

The sound embedding text includes genre, audio-derived moods, listening contexts, tempo, and numeric audio features. It does not include title, artist, album, description, or lyrics. Title and artist matching activate only when the request names them.

Lyrics mode combines passage-level and whole-song semantic similarity. Raw lyrics are fetched temporarily for local processing and are not stored. An optional narrative verifier can filter related but incomplete themes. It is disabled by default because 120 structured meaning records do not provide enough coverage for useful playlists.

## Technology

- Frontend: React, TypeScript, Vite
- Backend: Python, FastAPI, Pydantic, SQLAlchemy
- Database: PostgreSQL 17, pgvector
- Migrations: Alembic
- AI: OpenAI Responses API and Embeddings API
- Tests: pytest and TypeScript production builds

## Run locally

Requirements:

- Node.js 18+
- Python 3.10+
- PostgreSQL 17
- pgvector
- An OpenAI API key for live searches and embedding jobs

Install and start PostgreSQL on macOS:

```bash
brew install postgresql@17 pgvector
brew services start postgresql@17
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"
createdb resonance
psql -d resonance -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

Set up the backend:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env
# Add your own OPENAI_API_KEY to .env. Never commit it.
alembic upgrade head
python -m uvicorn backend.app.main:app --reload --port 8000
```

Set up the frontend in another terminal:

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). FastAPI documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Configuration

The template keeps stored-vector features disabled until their data has been
populated:

```env
USE_SOUND_EMBEDDINGS=false
USE_LYRICS_EMBEDDINGS=false
USE_LYRICS_CHUNKS=false
USE_LYRICS_VERIFIER=false
AI_PROVIDER=openai
INTENT_PROVIDER=openai
```

Enable each feature after running its corresponding embedding job.
`USE_LYRICS_VERIFIER=false` is the recommended prototype setting. Lyrics
results are then semantic candidates rather than guaranteed factual matches.
Enabling the verifier adds another LLM call to each explicit lyrics search and
may produce sparse playlists until structured meaning coverage is much larger.

## API example

```bash
curl -X POST http://localhost:8000/api/recommendations \
  -H 'Content-Type: application/json' \
  -d '{"query":"dreamy electronic music for late-night focus, not too slow","limit":6,"focus":"balanced"}'
```

Valid focus values are `auto`, `sound`, `balanced`, and `lyrics`.

## Tests

```bash
source .venv/bin/activate
python -m pytest -q
npm run build
```

The automated tests use local fakes and do not make paid OpenAI calls.

## Catalog and embedding workflow

Import tracks from the configured CSV:

```bash
python -m backend.scripts.ingest_tracks --batch-size 1000
```

The source dataset is not committed to this repository. Place a compatible Spotify tracks CSV at `data/raw/spotify-tracks-dataset.csv` before running the import. The ingestion script expects track identity, genre, tempo, popularity, duration, and Spotify-style audio-feature columns.

Every billable batch script is a free dry run unless `--execute` is supplied. Preview pending sound embeddings:

```bash
python -m backend.scripts.embed_sound_songs --limit 100000 --batch-size 100
```

After reviewing the estimated cost, run the batch:

```bash
python -m backend.scripts.embed_sound_songs --limit 100000 --batch-size 100 --execute
```

The current local database reports that all 89,740 sound embeddings are complete, so rerunning the preview should report nothing to do.

Lyrics jobs use smaller reviewed batches:

```bash
# Whole-song lyrics lookup and embeddings
python -m backend.scripts.embed_lyrics --limit 100

# Passage-level vectors for songs with successful prior lyrics analysis
python -m backend.scripts.embed_lyrics_chunks --limit 100

# Optional structured meaning records, capped at 100 per reviewed batch
python -m backend.scripts.analyze_lyrics_meaning --limit 100
```

Add `--execute` only after reviewing the displayed cost ceiling. Jobs commit each processed song independently, so interrupted work is resumable.

### Why not embed lyrics for every song?

Embedding is only the final step. Each catalog record first needs a correctly matched lyrics source. The LRCLIB workflow has incomplete coverage, can return no match, and is not treated as licensed production data. Processing all 89,740 songs would add provider traffic and cost without resolving data quality or publication rights.

This version therefore has complete sound coverage and a clearly labeled, limited lyrics beta.

## API usage and cost

With the default live configuration, every website search uses:

1. One small `gpt-4o-mini` intent-parsing call.
2. One `text-embedding-3-small` request. Lyrics searches batch the sound and lyrics query texts into that request.
3. Local PostgreSQL similarity search, which consumes no OpenAI tokens.

The optional lyrics verifier adds one more `gpt-4o-mini` call and is disabled by default. Offline catalog embeddings and structured meaning analysis cost money only when their scripts are run with `--execute`.

Before a public deployment, the backend should add response caching, per-IP rate limits, a daily usage budget, monitoring, and provider timeouts/fallbacks. The OpenAI key must remain server-side and must never be included in the frontend bundle.

## Current limitations

- Play opens Spotify search; Resonance does not stream audio.
- Save Playlist is currently a client-side interaction and is not persisted.
- There are no user accounts or playlist-history records.
- Refine is not yet a stateful, multi-turn conversation.
- Lyrics coverage is experimental and incomplete.
- LRCLIB-derived data has not been cleared for production publication.
- Record-card artwork is a local placeholder; the app does not fetch album covers.
- The application is not deployed.
- There is no production rate limiting, caching, or spending cap yet.

## Next milestones

1. Add caching, rate limiting, and a daily API budget.
2. Build a representative human-judged evaluation set and track ranking metrics.
3. Replace Spotify search links with exact track links where reliable IDs are available.
4. Choose a licensed metadata/artwork and lyrics strategy.
5. Deploy the frontend, backend, and PostgreSQL database with secrets stored server-side.
6. Add accounts, persistent playlists, and history only if the product scope requires them.
