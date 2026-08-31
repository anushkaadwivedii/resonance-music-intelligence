"""Evaluate narrative verification against candidates retrieved like the website."""

import argparse

from backend.app.embeddings import OpenAIEmbeddingProvider
from backend.app.lyrics_analysis import OpenAILyricsVerifier
from backend.app.repository import PostgresSongRepository
from backend.app.retrieval import calibrated_lyrics_scores


PROMPTS = [
    "songs about still longing for someone who is with somebody else",
    "songs about heartbreak and being unable to love again",
    "songs about intense forbidden attraction and desire",
]
MAXIMUM_COST_USD = 0.007


def retrieved_songs(prompt: str, limit: int, embedder: OpenAIEmbeddingProvider):
    vector = embedder.embed_many([
        "Lyrical themes, story, and emotional meaning: " + prompt
    ]).vectors[0]
    rows = PostgresSongRepository().search_by_lyrics(vector, vector, 100)
    scores = calibrated_lyrics_scores({song.id: score for song, score, _ in rows})
    eligible = [
        (song, scores[song.id])
        for song, _, _ in rows
        if song.id in scores and song.lyrics_meaning is not None
    ]
    eligible.sort(key=lambda item: item[1], reverse=True)
    return [song for song, _ in eligible[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the lyrical narrative verifier")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 12:
        parser.error("--limit must be between 1 and 12 to mirror the website verifier")

    print({
        "mode": "retrieval_plus_lyrics_verifier_pilot",
        "prompts": len(PROMPTS),
        "maximum_candidates_per_prompt": args.limit,
        "raw_lyrics_sent": False,
        "maximum_cost_usd": MAXIMUM_COST_USD,
        "execute": args.execute,
    }, flush=True)
    if not args.execute:
        print("Dry run only. Add --execute to authorize the three verifier calls.", flush=True)
        return

    verifier = OpenAILyricsVerifier()
    embedder = OpenAIEmbeddingProvider()
    for prompt in PROMPTS:
        songs = retrieved_songs(prompt, args.limit, embedder)
        try:
            verdicts = verifier.verify(prompt, songs)
        except RuntimeError as error:
            print(f"\nVERIFIER STOPPED: {error}", flush=True)
            print("No automatic retry was made.", flush=True)
            return
        ranked = sorted(
            (verdict for verdict in verdicts.values() if verdict.verdict != "no_match"),
            key=lambda verdict: (verdict.verdict == "match", verdict.confidence),
            reverse=True,
        )
        print(f"\nPROMPT: {prompt}", flush=True)
        print(f"  Retrieved {len(songs)} analyzed candidates", flush=True)
        if not ranked:
            print("  No verified matches in the retrieved analyzed candidates.", flush=True)
            continue
        songs_by_id = {song.id: song for song in songs}
        for verdict in ranked:
            song = songs_by_id[verdict.song_id]
            print(
                f"  {verdict.verdict.upper():7} {verdict.confidence:.2f} "
                f"{song.title} — {song.artist}: {verdict.reason}",
                flush=True,
            )


if __name__ == "__main__":
    main()
