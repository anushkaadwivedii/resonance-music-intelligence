import { FormEvent, useMemo, useRef, useState } from "react";
import {
  ArrowUpRight,
  AudioLines,
  Check,
  ExternalLink,
  Library,
  LoaderCircle,
  MessageCircleMore,
  Music2,
  ScanHeart,
  Search,
  Sparkles,
  X,
} from "lucide-react";
import type { Intent, MatchFocus, Recommendation, RecommendationResponse } from "./types";

type Palette = "wine" | "forest" | "blue" | "clay";

const prompts = [
  "songs about still longing for someone who is with somebody else",
  "dreamy songs for a rainy late night, but not too slow",
  "warm indie folk for a long road trip",
  "focused electronic music for coding after dark",
];

const initialResponse: RecommendationResponse = {
  query: "",
  summary: "Tell me where you are, how you feel, or what you want the next hour to sound like.",
  intent: {
    search_description: null,
    lyrics_search_description: null,
    desired_lyrical_themes: [],
    avoid_lyrical_themes: [],
    lyrics_required: false,
    avoid_sound: [],
    signal_weights: { semantic: 1, mood: 1, audio: 1, context: 1, tempo: 1, genre: 1, artist: 1, lyrics: 1, popularity_tiebreak: 0.04 },
    moods: [], genres: [], contexts: [], excluded_genres: [], title_contains: null,
    artist_reference: null, bpm_min: null, bpm_max: null, bpm_is_explicit: false,
    valence_target: null, energy_target: null, danceability_target: null,
    acousticness_target: null, instrumentalness_target: null,
  },
  recommendations: [],
};

async function getRecommendations(query: string, focus: MatchFocus): Promise<RecommendationResponse> {
  const response = await fetch("/api/recommendations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, limit: 20, focus }),
  });
  if (!response.ok) throw new Error("The recommendation service could not complete that request.");
  return response.json() as Promise<RecommendationResponse>;
}

function spotifySearchUrl(item: Recommendation): string {
  return `https://open.spotify.com/search/${encodeURIComponent(`${item.song.title} ${item.song.artist}`)}`;
}

function IntentSignals({ intent }: { intent: Intent }) {
  const signals = [...intent.moods, ...intent.contexts, ...intent.genres].slice(0, 5);
  if (!signals.length) return null;
  return <div className="intent-signals"><span><Sparkles size={12} /> I heard</span>{signals.map((signal) => <b key={signal}>{signal}</b>)}</div>;
}

function MatchBreakdown({ item }: { item: Recommendation }) {
  const labels: Record<string, string> = {
    semantic: "Sound meaning", mood: "Mood", context: "Setting", tempo: "Tempo",
    genre: "Genre", audio: "Audio profile", popularity: "Catalog confidence", lyrics: "Lyrical meaning",
  };
  return (
    <div className="breakdown">
      {Object.entries(item.breakdown).filter(([key, value]) => key !== "lyrics" || value > 0).map(([key, value]) => (
        <div className="breakdown-line" key={key}>
          <span>{labels[key]}</span><i><em style={{ width: `${value}%` }} /></i><b>{value}%</b>
        </div>
      ))}
    </div>
  );
}

function LinerNote({ item, onClose }: { item: Recommendation; onClose: () => void }) {
  const lyricsUsed = item.song.lyrics_evidence === "analyzed" && item.breakdown.lyrics > 0;
  const evidence = item.lyrics_verified
    ? item.lyrics_verification_reason ?? "A narrative check confirmed that the stored lyrical meaning fits this request."
    : lyricsUsed
      ? "Passage-level lyrical similarity contributed to the ranking, but it is not a verified interpretation."
      : "This result was ranked from sound and catalog metadata; lyrics did not affect its score.";
  return (
    <aside className="liner-note" aria-live="polite">
      <button type="button" className="close-note" onClick={onClose} aria-label="Close liner notes"><X size={15} /></button>
      <span>liner notes · why it fits</span>
      <p>{item.explanation}</p>
      <div className="evidence"><Check size={13} /><small>{evidence}</small></div>
      <MatchBreakdown item={item} />
    </aside>
  );
}

function RecordShelf({ items, selectedId, onSelect, onRemove }: {
  items: Recommendation[]; selectedId: string | null;
  onSelect: (item: Recommendation) => void; onRemove: (id: string) => void;
}) {
  return (
    <section className="record-shelf" aria-label="Recommendation queue">
      <span className="shelf-label">Pull a record from the shelf · {items.length} pressings</span>
      <div className="shelf-scroll">
        {items.map((item, index) => (
          <div className="shelf-item" key={item.song.id}>
            <button
              type="button"
              className="mini-record"
              aria-pressed={selectedId === item.song.id}
              onClick={() => onSelect(item)}
              style={{ "--record-accent": item.song.accent, "--lean": `${((index % 5) - 2) * 1.4}deg` } as React.CSSProperties}
            >
              <span className="mini-vinyl" />
              <span className="mini-sleeve"><strong>{item.song.title}</strong><small>{item.song.artist}</small><em>{String(index + 1).padStart(2, "0")} · {item.score}%</em></span>
            </button>
            <button type="button" className="shelf-remove" onClick={() => onRemove(item.song.id)} aria-label={`Remove ${item.song.title}`}><X size={12} /></button>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [focus, setFocus] = useState<MatchFocus>("balanced");
  const [palette, setPalette] = useState<Palette>("wine");
  const [data, setData] = useState(initialResponse);
  const [playlist, setPlaylist] = useState<Recommendation[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showNote, setShowNote] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const selected = useMemo(
    () => playlist.find((item) => item.song.id === selectedId) ?? playlist[0] ?? null,
    [playlist, selectedId],
  );
  const hasSearched = data.query.length > 0;

  const runQuery = async (value: string) => {
    const clean = value.trim();
    if (clean.length < 3 || loading) return;
    setLoading(true); setError(""); setShowNote(false);
    try {
      const response = await getRecommendations(clean, focus);
      setData(response);
      setPlaylist(response.recommendations);
      setSelectedId(response.recommendations[0]?.song.id ?? null);
      setQuery("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Something went wrong.");
    } finally { setLoading(false); }
  };

  const submit = (event: FormEvent) => { event.preventDefault(); void runQuery(query); };
  const choosePrompt = (prompt: string) => { setQuery(prompt); void runQuery(prompt); };
  const removeTrack = (id: string) => {
    setPlaylist((current) => current.filter((item) => item.song.id !== id));
    if (selectedId === id) { setSelectedId(null); setShowNote(false); }
  };

  return (
    <main className="listening-room" data-palette={palette}>
      <aside className="mast">
        <a className="seal" href="#discover" aria-label="Resonance home">R</a>
        <div className="vertical-brand">Resonance · find the feeling</div>
        <nav className="mast-nav" aria-label="Page navigation">
          <a className="active" href="#discover" aria-label="Discover"><Sparkles /></a>
          <a href="#records" aria-label="Recommendation queue"><Library /></a>
          <a href="#method" aria-label="How matching works"><ScanHeart /></a>
        </nav>
      </aside>

      <div className="room-content" id="discover">
        <header className="room-header">
          <div><p className="eyebrow">Side A · lyrical discovery</p><h1>find the song hiding <em>between the lines.</em></h1></div>
          <div className="palette" aria-label="Choose a dark color palette">
            {(["wine", "forest", "blue", "clay"] as Palette[]).map((color) => <button type="button" key={color} data-color={color} aria-label={`${color} palette`} aria-pressed={palette === color} onClick={() => setPalette(color)} />)}
          </div>
        </header>

        <div className="query-row">
          <form className="query-form" onSubmit={submit}>
            <Search aria-hidden="true" />
            <input ref={inputRef} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="songs about still longing for someone who is with somebody else" aria-label="Describe the music you want" />
            <button type="submit" disabled={query.trim().length < 3 || loading}>{loading ? <LoaderCircle className="spin" /> : <>find it <ArrowUpRight /></>}</button>
          </form>
          <div className="focus" aria-label="Match by">
            {([ ["sound", "sound"], ["balanced", "both"], ["lyrics", "lyrics"] ] as [MatchFocus, string][]).map(([value, label]) => <button type="button" key={value} aria-pressed={focus === value} onClick={() => setFocus(value)}>{label}</button>)}
          </div>
        </div>
        {error && <p className="error">{error} Make sure the backend is running on port 8000.</p>}

        {!hasSearched && (
          <section className="empty-stage">
            <div className="empty-copy"><span>Listening desk · ready</span><h2>Describe the story,<br />not just the genre.</h2><p>Resonance can match the sound, the lyrical narrative, or a balance of both.</p></div>
            <div className="display-sleeve"><span>waiting<br />for a<br />feeling</span><i /></div>
            <div className="display-turntable"><span /></div>
            <div className="prompt-tapes">{prompts.map((prompt, index) => <button type="button" key={prompt} onClick={() => choosePrompt(prompt)} style={{ "--tape-lean": `${index % 2 ? 1.5 : -1.5}deg` } as React.CSSProperties}>{prompt}</button>)}</div>
          </section>
        )}

        {hasSearched && selected && (
          <section className="result-stage">
            <div className="selected-copy" aria-live="polite">
              <span>Selected pressing · {String(playlist.indexOf(selected) + 1).padStart(2, "0")}</span>
              <h2>{selected.song.title}</h2>
              <p className="selected-artist">{selected.song.artist}</p>
              <p className="selected-story">{selected.explanation}</p>
              <div className="selected-actions">
                <a href={spotifySearchUrl(selected)} target="_blank" rel="noreferrer">Open in Spotify <ExternalLink size={13} /></a>
                <button type="button" onClick={() => setShowNote((current) => !current)}><MessageCircleMore size={13} /> Why this fits</button>
              </div>
            </div>

            <div className="record-scene" aria-hidden="true" style={{ "--selected-accent": selected.song.accent } as React.CSSProperties}>
              <div className="selected-vinyl" />
              <div className="selected-sleeve"><span>{selected.song.title}</span><small>Resonance selection · {selected.score}% fit</small></div>
            </div>
            <div className="turntable"><span className="tonearm" /></div>
            <div className="fit-stamp"><strong>{selected.score}%</strong><span>relative<br />story fit</span></div>
            {showNote && <LinerNote item={selected} onClose={() => setShowNote(false)} />}
            <div className="resonance-note"><span>A note from Resonance</span><p>{data.summary}</p><IntentSignals intent={data.intent} /></div>
            <RecordShelf items={playlist} selectedId={selected.song.id} onSelect={(item) => { setSelectedId(item.song.id); setShowNote(false); }} onRemove={removeTrack} />
          </section>
        )}

        {hasSearched && !selected && (
          <section className="no-results"><Music2 /><h2>No confident matches yet</h2><p>Try a broader lyrical theme, or describe the sound and mood you want.</p><button type="button" onClick={() => { setQuery(data.query); inputRef.current?.focus(); }}>Refine your request</button></section>
        )}

        <section className="method" id="method">
          <span>Behind the mix</span><p><b>01</b> Parse the intent</p><p><b>02</b> Retrieve by sound and lyrical meaning</p><p><b>03</b> Explain why each pressing belongs</p>
        </section>
      </div>
    </main>
  );
}
