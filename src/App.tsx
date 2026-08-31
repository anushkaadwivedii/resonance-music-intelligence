import { FormEvent, useMemo, useRef, useState } from "react";
import {
  ArrowRight,
  AudioLines,
  Check,
  ChevronDown,
  ChevronUp,
  Disc3,
  GripVertical,
  Headphones,
  LoaderCircle,
  MessageCircleMore,
  Music2,
  Play,
  Plus,
  Send,
  Sparkles,
  X,
} from "lucide-react";
import type { Intent, MatchFocus, Recommendation, RecommendationResponse } from "./types";

const prompts = [
  "Dreamy songs for a rainy late night, but not too slow",
  "Warm indie folk for a long road trip",
  "Focused electronic music for coding after dark",
  "Joyful songs that feel like a fresh start",
];

const initialResponse: RecommendationResponse = {
  query: "",
  summary: "Tell me where you are, how you feel, or what you want the next hour to sound like.",
  intent: {
    search_description: null, lyrics_search_description: null, desired_lyrical_themes: [], avoid_lyrical_themes: [], lyrics_required: false, avoid_sound: [],
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

function IntentPills({ intent }: { intent: Intent }) {
  const signals = [
    ...intent.moods.map((label) => ({ label, kind: "mood" })),
    ...intent.contexts.map((label) => ({ label, kind: "context" })),
    ...intent.genres.map((label) => ({ label, kind: "genre" })),
    ...(intent.title_contains ? [{ label: `title: ${intent.title_contains}`, kind: "title" }] : []),
    ...(intent.bpm_min ? [{ label: `${intent.bpm_min}+ BPM`, kind: "tempo" }] : []),
    ...(intent.bpm_max ? [{ label: `≤ ${intent.bpm_max} BPM`, kind: "tempo" }] : []),
  ];
  if (!signals.length) return null;
  return (
    <div className="intent-row" aria-label="Signals understood">
      <span className="intent-label"><Sparkles size={13} /> I heard</span>
      {signals.map(({ label, kind }) => <span className={`intent-pill ${kind}`} key={`${kind}-${label}`}>{label}</span>)}
    </div>
  );
}

function ScoreRing({ score }: { score: number }) {
  return <div className="score-ring" title="Relative fit score, not a probability" aria-label={`Relative fit score ${score} out of 100`} style={{ "--score": `${score * 3.6}deg` } as React.CSSProperties}><span>{score}</span></div>;
}

function spotifySearchUrl(item: Recommendation): string {
  return `https://open.spotify.com/search/${encodeURIComponent(`${item.song.title} ${item.song.artist}`)}`;
}

function TrackArt({ item, index }: { item: Recommendation; index: number }) {
  return (
    <div className="track-art" style={{ "--accent": item.song.accent, "--rotation": `${index * 19}deg` } as React.CSSProperties}>
      <div className="vinyl"><span /></div>
      <div className="art-line" />
    </div>
  );
}

function TrackCard({ item, index, expanded, onExpand, onRemove }: {
  item: Recommendation; index: number; expanded: boolean;
  onExpand: () => void; onRemove: () => void;
}) {
  const breakdownLabels: Record<string, string> = { semantic: "Sound meaning", mood: "Mood", context: "Setting", tempo: "Tempo", genre: "Genre", audio: "Audio profile", popularity: "Catalog confidence", lyrics: "Lyrical similarity" };
  const visibleBreakdown = Object.entries(item.breakdown).filter(
    ([key, value]) => key !== "lyrics" || (item.song.lyrics_evidence === "analyzed" && value > 0),
  );
  const lyricsContributed = item.song.lyrics_evidence === "analyzed" && item.breakdown.lyrics > 0;
  const lyricsEvidence = item.lyrics_verified
    ? {
        label: "Lyrical meaning verified",
        detail: item.lyrics_verification_reason ?? "A narrative check confirmed that the stored lyrical meaning fits this request.",
      }
    : ({
    analyzed: lyricsContributed
      ? { label: "Lyrical similarity used", detail: "Passage-level similarity contributed to this ranking; it is not a verified interpretation." }
      : { label: "Lyrics available · not used", detail: "Lyrics did not affect the score for this search." },
    unavailable: { label: "Lyrics unavailable", detail: "This result is based on sound and metadata; missing lyrics are not treated as a bad match." },
    not_analyzed: { label: "Lyrics not yet analyzed", detail: "This result is based on sound and metadata only." },
  }[item.song.lyrics_evidence]);
  return (
    <article className={`track-card ${expanded ? "expanded" : ""}`}>
      <div className="track-main">
        <button className="drag-handle" aria-label={`Reorder ${item.song.title}`}><GripVertical size={17} /></button>
        <TrackArt item={item} index={index} />
        <a
          className="play-button"
          href={spotifySearchUrl(item)}
          target="_blank"
          rel="noreferrer"
          aria-label={`Open ${item.song.title} by ${item.song.artist} on Spotify`}
          title="Open on Spotify"
        >
          <Play size={15} fill="currentColor" />
        </a>
        <div className="track-info">
          <div className="track-heading"><h3>{item.song.title}</h3><span>·</span><p>{item.song.artist}</p></div>
          <div className="track-meta"><span>{item.song.genre}</span><span>{Math.round(item.song.bpm)} BPM</span>{item.song.perceived_bpm && <span>≈ {Math.round(item.song.perceived_bpm)} half-time feel</span>}{item.song.year && <span>{item.song.year}</span>}</div>
          <div className="match-tags">{item.matched_on.slice(0, 3).map((match) => <span key={match}><Check size={11} />{match}</span>)}</div>
        </div>
        <ScoreRing score={item.score} />
        <button className="why-button" onClick={onExpand}>Why this song? {expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}</button>
        <button className="remove-button" onClick={onRemove} aria-label={`Remove ${item.song.title}`}><X size={16} /></button>
      </div>
      {expanded && (
        <div className="track-reason">
          <div className="reason-details">
            <div className="reason-copy"><MessageCircleMore size={17} /><p>{item.explanation}</p></div>
            <div className={`lyrics-evidence ${lyricsContributed ? "matched" : item.song.lyrics_evidence}`}>
              <Music2 size={13} />
              <div><b>{lyricsEvidence.label}</b><span>{lyricsEvidence.detail}</span></div>
            </div>
          </div>
          <div className="score-bars">
            {visibleBreakdown.map(([key, value]) => (
              <div className="bar-group" key={key}><span>{breakdownLabels[key]}</span><div className="bar"><i style={{ width: `${value}%` }} /></div><b>{value}%</b></div>
            ))}
          </div>
        </div>
      )}
    </article>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [focus, setFocus] = useState<MatchFocus>("sound");
  const [data, setData] = useState(initialResponse);
  const [playlist, setPlaylist] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const hasResults = playlist.length > 0;
  const hasSearched = data.query.length > 0;
  const averageFit = useMemo(() => hasResults ? Math.round(playlist.reduce((sum, item) => sum + item.score, 0) / playlist.length) : 0, [playlist, hasResults]);

  const runQuery = async (value: string) => {
    const clean = value.trim();
    if (clean.length < 3 || loading) return;
    setLoading(true); setError(""); setSaved(false);
    try {
      const response = await getRecommendations(clean, focus);
      setData(response); setPlaylist(response.recommendations); setQuery(""); setExpanded(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Something went wrong.");
    } finally { setLoading(false); }
  };

  const submit = (event: FormEvent) => { event.preventDefault(); void runQuery(query); };
  const choosePrompt = (prompt: string) => { setQuery(prompt); void runQuery(prompt); };

  return (
    <main className="app-shell">
      <div className="ambient ambient-one" /><div className="ambient ambient-two" />
      <nav className="topbar">
        <a className="brand" href="#top" aria-label="Resonance home"><span className="brand-mark"><AudioLines size={20} /></span><span>resonance</span></a>
        <div className="nav-center"><a className="active" href="#discover">Discover</a><a href="#playlist">My playlist</a><a href="#how">How it works</a></div>
        <button className="profile-button"><span>AD</span><ChevronDown size={14} /></button>
      </nav>

      <section className={`hero ${hasResults ? "with-results" : ""}`} id="discover">
        <div className="eyebrow"><span /><Sparkles size={14} /> EXPLAINABLE MUSIC DISCOVERY <span /></div>
        <h1>Find the feeling.<br /><em>We'll find the sound.</em></h1>
        <p className="hero-copy">Describe a moment, mood, or memory, then choose whether to match by sound, lyrics, or both.</p>

        <form className="prompt-box" onSubmit={submit}>
          <div className="prompt-icon"><Headphones size={20} /></div>
          <textarea ref={inputRef} value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); }
          }} placeholder="Try “dreamy songs for a rainy late night, but not too slow”" aria-label="Describe the music you want" rows={1} />
          <button type="submit" disabled={query.trim().length < 3 || loading} aria-label="Find music">
            {loading ? <LoaderCircle className="spin" size={19} /> : <Send size={18} />}
          </button>
        </form>
        <div className="focus-control" aria-label="Choose what the recommendations should match">
          <span>Match by</span>
          {([
            ["sound", "Sound", AudioLines],
            ["balanced", "Both", Sparkles],
            ["lyrics", "Lyrics", MessageCircleMore],
          ] as const).map(([value, label, Icon]) => (
            <button
              type="button"
              key={value}
              className={focus === value ? "active" : ""}
              aria-pressed={focus === value}
              onClick={() => setFocus(value)}
            >
              <Icon size={12} /> {label}
            </button>
          ))}
        </div>
        {error && <div className="error-message">{error} Make sure the API is running on port 8000.</div>}
        {!hasSearched && <div className="prompt-suggestions">{prompts.map((prompt) => <button key={prompt} onClick={() => choosePrompt(prompt)}>{prompt}<ArrowRight size={14} /></button>)}</div>}
      </section>

      {hasSearched && (
        <section className="results" id="playlist">
          <div className="conversation-card">
            <div className="ai-avatar"><Sparkles size={18} /></div>
            <div><span className="ai-name">RESONANCE</span><p>{data.summary}</p><IntentPills intent={data.intent} /></div>
          </div>

          {hasResults ? <><div className="playlist-header">
            <div><span className="section-kicker"><Disc3 size={14} /> YOUR PLAYLIST</span><h2>A soundtrack for right now</h2><p>{playlist.length} tracks · about {playlist.length * 4} min · {averageFit}% average fit</p></div>
            <div className="playlist-actions">
              <button className="secondary-button" onClick={() => { setQuery(data.query); inputRef.current?.focus(); }}><Sparkles size={15} /> Refine</button>
              <button className={`save-button ${saved ? "saved" : ""}`} onClick={() => setSaved(true)}>{saved ? <Check size={16} /> : <Plus size={16} />}{saved ? "Saved" : "Save playlist"}</button>
            </div>
          </div>

          <div className="track-list">
            {playlist.map((item, index) => <TrackCard key={item.song.id} item={item} index={index} expanded={expanded === item.song.id} onExpand={() => setExpanded(expanded === item.song.id ? null : item.song.id)} onRemove={() => setPlaylist((current) => current.filter((track) => track.song.id !== item.song.id))} />)}
          </div>
          <button className="add-more" onClick={() => { inputRef.current?.focus(); window.scrollTo({ top: 0, behavior: "smooth" }); }}><Plus size={16} /> Describe what to add</button>
          </> : (
            <div className="no-results">
              <Music2 size={22} />
              <h2>No confident matches yet</h2>
              <p>Try a broader lyrical theme, or describe the sound and mood you want.</p>
              <button onClick={() => { setQuery(data.query); inputRef.current?.focus(); window.scrollTo({ top: 0, behavior: "smooth" }); }}>Refine your request</button>
            </div>
          )}
        </section>
      )}

      <section className="how" id="how">
        <span className="section-kicker">HOW IT LISTENS</span>
        <div className="how-grid">
          <div><b>01</b><Music2 /><h3>Understands intent</h3><p>Reads mood, setting, energy, genre, and exclusions from natural language.</p></div>
          <div><b>02</b><AudioLines /><h3>Blends the signals</h3><p>Balances semantic meaning with tempo and catalog metadata.</p></div>
          <div><b>03</b><Sparkles /><h3>Explains the match</h3><p>Shows exactly why every track earned its place in the playlist.</p></div>
        </div>
      </section>

      <footer><a className="brand" href="#top"><span className="brand-mark"><AudioLines size={18} /></span><span>resonance</span></a><p>Music discovery that speaks your language.</p><span>Built with hybrid retrieval</span></footer>
    </main>
  );
}
