export type Song = {
  id: string;
  title: string;
  artist: string;
  album: string | null;
  genre: string;
  genres: string[];
  moods: string[];
  contexts: string[];
  bpm: number;
  perceived_bpm: number | null;
  year: number | null;
  description: string;
  accent: string;
  popularity: number | null;
  energy: number | null;
  danceability: number | null;
  valence: number | null;
  acousticness: number | null;
  instrumentalness: number | null;
  lyrics_evidence: "analyzed" | "unavailable" | "not_analyzed";
};

export type Intent = {
  search_description: string | null;
  lyrics_search_description: string | null;
  desired_lyrical_themes: string[];
  avoid_lyrical_themes: string[];
  lyrics_required: boolean;
  avoid_sound: string[];
  signal_weights: {
    semantic: number;
    mood: number;
    audio: number;
    context: number;
    tempo: number;
    genre: number;
    artist: number;
    lyrics: number;
    popularity_tiebreak: number;
  };
  moods: string[];
  genres: string[];
  contexts: string[];
  excluded_genres: string[];
  title_contains: string | null;
  artist_reference: string | null;
  bpm_min: number | null;
  bpm_max: number | null;
  bpm_is_explicit: boolean;
  valence_target: number | null;
  energy_target: number | null;
  danceability_target: number | null;
  acousticness_target: number | null;
  instrumentalness_target: number | null;
};

export type Recommendation = {
  song: Song;
  score: number;
  explanation: string;
  matched_on: string[];
  breakdown: Record<"semantic" | "mood" | "context" | "tempo" | "genre" | "audio" | "popularity" | "lyrics", number>;
};

export type RecommendationResponse = {
  query: string;
  summary: string;
  intent: Intent;
  recommendations: Recommendation[];
};
