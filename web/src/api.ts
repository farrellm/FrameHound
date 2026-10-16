// Thin client for the FrameHound api (design.md §8). Everything is under /api.

export type StageState = "pending" | "running" | "done" | "failed" | "skipped";

export interface Stage {
  stage: string;
  state: StageState;
  pct: number;
  eta_s: number | null;
  job?: "ingest" | "index";
}

export interface Job {
  id: string;
  movie_id: string;
  kind: "ingest" | "index" | "clip";
  state: "queued" | "running" | "done" | "failed" | "cancelled";
  stage: string | null;
  pct: number | null;
  eta_s: number | null;
  error: string | null;
  error_code: string | null;
  gpu: string | null;
  compute_units_used: number | null;
  seq: number;
  detail: string | null;
  stages?: Stage[];
}

export type MovieStatus = "ingesting" | "indexing" | "ready" | "failed" | "cancelled" | "merged";

export interface Movie {
  id: string;
  title: string | null;
  status: MovieStatus;
  error: string | null;
  merged_into: string | null;
  duration_s: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  has_source: boolean;
  has_proxy: boolean;
  capabilities: { dialogue: boolean; visual: boolean; faces: boolean };
  models_ok: boolean | null;
  job: Job | null;
  thumbs: string[];
}

export interface IndexProgress {
  movie_id: string;
  status: MovieStatus;
  error: string | null;
  job: Job | null;
  stages: Stage[];
  eta_s: number | null;
  gpu: string | null;
  compute_units_used: number | null;
}

export interface Word {
  word: string;
  start: number;
  end: number;
}

export interface Candidate {
  start: number;
  end: number;
  score: number;
  peak_t: number;
  reason: "visual" | "face" | "dialogue";
  thumbs: string[];
  has_face: boolean;
  matched_words?: Word[];
  text?: string;
}

export interface Shot {
  idx: number;
  start: number;
  end: number;
  thumb_url: string;
}

export interface Health {
  colab: { state: "ok" | "auth_broken" | "error" | "not_configured"; balance_cu?: number; error?: string };
  index_backend: string;
}

export interface SearchResult {
  candidates: Candidate[];
  heat: number[];
  heat_threshold: number;
  latency_ms: number;
}

export interface Choice {
  identifier: string;
  title: string;
  source: string;
}

export interface CollectionResult {
  collection: { identifier: string; title: string; total: number };
  choices: Choice[];
}

export interface Clip {
  id: string;
  movie_id: string;
  start: number;
  end: number;
  ready: boolean;
  file_url: string;
  job: Job | null;
}

export class ApiError extends Error {
  code: string;
  detail?: string;
  status: number;
  extra: Record<string, unknown>;
  constructor(status: number, body: Record<string, unknown>) {
    super(String(body.message ?? "Something went wrong on the server."));
    this.status = status;
    this.code = String(body.code ?? "error");
    this.detail = body.detail as string | undefined;
    this.extra = body;
  }
}

const BASE = "/api";

// The api returns media paths relative to itself; the SPA reaches it under /api.
const media = (u: string) => (u.startsWith("/") ? BASE + u : u);
const withThumbs = <T extends { thumbs: string[] }>(x: T): T => ({ ...x, thumbs: x.thumbs.map(media) });

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, {
      code: "offline",
      message: "FrameHound's server isn't answering. Check that it's running.",
    });
  }
  if (res.status === 204) return undefined as T;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, data);
  return data as T;
}

export const api = {
  movies: () =>
    req<{ movies: Movie[] }>("GET", "/movies").then((r) => ({ movies: r.movies.map(withThumbs) })),
  movie: (id: string) => req<Movie>("GET", `/movies/${id}`).then(withThumbs),
  shots: (id: string) =>
    req<Shot[]>("GET", `/movies/${id}/shots`).then((r) => r.map((s) => ({ ...s, thumb_url: media(s.thumb_url) }))),
  health: () => req<Health>("GET", "/health"),
  addMovie: (source: string) =>
    req<{ movie_id: string; job_id: string } | CollectionResult>("POST", "/movies", { source }),
  collection: (id: string, q: string) =>
    req<CollectionResult>("GET", `/collections/${encodeURIComponent(id)}?q=${encodeURIComponent(q)}`),
  retry: (id: string) => req<{ job_id: string }>("POST", `/movies/${id}/retry`),
  remove: (id: string) => req<void>("DELETE", `/movies/${id}`),
  progress: (id: string) => req<IndexProgress>("GET", `/movies/${id}/index-progress`),
  search: (id: string, body: { type: string; q: string; heat_bins?: number; top_k?: number }) =>
    req<SearchResult>("POST", `/movies/${id}/search`, body).then((r) => ({
      ...r,
      candidates: r.candidates.map(withThumbs),
    })),
  cutClip: (id: string, body: { start: number; end: number; mode: "accurate" | "fast" }) =>
    req<{ clip_id: string; job_id: string }>("POST", `/movies/${id}/clips`, { ...body, format: "mp4" }),
  clip: (id: string) => req<Clip>("GET", `/clips/${id}`),
  cancel: (jobId: string) => req<void>("POST", `/jobs/${jobId}/cancel`),
};

export const proxyUrl = (id: string) => `${BASE}/media/${id}/proxy.mp4`;
export const clipFileUrl = (id: string) => `${BASE}/clips/${id}/file`;

/** Follow a job over SSE. EventSource reconnects with Last-Event-ID by itself. */
export function followJob(jobId: string, onEvent: (job: Job) => void): () => void {
  const es = new EventSource(`${BASE}/jobs/${jobId}/events`);
  es.addEventListener("progress", (e) => {
    const job = JSON.parse((e as MessageEvent).data) as Job;
    onEvent(job);
    if (["done", "failed", "cancelled"].includes(job.state)) es.close();
  });
  es.addEventListener("gone", () => es.close());
  return () => es.close();
}

export function isCollection(r: unknown): r is CollectionResult {
  return typeof r === "object" && r !== null && "choices" in r;
}
