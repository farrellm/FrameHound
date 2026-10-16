import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, isCollection, type CollectionResult, type Movie } from "../api";
import { useAnnouncer } from "../components/Announcer";
import { ErrorDetails } from "../components/ErrorDetails";
import { MiniStrip } from "../components/MiniStrip";
import { Sheet } from "../components/Sheet";
import { fmtDuration } from "../format";
import { linkProps, navigate } from "../router";
import { activeLabel } from "../stages";

/** The same rule the server uses to tell the three source types apart. */
function parses(text: string): boolean {
  const t = text.trim();
  if (/^magnet:\?/i.test(t)) return true;
  try {
    const u = new URL(t);
    return (u.protocol === "https:" || u.protocol === "http:") && !!u.hostname.includes(".");
  } catch {
    return false;
  }
}

export function Library() {
  const [movies, setMovies] = useState<Movie[] | null>(null);
  const [loadError, setLoadError] = useState<ApiError | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await api.movies();
      setMovies(r.movies);
      setLoadError(null);
    } catch (e) {
      setLoadError(e as ApiError);
    }
  }, []);

  useEffect(() => {
    load();
    const onVis = () => document.visibilityState === "visible" && load();
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, [load]);

  // Rows in progress update by polling; it's one cheap request for the whole list.
  const busy = movies?.some((m) => m.status === "ingesting" || m.status === "indexing");
  useEffect(() => {
    if (!busy) return;
    const t = window.setInterval(load, 3000);
    return () => window.clearInterval(t);
  }, [busy, load]);

  return (
    <main className="page">
      <h1 className="screen-title">FrameHound</h1>
      <AddMovie onAdded={load} />
      {loadError && (
        <p className="message danger" role="alert">
          {loadError.message}
        </p>
      )}
      {movies && movies.length === 0 && (
        <section className="empty measure">
          <h2 className="screen-title">Nothing on the table yet.</h2>
          <p>
            Paste a link to a movie. Public-domain films from the Internet Archive are a good place to start.
          </p>
        </section>
      )}
      {movies && movies.length > 0 && (
        <ul className="library" aria-label="Movies">
          {movies.map((m) => (
            <li key={m.id}>
              <MovieRow movie={m} />
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}

function MovieRow({ movie: m }: { movie: Movie }) {
  let state: React.ReactNode;
  if (m.status === "ready") state = <>Ready to search</>;
  else if (m.status === "failed")
    state = (
      <>
        <span className="danger">{m.job?.kind === "ingest" ? "Adding failed" : "Indexing failed"}</span>
        {m.error ? ` ${m.error}` : ""}
      </>
    );
  else if (m.status === "cancelled") state = <>Indexing stopped</>;
  else
    state = (
      <span className="num">
        {activeLabel(m.job?.stage ?? null, m.job?.state === "running" ? m.job.pct : null)}
      </span>
    );
  return (
    <a className="movie-row" {...linkProps(`/m/${m.id}`)}>
      <MiniStrip thumbs={m.thumbs} />
      <span className="meta">
        <span className="title">{m.title || "Untitled movie"}</span>
        <br />
        <span className="sub">
          {m.duration_s ? <span className="num">{fmtDuration(m.duration_s)}</span> : null}
          <span>{state}</span>
        </span>
      </span>
    </a>
  );
}

function AddMovie({ onAdded }: { onAdded: () => void }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [focused, setFocused] = useState(false);
  const [collection, setCollection] = useState<CollectionResult | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const { announce } = useAnnouncer();
  const canPaste = typeof navigator.clipboard?.readText === "function";

  async function submit(source: string) {
    setBusy(true);
    setError(null);
    try {
      const r = await api.addMovie(source);
      if (isCollection(r)) {
        setCollection(r);
      } else {
        setText("");
        setCollection(null);
        onAdded();
        navigate(`/m/${r.movie_id}`);
      }
    } catch (e) {
      const err = e as ApiError;
      setError(err);
      announce(err.message, true);
    } finally {
      setBusy(false);
    }
  }

  const valid = parses(text);
  return (
    <>
      <form
        className="add"
        onSubmit={(e) => {
          e.preventDefault();
          if (!valid) {
            setError(
              new ApiError(422, {
                code: "unrecognized",
                message:
                  "That isn't a web address or magnet link. Paste a link that starts with https:// or magnet:.",
              }),
            );
            return;
          }
          submit(text.trim());
        }}
      >
        <label htmlFor="source" className="visually-hidden">
          Movie link
        </label>
        <div className="add-row">
          <input
            id="source"
            ref={input}
            className="field"
            type="text"
            inputMode="url"
            autoComplete="off"
            autoCapitalize="off"
            autoCorrect="off"
            spellCheck={false}
            enterKeyHint="go"
            placeholder="Paste a movie link or archive.org link"
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setError(null);
            }}
            onFocus={() => setFocused(true)}
            onBlur={() => setTimeout(() => setFocused(false), 200)}
            aria-invalid={!!error}
            aria-describedby={error ? "add-error" : undefined}
          />
          <button className="btn btn-primary" type="submit" disabled={!valid || busy}>
            {busy ? "Adding…" : "Add movie"}
          </button>
        </div>
        {focused && canPaste && !text && (
          <button
            type="button"
            className="paste-chip"
            onMouseDown={(e) => e.preventDefault()}
            onClick={async () => {
              try {
                const t = (await navigator.clipboard.readText()).trim();
                setText(t);
                input.current?.focus();
              } catch {
                /* permission refused; the system paste menu still works */
              }
            }}
          >
            Paste
          </button>
        )}
        {error && (
          <div id="add-error" role="alert">
            <p className="message danger">{error.message}</p>
            {error.code === "duplicate" && typeof error.extra.movie_id === "string" && (
              <button type="button" className="btn" onClick={() => navigate(`/m/${error.extra.movie_id}`)}>
                Open it
              </button>
            )}
            <ErrorDetails detail={error.detail} />
          </div>
        )}
      </form>
      {collection && (
        <CollectionSheet
          initial={collection}
          busy={busy}
          error={error}
          onPick={(source) => submit(source)}
          onClose={() => {
            setCollection(null);
            setError(null);
          }}
        />
      )}
    </>
  );
}

function CollectionSheet({
  initial,
  busy,
  error,
  onPick,
  onClose,
}: {
  initial: CollectionResult;
  busy: boolean;
  error: ApiError | null;
  onPick: (source: string) => void;
  onClose: () => void;
}) {
  const [q, setQ] = useState("");
  const [result, setResult] = useState(initial);
  const [picked, setPicked] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const id = initial.collection.identifier;

  useEffect(() => {
    if (!q.trim()) {
      setResult(initial);
      return;
    }
    const t = window.setTimeout(async () => {
      setSearching(true);
      try {
        setResult(await api.collection(id, q));
      } catch {
        /* keep the last list */
      } finally {
        setSearching(false);
      }
    }, 300);
    return () => window.clearTimeout(t);
  }, [q, id, initial]);

  const shown = result.choices.length;
  const total = result.collection.total;
  return (
    <Sheet label="Pick a movie from this collection" onClose={onClose}>
      <h2>
        This is a collection.
        <br />
        Pick a movie to add.
      </h2>
      <p className="fog">{initial.collection.title}</p>
      <label htmlFor="coll-q" className="visually-hidden">
        Search this collection
      </label>
      <input
        id="coll-q"
        data-autofocus
        className="field"
        type="search"
        placeholder="Search this collection"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        enterKeyHint="search"
      />
      <p className="note num" aria-live="polite">
        {searching
          ? "Searching…"
          : shown < total
            ? `Showing ${shown.toLocaleString()} of ${total.toLocaleString()}`
            : `${total.toLocaleString()} movies`}
      </p>
      <ul className="choices" role="radiogroup" aria-label="Movies in this collection">
        {result.choices.map((c) => (
          <li key={c.identifier}>
            <label className="radio">
              <input
                type="radio"
                name="choice"
                checked={picked === c.source}
                onChange={() => setPicked(c.source)}
              />
              <span>{c.title}</span>
            </label>
          </li>
        ))}
      </ul>
      {!shown && !searching && <p className="fog">No movies in this collection match that.</p>}
      {error && (
        <p className="message danger" role="alert">
          {error.message}
        </p>
      )}
      <div className="sheet-actions">
        <button
          type="button"
          className="btn btn-primary btn-block"
          disabled={!picked || busy}
          onClick={() => picked && onPick(picked)}
        >
          {busy ? "Adding…" : "Add movie"}
        </button>
      </div>
    </Sheet>
  );
}
