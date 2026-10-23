// Find a movie by name in Bitmagnet's crawl: releases grouped under the film
// they were matched to, like labels on film cans (ui-design.md §4.1).
import { useEffect, useState } from "react";
import { ApiError, api, type FilmGroup, type TorrentRelease, type TorrentSearch } from "../api";
import { ErrorDetails } from "../components/ErrorDetails";
import { Sheet } from "../components/Sheet";
import { fmtBytes } from "../format";
import { navigate } from "../router";

function fmtRuntime(min: number | null): string | null {
  if (!min) return null;
  const h = Math.floor(min / 60);
  return h ? `${h} h ${min % 60} min` : `${min} min`;
}

function seeders(n: number | null): string {
  if (!n) return "No seeders seen. It may not download.";
  return `${n.toLocaleString()} seeder${n === 1 ? "" : "s"}`;
}

/** The title the movie gets in the library: "Night of the Living Dead (1968)". */
const filmTitle = (f: FilmGroup) => (f.year ? `${f.title} (${f.year})` : f.title);

interface Pick {
  magnet: string;
  title?: string;
}

export function FindSheet({
  initialQuery,
  busy,
  error: addError,
  onPick,
  onClose,
}: {
  initialQuery: string;
  busy: boolean;
  error: ApiError | null;
  onPick: (magnet: string, title?: string) => void;
  onClose: () => void;
}) {
  const [q, setQ] = useState(initialQuery);
  const [result, setResult] = useState<TorrentSearch | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [looking, setLooking] = useState(true);
  const [picked, setPicked] = useState<Pick | null>(null);

  useEffect(() => {
    const term = q.trim();
    if (term.length < 2) return;
    let live = true;
    // The first search runs at once; refining waits for a pause in typing.
    const t = window.setTimeout(
      async () => {
        setLooking(true);
        try {
          const r = await api.findTorrents(term);
          if (!live) return;
          setResult(r);
          setError(null);
          setPicked(null);
        } catch (e) {
          if (live) setError(e as ApiError);
        } finally {
          if (live) setLooking(false);
        }
      },
      result ? 300 : 0,
    );
    return () => {
      live = false;
      window.clearTimeout(t);
    };
  }, [q]);

  const empty = result && !result.films.length && !result.unmatched.length;
  const limit = result ? fmtBytes(result.max_bytes) : null;
  const row = (r: TorrentRelease, title?: string) => (
    <ReleaseRow
      key={r.info_hash}
      release={r}
      showFiles={!title}
      limit={limit}
      checked={picked?.magnet === r.magnet}
      onChange={() => setPicked({ magnet: r.magnet, title })}
    />
  );

  return (
    <Sheet label="Pick a release to add" onClose={onClose}>
      <h2>Pick a release to add.</h2>
      <label htmlFor="find-q" className="visually-hidden">
        Movie name
      </label>
      <input
        id="find-q"
        className="field"
        type="search"
        autoComplete="off"
        autoCorrect="off"
        spellCheck={false}
        enterKeyHint="search"
        value={q}
        onChange={(e) => setQ(e.target.value)}
      />
      <p className="note num" aria-live="polite">
        {looking
          ? "Looking…"
          : result && !empty
            ? `${result.total.toLocaleString()} torrent${result.total === 1 ? "" : "s"} Bitmagnet has seen`
            : ""}
      </p>
      {error && (
        <div role="alert">
          <p className="message danger">{error.message}</p>
          <ErrorDetails detail={error.detail} />
        </div>
      )}
      {empty && !looking && !error && (
        <p className="fog measure">
          Bitmagnet hasn't seen a movie torrent called that. It keeps crawling, so try another spelling or check
          back later.
        </p>
      )}
      {result && !error && (
        <div className="finds" role="radiogroup" aria-label="Releases">
          {result.films.map((f) => (
            <section className="film-group" key={filmTitle(f) + f.releases[0]?.info_hash}>
              <h3>{f.title}</h3>
              <p className="film-meta num">
                {f.year && <span>{f.year}</span>}
                {fmtRuntime(f.runtime_min) && <span>{fmtRuntime(f.runtime_min)}</span>}
              </p>
              <ul className="choices">{f.releases.map((r) => row(r, filmTitle(f)))}</ul>
            </section>
          ))}
          {result.unmatched.length > 0 && (
            <section className="film-group">
              <h3>Not matched to a film</h3>
              <ul className="choices">{result.unmatched.map((r) => row(r))}</ul>
            </section>
          )}
        </div>
      )}
      {addError && (
        <div role="alert">
          <p className="message danger">{addError.message}</p>
          {addError.code === "duplicate" && typeof addError.extra.movie_id === "string" && (
            <button type="button" className="btn" onClick={() => navigate(`/m/${addError.extra.movie_id}`)}>
              Open it
            </button>
          )}
          <ErrorDetails detail={addError.detail} />
        </div>
      )}
      <div className="sheet-actions sticky">
        <button
          type="button"
          className="btn btn-primary btn-block"
          disabled={!picked || busy}
          onClick={() => picked && onPick(picked.magnet, picked.title)}
        >
          {busy ? "Adding…" : "Add movie"}
        </button>
      </div>
    </Sheet>
  );
}

function ReleaseRow({
  release: r,
  limit,
  showFiles,
  checked,
  onChange,
}: {
  release: TorrentRelease;
  /** Unmatched torrents: a file count gives away a season pack. */
  showFiles: boolean;
  limit: string | null;
  checked: boolean;
  onChange: () => void;
}) {
  const quality = [r.resolution, r.source, r.codec].filter(Boolean).join(" ");
  return (
    <li>
      <label className={`radio release${r.too_big ? " disabled" : ""}`}>
        <input type="radio" name="release" checked={checked} disabled={r.too_big} onChange={onChange} />
        <span className="release-body">
          <span className="release-facts num">
            {quality && <span className="quality">{quality}</span>}
            {fmtBytes(r.size) && <span>{fmtBytes(r.size)}</span>}
            {showFiles && r.files != null && r.files > 1 && <span>{r.files.toLocaleString()} files</span>}
          </span>
          <span className="hint">
            {r.too_big ? `Larger than this FrameHound's ${limit ?? ""} limit` : seeders(r.seeders)}
          </span>
          <span className="release-name" title={r.name}>
            {r.name}
          </span>
        </span>
      </label>
    </li>
  );
}
