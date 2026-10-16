// Indexing progress (ui-design.md §4.2): show it's moving, and that you can leave.
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, followJob, type IndexProgress, type Movie } from "../api";
import { useAnnouncer } from "../components/Announcer";
import { ErrorDetails } from "../components/ErrorDetails";
import { Sheet } from "../components/Sheet";
import { fmtMinutesLeft } from "../format";
import { navigate } from "../router";
import { groupStages } from "../stages";

export function Indexing({ movie, onChanged }: { movie: Movie; onChanged: () => void }) {
  const [snap, setSnap] = useState<IndexProgress | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [actionError, setActionError] = useState<ApiError | null>(null);
  const { announce } = useAnnouncer();
  const lastFetch = useRef(0);
  const pending = useRef<number>(undefined);

  const refresh = useCallback(async () => {
    lastFetch.current = Date.now();
    try {
      const s = await api.progress(movie.id);
      setSnap(s);
      if (s.status !== movie.status) onChanged();
    } catch {
      /* keep the last snapshot; SSE or the next resume will catch up */
    }
  }, [movie.id, movie.status, onChanged]);

  // Each SSE event triggers a snapshot refresh, at most once a second.
  const throttled = useCallback(() => {
    const wait = 1000 - (Date.now() - lastFetch.current);
    if (wait <= 0) refresh();
    else if (!pending.current) {
      pending.current = window.setTimeout(() => {
        pending.current = undefined;
        refresh();
      }, wait);
    }
  }, [refresh]);

  useEffect(() => {
    refresh();
    return () => window.clearTimeout(pending.current);
  }, [refresh]);

  // Follow the active job. iOS kills background connections, so on resume we
  // refetch the snapshot and reopen the stream.
  const jobId = snap?.job?.id;
  const jobLive = snap?.job && (snap.job.state === "running" || snap.job.state === "queued");
  const [epoch, setEpoch] = useState(0);
  useEffect(() => {
    if (!jobId || !jobLive) return;
    return followJob(jobId, (j) => {
      throttled();
      if (["done", "failed", "cancelled"].includes(j.state)) {
        refresh();
        onChanged();
      }
    });
  }, [jobId, jobLive, throttled, refresh, onChanged, epoch]);

  useEffect(() => {
    const onVis = () => {
      if (document.visibilityState === "visible") {
        refresh();
        setEpoch((n) => n + 1);
      }
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, [refresh]);

  // Safety net for a queued job: poll until a worker picks it up.
  useEffect(() => {
    if (snap?.job?.state !== "queued") return;
    const t = window.setInterval(refresh, 3000);
    return () => window.clearInterval(t);
  }, [snap?.job?.state, refresh]);

  const groups = snap ? groupStages(snap.stages) : [];

  // Announce stage changes politely, and percentages only every 25%.
  const spoken = useRef<string>("");
  useEffect(() => {
    const active = groups.find((g) => g.state === "running");
    if (!active) return;
    const key = `${active.label} ${Math.floor(active.pct * 4) * 25}`;
    if (key !== spoken.current) {
      const pct = Math.floor(active.pct * 4) * 25;
      announce(pct ? `${active.label}, ${pct} percent` : active.label);
      spoken.current = key;
    }
  });

  const failed = movie.status === "failed";
  const cancelled = movie.status === "cancelled";
  const job = snap?.job;
  const isLocal = !snap?.gpu || snap.gpu === "local";

  async function act(fn: () => Promise<unknown>) {
    setActionError(null);
    try {
      await fn();
      onChanged();
      refresh();
    } catch (e) {
      setActionError(e as ApiError);
    }
  }

  const eta = fmtMinutesLeft(snap?.eta_s);
  const leaveNote = (
    <p className="measure">{eta ? `${eta} ` : ""}You can close this page. Indexing keeps going.</p>
  );

  return (
    <div className="indexing">
      <section aria-label="Indexing progress">
        {!snap && <p className="fog">Loading progress…</p>}
        <ol className="stages">
          {groups.map((g, i) => (
            <li key={g.label} className={`stage ${g.state}`}>
              <span className="n num">{i + 1}</span>
              <span className="label">
                {g.label}
                {g.label === "Getting a GPU" && snap?.gpu && !isLocal ? `  ${snap.gpu}` : ""}
              </span>
              <span className={`state num ${g.state === "failed" ? "danger" : "fog"}`}>
                {g.state === "done"
                  ? "done"
                  : g.state === "running"
                    ? `${Math.floor(g.pct * 100)}%`
                    : g.state === "failed"
                      ? "failed"
                      : ""}
              </span>
              {g.state === "running" && (
                <span className="bar" aria-hidden="true">
                  <span style={{ width: `${Math.max(2, g.pct * 100)}%` }} />
                </span>
              )}
            </li>
          ))}
        </ol>
      </section>

      <aside>
        {failed && (
          <div role="alert">
            <h2 className="danger">{job?.kind === "ingest" ? "Adding failed" : "Indexing failed"}</h2>
            <p className="measure">{movie.error || job?.error}</p>
            <ErrorDetails detail={job?.detail} />
            <div className="sheet-actions">
              <button className="btn btn-primary" onClick={() => act(() => api.retry(movie.id))}>
                Retry
              </button>
              <button
                className="btn btn-danger"
                onClick={() =>
                  act(async () => {
                    await api.remove(movie.id);
                    navigate("/", { replace: true });
                  })
                }
              >
                Remove movie
              </button>
            </div>
          </div>
        )}
        {cancelled && (
          <div>
            <h2>Indexing stopped.</h2>
            <p className="measure">The download is kept, so starting again skips it.</p>
            <div className="sheet-actions">
              <button className="btn btn-primary" onClick={() => act(() => api.retry(movie.id))}>
                Retry
              </button>
              <button
                className="btn btn-danger"
                onClick={() =>
                  act(async () => {
                    await api.remove(movie.id);
                    navigate("/", { replace: true });
                  })
                }
              >
                Remove movie
              </button>
            </div>
          </div>
        )}
        {!failed && !cancelled && (
          <>
            {leaveNote}
            <p className="fog">
              {isLocal ? "Indexing on this computer" : `GPU: Colab ${snap?.gpu}`}
              {snap?.compute_units_used != null && (
                <>
                  <br />
                  Used so far: <span className="num">{snap.compute_units_used.toFixed(1)}</span> compute units
                </>
              )}
            </p>
            {job && (job.state === "running" || job.state === "queued") && (
              <button className="btn btn-danger" onClick={() => setConfirming(true)}>
                Cancel indexing
              </button>
            )}
          </>
        )}
        {actionError && (
          <p className="message danger" role="alert">
            {actionError.message}
          </p>
        )}
      </aside>

      {confirming && job && (
        <Sheet label="Stop indexing?" onClose={() => setConfirming(false)}>
          <h2>Stop indexing {movie.title || "this movie"}?</h2>
          <p>
            {isLocal
              ? "Indexing stops and the download is kept."
              : "The GPU is released and the download is kept."}
          </p>
          <div className="sheet-actions">
            <button
              className="btn btn-primary btn-block"
              onClick={() => {
                setConfirming(false);
                act(() => api.cancel(job.id));
              }}
            >
              Stop indexing
            </button>
            <button className="btn btn-block" data-autofocus onClick={() => setConfirming(false)}>
              Keep going
            </button>
          </div>
        </Sheet>
      )}
    </div>
  );
}
