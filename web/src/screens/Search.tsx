// Search and timeline (ui-design.md §4.3).
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api, proxyUrl, type Candidate, type Movie, type SearchResult, type Shot } from "../api";
import { useAnnouncer } from "../components/Announcer";
import { Sheet } from "../components/Sheet";
import { MiniStrip } from "../components/MiniStrip";
import { Timecode } from "../components/Timecode";
import { fmtShort, fmtSpan, spoken } from "../format";
import { linkProps, setQuery } from "../router";
import { ExportSheet } from "./ExportSheet";
import { Strip, type View } from "./Strip";

type Mode = "describe" | "picture" | "quote";
const MODES: { id: Mode; label: string; short: string; type: string }[] = [
  { id: "describe", label: "Describe it", short: "Describe", type: "text" },
  { id: "picture", label: "Show a picture", short: "Picture", type: "image" },
  { id: "quote", label: "Quote a line", short: "Quote", type: "dialogue" },
];
const REASON: Record<Candidate["reason"], string> = {
  visual: "Looks like",
  face: "Same face",
  dialogue: "Says",
};
const ZOOM_PAD_S = 10;

const coarse = () => window.matchMedia("(pointer: coarse)").matches;
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function Search({ movie, onChanged }: { movie: Movie; onChanged: () => void }) {
  const duration = movie.duration_s || 0;
  const fps = movie.fps || 24;
  const frame = 1 / fps;
  const initial = useMemo(() => new URLSearchParams(location.search), []);
  // Show a picture needs image search, which arrives with faces (M3).
  const available = (m: Mode) =>
    m === "quote" ? movie.capabilities.dialogue : m === "describe" ? movie.capabilities.visual : false;

  const video = useRef<HTMLVideoElement>(null);
  const [t, setT] = useState(0);
  const [mode, setMode] = useState<Mode>(() => {
    const m = initial.get("mode") as Mode | null;
    if (m && MODES.some((x) => x.id === m) && available(m)) return m;
    return movie.capabilities.visual ? "describe" : "quote";
  });
  const [shots, setShots] = useState<Shot[]>([]);
  useEffect(() => {
    if (!movie.capabilities.visual) return;
    api.shots(movie.id).then(setShots, () => {});
  }, [movie.id, movie.capabilities.visual]);
  const [q, setQ] = useState(initial.get("q") ?? "");
  const [result, setResult] = useState<SearchResult | null>(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [modeNote, setModeNote] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const num = (k: string) => {
    const v = initial.get(k);
    return v != null && isFinite(+v) ? +v : null;
  };
  const [inPt, setIn] = useState<number | null>(num("in"));
  const [outPt, setOut] = useState<number | null>(num("out"));
  const [view, setView] = useState<View>({ from: 0, to: duration || 1 });
  const [sweepKey, setSweepKey] = useState(0);
  const [exporting, setExporting] = useState(false);
  const [showKeys, setShowKeys] = useState(false);
  const [typing, setTyping] = useState(false);
  const [sideOpen, setSideOpen] = useState(false);
  const { announce } = useAnnouncer();

  // Keep the selection in the URL so a reload (or a shared link) restores it.
  useEffect(() => {
    setQuery({
      in: inPt != null ? inPt.toFixed(2) : null,
      out: outPt != null ? outPt.toFixed(2) : null,
    });
  }, [inPt, outPt]);

  // Follow the video: rAF while playing, events otherwise.
  useEffect(() => {
    const v = video.current!;
    let raf = 0;
    const tick = () => {
      setT(v.currentTime);
      if (!v.paused) raf = requestAnimationFrame(tick);
    };
    const on = () => {
      cancelAnimationFrame(raf);
      tick();
    };
    v.addEventListener("play", on);
    v.addEventListener("seeked", on);
    v.addEventListener("timeupdate", on);
    v.addEventListener("pause", on);
    return () => {
      cancelAnimationFrame(raf);
      v.removeEventListener("play", on);
      v.removeEventListener("seeked", on);
      v.removeEventListener("timeupdate", on);
      v.removeEventListener("pause", on);
    };
  }, []);

  const seek = useCallback(
    (to: number, final = true) => {
      const v = video.current;
      const clamped = Math.min(Math.max(to, 0), duration);
      setT(clamped);
      if (!v) return;
      if (!final && "fastSeek" in v) v.fastSeek(clamped);
      else v.currentTime = clamped;
    },
    [duration],
  );

  // Zoom the strip, animated over 250 ms (instant with reduced motion).
  const viewRef = useRef(view);
  viewRef.current = view;
  const animateView = useCallback((target: View) => {
    const start = viewRef.current;
    if (reducedMotion()) return setView(target);
    const t0 = performance.now();
    const step = (now: number) => {
      const k = Math.min((now - t0) / 250, 1);
      const e = 1 - (1 - k) ** 3;
      setView({
        from: start.from + (target.from - start.from) * e,
        to: start.to + (target.to - start.to) * e,
      });
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, []);
  const whole = useCallback(() => animateView({ from: 0, to: duration || 1 }), [animateView, duration]);
  const zoomed = view.from > 0.01 || view.to < duration - 0.01;

  const runSearch = useCallback(
    async (text: string, m: Mode) => {
      const query = text.trim();
      if (!query) return;
      setSearching(true);
      setError(null);
      setQuery({ q: query, mode: m });
      try {
        const bins = Math.min(4096, Math.max(256, Math.round(window.innerWidth * 1.5)));
        const r = await api.search(movie.id, {
          type: MODES.find((x) => x.id === m)!.type,
          q: query,
          heat_bins: bins,
        });
        setResult(r);
        setSelected(null);
        setSweepKey((k) => k + 1);
        announce(
          r.candidates.length
            ? `${r.candidates.length} match${r.candidates.length === 1 ? "" : "es"}`
            : "No clear matches",
        );
      } catch (e) {
        setError(e as ApiError);
        announce((e as ApiError).message, true);
      } finally {
        setSearching(false);
      }
    },
    [movie.id, announce],
  );

  // A query restored from the URL runs on arrival.
  useEffect(() => {
    if (initial.get("q")) runSearch(initial.get("q")!, mode);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const select = useCallback(
    (i: number) => {
      const c = result?.candidates[i];
      if (!c) return;
      setSelected(i);
      setIn(c.start);
      setOut(c.end);
      animateView({ from: Math.max(0, c.start - ZOOM_PAD_S), to: Math.min(duration, c.end + ZOOM_PAD_S) });
      const v = video.current;
      if (v) {
        seek(c.peak_t);
        v.muted = coarse(); // muted on iPhone, sound on desktop
        v.play().catch(() => {});
      }
    },
    [result, animateView, duration, seek],
  );

  const setInAt = useCallback(
    (x: number) => {
      const v = Math.min(Math.max(x, 0), duration);
      setIn(v);
      if (outPt != null && outPt <= v) setOut(null);
      navigator.vibrate?.(8);
    },
    [duration, outPt],
  );
  const setOutAt = useCallback(
    (x: number) => {
      const v = Math.min(Math.max(x, 0), duration);
      setOut(v);
      if (inPt != null && inPt >= v) setIn(null);
      navigator.vibrate?.(8);
    },
    [duration, inPt],
  );
  const dragIn = (x: number) => setIn(Math.min(Math.max(x, 0), (outPt ?? duration) - frame));
  const dragOut = (x: number) => setOut(Math.max(Math.min(x, duration), (inPt ?? 0) + frame));

  const canCut = inPt != null && outPt != null && outPt > inPt;

  // Desktop keys, following editing convention.
  const shuttle = useRef({ dir: 0, speed: 0, raf: 0 });
  useEffect(() => {
    const stopReverse = () => {
      cancelAnimationFrame(shuttle.current.raf);
      shuttle.current.raf = 0;
    };
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (
        exporting ||
        showKeys ||
        el.closest("input, textarea, [role=dialog]") ||
        e.metaKey ||
        e.ctrlKey ||
        e.altKey
      )
        return;
      const v = video.current;
      if (!v) return;
      const sh = shuttle.current;
      const key = e.key.toLowerCase();
      if (key === "l" || key === "j") {
        const dir = key === "l" ? 1 : -1;
        sh.speed = sh.dir === dir ? Math.min(sh.speed * 2, 8) : 1;
        sh.dir = dir;
        stopReverse();
        if (dir > 0) {
          v.playbackRate = sh.speed;
          v.play().catch(() => {});
        } else {
          v.pause();
          let last = performance.now();
          const back = (now: number) => {
            v.currentTime = Math.max(0, v.currentTime - ((now - last) / 1000) * sh.speed);
            last = now;
            if (v.currentTime > 0) sh.raf = requestAnimationFrame(back);
          };
          sh.raf = requestAnimationFrame(back);
        }
      } else if (key === "k") {
        sh.dir = 0;
        stopReverse();
        v.pause();
        v.playbackRate = 1;
      } else if (key === "i") setInAt(v.currentTime);
      else if (key === "o") setOutAt(v.currentTime);
      else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        if (el.closest("[role=slider]")) return; // the slider handles its own arrows
        seek(v.currentTime + (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 0.5 : frame));
      } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        const n = result?.candidates.length ?? 0;
        if (!n) return;
        const cur = selected ?? -1;
        select(e.key === "ArrowDown" ? Math.min(cur + 1, n - 1) : Math.max(cur - 1, 0));
      } else if (e.key === "Enter") {
        if (!canCut || el.closest("button, a")) return;
        setExporting(true);
      } else if (e.key === "?") setShowKeys(true);
      else return;
      e.preventDefault();
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      stopReverse();
    };
  }, [exporting, showKeys, frame, seek, setInAt, setOutAt, result, selected, select, canCut]);

  const candidates = result?.candidates ?? [];
  const current = selected != null ? candidates[selected] : null;
  const modelsMismatch = movie.models_ok === false || error?.code === "models_mismatch";

  const queryBar = (
    <form
      className="querybar"
      role="search"
      onSubmit={(e) => {
        e.preventDefault();
        runSearch(q, mode);
        (document.activeElement as HTMLElement | null)?.blur?.();
      }}
    >
      <div className="segmented" role="radiogroup" aria-label="How to search">
        {MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            role="radio"
            aria-checked={mode === m.id}
            aria-disabled={!available(m.id)}
            aria-label={m.label}
            onClick={() => {
              if (!available(m.id))
                return setModeNote(
                  m.id === "picture"
                    ? "Searching by picture isn't available yet. Describe it in words instead."
                    : "This movie was indexed for dialogue only. Re-index it to describe what's on screen.",
                );
              setModeNote(null);
              setMode(m.id);
            }}
          >
            <span className="seg-label">{window.innerWidth >= 960 ? m.label : m.short}</span>
          </button>
        ))}
      </div>
      {modeNote && <p className="note">{modeNote}</p>}
      <div className="query-row">
        <label htmlFor="q" className="visually-hidden">
          {MODES.find((m) => m.id === mode)!.label}
        </label>
        {mode === "describe" ? (
          <textarea
            id="q"
            className="field"
            rows={Math.min(3, Math.max(1, Math.ceil(q.length / 32)))}
            placeholder="Describe what's on screen, like a man in a hat on the phone"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                runSearch(q, mode);
              }
            }}
            onFocus={() => setTyping(coarse())}
            onBlur={() => setTyping(false)}
            enterKeyHint="search"
          />
        ) : (
          <input
            id="q"
            className="field"
            type="text"
            placeholder="Type a line of dialogue, even roughly"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onFocus={() => setTyping(coarse())}
            onBlur={() => setTyping(false)}
            enterKeyHint="search"
            autoComplete="off"
          />
        )}
        <button className="btn btn-primary" type="submit" disabled={searching || !q.trim()}>
          {searching ? "Finding…" : "Find"}
        </button>
      </div>
    </form>
  );

  return (
    <main className={`page search${typing ? " typing" : ""}${sideOpen ? " side-open" : ""}`}>
      <div className="main">
        <nav className="topbar">
          <a className="back" {...linkProps("/")}>
            ‹ Library
          </a>
          <h1 className="movie-title" style={{ fontSize: "var(--t1)", margin: 0 }}>
            {movie.title || "Untitled movie"}
          </h1>
        </nav>
        <div className="player">
          <video ref={video} src={proxyUrl(movie.id)} playsInline preload="metadata" controls />
        </div>
        <div className="tc-row">
          <Timecode t={t} fps={fps} />
          {zoomed && (
            <button className="btn btn-text whole" onClick={whole}>
              Whole movie
            </button>
          )}
        </div>
        <div className="strip-area">
          <Strip
            duration={duration}
            fps={fps}
            t={t}
            view={view}
            heat={result?.heat ?? null}
            threshold={result?.heat_threshold ?? 3}
            sweepKey={sweepKey}
            dim={searching}
            inPt={inPt}
            outPt={outPt}
            shots={shots}
            zoomed={zoomed}
            onSeek={seek}
            onIn={dragIn}
            onOut={dragOut}
            onWhole={whole}
          />
        </div>
        <div className="precision">
          <button className="btn" onClick={() => setInAt(t)}>
            Set in
          </button>
          <span className="nudges">
            <button className="btn" aria-label="Back half a second" onClick={() => seek(t - 0.5)}>
              ‹‹
            </button>
            <button className="btn" aria-label="Back one frame" onClick={() => seek(t - frame)}>
              ‹
            </button>
            <button className="btn" aria-label="Forward one frame" onClick={() => seek(t + frame)}>
              ›
            </button>
            <button className="btn" aria-label="Forward half a second" onClick={() => seek(t + 0.5)}>
              ››
            </button>
          </span>
          <button className="btn" onClick={() => setOutAt(t)}>
            Set out
          </button>
        </div>
        <button className="btn btn-primary btn-block" disabled={!canCut} onClick={() => setExporting(true)}>
          Cut clip{canCut ? <span className="num">&nbsp;{fmtShort(outPt! - inPt!)}</span> : null}
        </button>
      </div>

      <div className="side">
        {queryBar}
        <section aria-label="Matches" aria-busy={searching}>
          {modelsMismatch ? (
            <div role="alert">
              <p>This movie was indexed with older models. Re-index it to search.</p>
              <button
                className="btn btn-primary"
                onClick={async () => {
                  try {
                    await api.retry(movie.id);
                    onChanged();
                  } catch (e) {
                    setError(e as ApiError);
                  }
                }}
              >
                Re-index
              </button>
            </div>
          ) : error ? (
            <p className="message danger" role="alert">
              {error.message}
            </p>
          ) : !result ? (
            <p className="fog measure">
              What are you looking for? Describe it, show a picture, or quote a line.
            </p>
          ) : candidates.length === 0 ? (
            <p className="measure">
              No clear matches. The strongest moments are marked on the strip. Try fewer words, or a different
              mode.
            </p>
          ) : (
            <>
              <h2 className="num" style={{ marginTop: 8 }}>
                {candidates.length} match{candidates.length === 1 ? "" : "es"}
              </h2>
              <ul className="matches">
                {candidates.map((c, i) => (
                  <li key={`${c.start}-${i}`}>
                    <button
                      className="match"
                      aria-current={selected === i}
                      aria-label={`${REASON[c.reason]}${c.matched_words?.length ? ` "${words(c)}"` : ""}, ${spoken(c.start)} to ${spoken(c.end)}, match ${i + 1} of ${candidates.length}`}
                      onClick={() => {
                        select(i);
                        setSideOpen(false);
                      }}
                    >
                      <MiniStrip thumbs={c.thumbs} count={3} />
                      <span>
                        <span className="reason">{REASON[c.reason]}</span>
                        {c.matched_words?.length ? <span className="quote">“{words(c)}”</span> : null}
                        <span className="when num">{fmtSpan(c.start, c.end)}</span>
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
        <p className="note desktop-only">
          Press <kbd>?</kbd> for keyboard shortcuts.
        </p>
      </div>

      <button
        className="btn landscape-handle"
        aria-expanded={sideOpen}
        onClick={() => setSideOpen((o) => !o)}
      >
        {sideOpen ? "Hide matches" : "Matches and search"}
      </button>

      {exporting && canCut && (
        <ExportSheet
          movie={movie}
          start={inPt!}
          end={outPt!}
          hasFace={!!current?.has_face}
          onClose={() => setExporting(false)}
        />
      )}
      {showKeys && <KeysSheet onClose={() => setShowKeys(false)} />}
    </main>
  );
}

function words(c: Candidate): string {
  return (c.matched_words ?? []).map((w) => w.word).join(" ");
}

function KeysSheet({ onClose }: { onClose: () => void }) {
  const rows: [string, string][] = [
    ["J  K  L", "Reverse, pause, forward. Repeat J or L to go faster."],
    ["I  O", "Set in, set out"],
    ["←  →", "One frame. With Shift, half a second."],
    ["↑  ↓", "Previous or next match"],
    ["Enter", "Cut clip"],
  ];
  return (
    <Sheet label="Keyboard shortcuts" onClose={onClose}>
      <h2>Keyboard shortcuts</h2>
      <dl className="keys">
        {rows.map(([k, v]) => (
          <div key={k} style={{ display: "contents" }}>
            <dt>
              {k.split("  ").map((x) => (
                <kbd key={x} style={{ marginRight: 4 }}>
                  {x}
                </kbd>
              ))}
            </dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      <div className="sheet-actions">
        <button className="btn btn-block" onClick={onClose}>
          Done
        </button>
      </div>
    </Sheet>
  );
}
