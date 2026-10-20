// The film strip (ui-design.md §4.3): Emulsion body, sprocket edges, frames in
// the gate at their real times, grease-pencil heat under the gate (a red bar over each listed match), china-marker
// in/out brackets, and the playhead.
import { useLayoutEffect, useRef, useState } from "react";
import type { Shot } from "../api";
import { spoken } from "../format";

export interface View {
  from: number;
  to: number;
}

interface Props {
  duration: number;
  fps: number;
  t: number;
  view: View;
  heat: number[] | null;
  /** The listed matches: their heat is drawn in the red marker. */
  matches: { start: number; end: number }[];
  threshold: number;
  sweepKey: number;
  dim: boolean;
  inPt: number | null;
  outPt: number | null;
  shots: Shot[];
  zoomed: boolean;
  onSeek: (t: number, final: boolean) => void;
  onIn: (t: number) => void;
  onOut: (t: number) => void;
  onWhole: () => void;
}

const GATE_H = 64;
const HEAT_H = 34;
const SPROCKET_H = 9;
const FRAME_W = Math.round((GATE_H * 4) / 3);

export function Strip(p: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(320);

  useLayoutEffect(() => {
    const el = wrap.current!;
    const ro = new ResizeObserver(() => setW(el.clientWidth || 320));
    ro.observe(el);
    setW(el.clientWidth || 320);
    return () => ro.disconnect();
  }, []);

  const span = Math.max(p.view.to - p.view.from, 0.001);
  const x = (t: number) => ((t - p.view.from) / span) * w;
  const timeAt = (clientX: number) => {
    const r = wrap.current!.getBoundingClientRect();
    const f = (clientX - r.left) / r.width;
    return Math.min(Math.max(p.view.from + f * span, 0), p.duration);
  };

  // Scrub with one finger; two fingers pinching together return to the whole movie.
  const pointers = useRef(new Map<number, number>());
  const pinchStart = useRef<number | null>(null);
  function down(e: React.PointerEvent) {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    pointers.current.set(e.pointerId, e.clientX);
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      pinchStart.current = Math.abs(a - b);
    } else {
      p.onSeek(timeAt(e.clientX), false);
    }
  }
  function move(e: React.PointerEvent) {
    if (!pointers.current.has(e.pointerId)) return;
    pointers.current.set(e.pointerId, e.clientX);
    if (pointers.current.size === 2 && pinchStart.current != null) {
      const [a, b] = [...pointers.current.values()];
      if (Math.abs(a - b) < pinchStart.current * 0.7 && p.zoomed) {
        pinchStart.current = null;
        p.onWhole();
      }
      return;
    }
    if (pointers.current.size === 1) p.onSeek(timeAt(e.clientX), false);
  }
  function up(e: React.PointerEvent) {
    if (!pointers.current.has(e.pointerId)) return;
    const wasSingle = pointers.current.size === 1;
    pointers.current.delete(e.pointerId);
    if (pointers.current.size < 2) pinchStart.current = null;
    if (wasSingle && e.type === "pointerup") p.onSeek(timeAt(e.clientX), true);
  }

  // Frames in the gate at real positions in time, each showing the shot playing
  // at its middle (empty cells for a movie without a visual index).
  const step = niceStep((span * (FRAME_W + 3)) / w);
  const frames: { t: number; left: number; width: number; thumb?: string }[] = [];
  for (let k = Math.floor(p.view.from / step); k * step < p.view.to && frames.length < 400; k++) {
    const t0 = k * step;
    const left = x(t0);
    const width = x(Math.min(t0 + step, p.duration)) - left - 3;
    if (width > 2) frames.push({ t: t0, left, width, thumb: shotAt(p.shots, t0 + step / 2)?.thumb_url });
  }
  // Shot boundaries, only once the zoom spreads them far enough apart to read
  // (in the overview they'd be a picket fence over the frames).
  const cuts: number[] = [];
  const visible = p.shots.filter((sh) => sh.start > p.view.from && sh.start < p.view.to);
  if (visible.length && w / visible.length >= 24) for (const sh of visible) cuts.push(x(sh.start));

  return (
    <div className="table">
      {/* Top-right of the strip (ui-design.md §4.3), outside the drag surface. */}
      {p.zoomed && (
        <button className="btn whole" onClick={p.onWhole}>
          Whole movie
        </button>
      )}
      <div className="strip-wrap" ref={wrap}>
        <div
          className={`strip${p.dim ? " dim" : ""}`}
          role="slider"
          tabIndex={0}
          aria-label="Playhead"
          aria-valuemin={0}
          aria-valuemax={Math.round(p.duration)}
          aria-valuenow={Math.round(p.t)}
          aria-valuetext={spoken(p.t)}
          onPointerDown={down}
          onPointerMove={move}
          onPointerUp={up}
          onPointerCancel={up}
          onKeyDown={(e) => {
            const d = e.shiftKey ? 0.5 : 1 / p.fps;
            if (e.key === "ArrowLeft") p.onSeek(Math.max(0, p.t - d), true);
            else if (e.key === "ArrowRight") p.onSeek(Math.min(p.duration, p.t + d), true);
            else if (e.key === "Home") p.onSeek(0, true);
            else if (e.key === "End") p.onSeek(p.duration, true);
            else return;
            e.preventDefault();
            e.stopPropagation();
          }}
        >
          <div className="sprockets" />
          <div className="gate">
            {frames.map((f) => (
              <div
                key={f.t}
                className="frame"
                style={{
                  left: f.left,
                  width: f.width,
                  backgroundImage: f.thumb ? `url(${f.thumb})` : undefined,
                }}
              />
            ))}
            {cuts.map((cx) => (
              <div key={cx} className="cut" style={{ left: cx }} />
            ))}
          </div>
          <Heat key={p.sweepKey} {...p} w={w} x={x} />
          <div className="sprockets" />
          {p.inPt != null && p.outPt != null && (
            <div className="span" style={{ left: x(p.inPt), width: Math.max(x(p.outPt) - x(p.inPt), 1) }} />
          )}
          <div className="playhead" style={{ left: x(p.t) }} />
        </div>
        {p.inPt != null && (
          <Bracket
            kind="in"
            t={p.inPt}
            left={x(p.inPt)}
            duration={p.duration}
            fps={p.fps}
            timeAt={timeAt}
            onMove={p.onIn}
          />
        )}
        {p.outPt != null && (
          <Bracket
            kind="out"
            t={p.outPt}
            left={x(p.outPt)}
            duration={p.duration}
            fps={p.fps}
            timeAt={timeAt}
            onMove={p.onOut}
          />
        )}
      </div>
    </div>
  );
}

// The narrowest a listed match is drawn, in px.
const MATCH_W = 5;

function Heat(p: Props & { w: number; x: (t: number) => number }) {
  if (!p.heat?.length) return <svg className="heat" aria-hidden="true" height={HEAT_H} />;
  const n = p.heat.length;
  const binS = p.duration / n;
  const scale = Math.max(p.threshold * 1.6, ...p.heat.map((z) => z * 0.9), 1);
  const marks: React.ReactNode[] = [];
  const lo = Math.max(0, Math.floor(p.view.from / binS));
  const hi = Math.min(n, Math.ceil(p.view.to / binS));
  const height = (z: number) => Math.max(3, Math.min(z / scale, 1) * (HEAT_H - 4));
  const delay = (left: number) => ({ animationDelay: `${Math.round((Math.max(left, 0) / p.w) * 450)}ms` });
  for (let i = lo; i < hi; i++) {
    const z = p.heat[i];
    if (z < 1) continue; // below 1 MAD is noise, not a mark
    const x0 = p.x(i * binS);
    const x1 = p.x((i + 1) * binS);
    const h = height(z);
    // Above threshold the marks thicken into a solid bar; below, a thin stroke.
    const solid = z >= p.threshold;
    const width = solid ? Math.max(x1 - x0 + 0.5, 2) : Math.min(1.5, Math.max(x1 - x0, 1));
    // Kept inside the strip, so a mark in the last second is still visible.
    const left = Math.min(solid ? x0 : (x0 + x1) / 2 - width / 2, p.w - width);
    marks.push(<rect key={i} x={left} y={HEAT_H - 2 - h} width={width} height={h} style={delay(left)} />);
  }
  // Each listed match is one red bar over its span, as tall as its hottest bin. A match covers
  // a bin or two, under 2px in the overview, so it gets a minimum width to read at a glance.
  // Drawn last so the yellow marks around it can't cover it.
  const tops = p.matches.flatMap((m, k) => {
    if (m.end <= p.view.from || m.start >= p.view.to) return [];
    const b0 = Math.min(n - 1, Math.max(0, Math.floor(m.start / binS)));
    const b1 = Math.min(n, Math.max(b0 + 1, Math.ceil(m.end / binS)));
    const h = height(Math.max(...p.heat!.slice(b0, b1)));
    const x0 = p.x(m.start);
    const x1 = p.x(m.end);
    const width = Math.max(x1 - x0, MATCH_W);
    const left = Math.max(0, Math.min((x0 + x1) / 2 - width / 2, p.w - width));
    return [<rect key={`m${k}`} className="top" x={left} y={HEAT_H - 2 - h} width={width} height={h} style={delay(left)} />];
  });
  return (
    <svg
      className={`heat${p.sweepKey ? " sweep" : ""}`}
      aria-hidden="true"
      width={p.w}
      height={HEAT_H}
      viewBox={`0 0 ${p.w} ${HEAT_H}`}
    >
      {marks}
      {tops}
    </svg>
  );
}

function Bracket(p: {
  kind: "in" | "out";
  t: number;
  left: number;
  duration: number;
  fps: number;
  timeAt: (x: number) => number;
  onMove: (t: number) => void;
}) {
  const dragging = useRef(false);
  return (
    <button
      className={`bracket ${p.kind}`}
      style={{ left: p.left, top: SPROCKET_H - 4, height: GATE_H + 8 }}
      role="slider"
      aria-label={p.kind === "in" ? "Clip start" : "Clip end"}
      aria-valuemin={0}
      aria-valuemax={Math.round(p.duration)}
      aria-valuenow={Math.round(p.t)}
      aria-valuetext={spoken(p.t)}
      onPointerDown={(e) => {
        e.stopPropagation();
        dragging.current = true;
        e.currentTarget.setPointerCapture(e.pointerId);
      }}
      onPointerMove={(e) => dragging.current && p.onMove(p.timeAt(e.clientX))}
      onPointerUp={() => (dragging.current = false)}
      onPointerCancel={() => (dragging.current = false)}
      onKeyDown={(e) => {
        const d = e.shiftKey ? 0.5 : 1 / p.fps;
        if (e.key === "ArrowLeft") p.onMove(p.t - d);
        else if (e.key === "ArrowRight") p.onMove(p.t + d);
        else return;
        e.preventDefault();
        e.stopPropagation();
      }}
    >
      <span className="mark" />
    </button>
  );
}

function shotAt(shots: Shot[], t: number): Shot | undefined {
  let lo = 0;
  let hi = shots.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (shots[mid].end <= t) lo = mid + 1;
    else if (shots[mid].start > t) hi = mid - 1;
    else return shots[mid];
  }
  return undefined;
}

function niceStep(s: number): number {
  const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  return steps.find((x) => x >= s) ?? 3600;
}
