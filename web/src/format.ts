// Time formatting. Spans and durations use tabular numerals in the body face.

const pad = (n: number, w = 2) => String(Math.floor(n)).padStart(w, "0");

/** 1:36:12 or 9:15 */
export function fmtDuration(s: number | null | undefined): string {
  if (s == null || !isFinite(s)) return "";
  s = Math.round(s);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${m}:${pad(s % 60)}`;
}

/** 0:14:02 */
export function fmtClock(s: number): string {
  s = Math.max(0, Math.floor(s));
  return `${Math.floor(s / 3600)}:${pad((s % 3600) / 60)}:${pad(s % 60)}`;
}

export const fmtSpan = (a: number, b: number) => `${fmtClock(a)}–${fmtClock(b)}`;

/** Frames within the second, as editors count them. */
function frames(s: number, fps: number): number {
  return Math.min(Math.floor((s - Math.floor(s)) * fps + 1e-6), Math.ceil(fps) - 1);
}

/** 0:14:02.12 (h:mm:ss.ff) */
export function fmtPrecise(s: number, fps: number): string {
  return `${fmtClock(s)}.${pad(frames(Math.max(0, s), fps))}`;
}

/** The big playhead timecode, 00:14:06.12, as an array of characters. */
export function timecodeChars(s: number, fps: number): string[] {
  s = Math.max(0, s);
  const t = Math.floor(s);
  return `${pad(t / 3600)}:${pad((t % 3600) / 60)}:${pad(t % 60)}.${pad(frames(s, fps))}`.split("");
}

/** "17 s", "2 min 5 s" */
export function fmtLength(s: number): string {
  const r = Math.round(s);
  if (r < 60) return `${Math.max(r, 1)} s`;
  return r % 60 ? `${Math.floor(r / 60)} min ${r % 60} s` : `${r / 60} min`;
}

/** 0:17 for the Cut clip button */
export const fmtShort = (s: number) => fmtDuration(Math.max(1, Math.round(s)));

/** "14 minutes 6 seconds", for aria-valuetext */
export function spoken(s: number): string {
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const parts: string[] = [];
  const unit = (n: number, w: string) => `${n} ${w}${n === 1 ? "" : "s"}`;
  if (h) parts.push(unit(h, "hour"));
  if (m) parts.push(unit(m, "minute"));
  if (sec || !parts.length) parts.push(unit(sec, "second"));
  return parts.join(" ");
}

export function fmtMinutesLeft(eta: number | null | undefined): string | null {
  if (eta == null || !isFinite(eta)) return null;
  if (eta < 60) return "Less than a minute left.";
  const m = Math.round(eta / 60);
  return `About ${m} minute${m === 1 ? "" : "s"} left.`;
}
