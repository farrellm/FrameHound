import { timecodeChars } from "../format";

/** Big Shoulders digits in fixed cells, so the counter doesn't shimmy. */
export function Timecode({ t, fps }: { t: number; fps: number }) {
  return (
    <div className="timecode" aria-hidden="true">
      {timecodeChars(t, fps).map((c, i) => (
        <span key={i} className={/\d/.test(c) ? "d" : "s"}>
          {c}
        </span>
      ))}
    </div>
  );
}
