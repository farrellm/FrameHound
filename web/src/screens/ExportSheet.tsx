// Export (ui-design.md §4.4): turn the span into a file on the device.
import { useEffect, useRef, useState } from "react";
import { ApiError, api, clipFileUrl, followJob, type Movie } from "../api";
import { useAnnouncer } from "../components/Announcer";
import { ErrorDetails } from "../components/ErrorDetails";
import { Sheet } from "../components/Sheet";
import { fmtLength, fmtPrecise } from "../format";

type Phase =
  | { kind: "choose" }
  | { kind: "cutting"; pct: number }
  | { kind: "ready"; clipId: string }
  | { kind: "failed"; error: ApiError };

export function ExportSheet({
  movie,
  start,
  end,
  faceBox,
  onClose,
}: {
  movie: Movie;
  start: number;
  end: number;
  /** A Same face match's face at its peak; enables Crop to the face. */
  faceBox: number[] | null;
  onClose: () => void;
}) {
  const fps = movie.fps || 24;
  const [mode, setMode] = useState<"accurate" | "fast">("accurate");
  const [crop, setCrop] = useState(false);
  // Cropping re-encodes, so it always cuts exactly.
  const effectiveMode = crop ? "accurate" : mode;
  const [phase, setPhase] = useState<Phase>({ kind: "choose" });
  const [file, setFile] = useState<File | null>(null);
  const stop = useRef<() => void>(undefined);
  const { announce, toast } = useAnnouncer();
  const fileName = `${
    (movie.title || "clip")
      .replace(/[^A-Za-z0-9]+/g, "-")
      .replace(/^-|-$/g, "")
      .slice(0, 60) || "clip"
  }-${fmtPrecise(start, fps).replace(/[:.]/g, "-")}.mp4`;

  useEffect(() => () => stop.current?.(), []);

  async function cut() {
    setPhase({ kind: "cutting", pct: 0 });
    announce("Cutting");
    try {
      const { clip_id, job_id } = await api.cutClip(movie.id, {
        start,
        end,
        mode: effectiveMode,
        ...(crop && faceBox ? { crop: true, face_bbox: faceBox } : {}),
      });
      stop.current = followJob(job_id, (j) => {
        if (j.state === "done") {
          setPhase({ kind: "ready", clipId: clip_id });
          announce("Clip ready");
        } else if (j.state === "failed" || j.state === "cancelled") {
          setPhase({
            kind: "failed",
            error: new ApiError(500, {
              code: j.error_code,
              message: j.error ?? "The clip couldn't be cut.",
              detail: j.detail,
            }),
          });
        } else setPhase({ kind: "cutting", pct: j.pct ?? 0 });
      });
    } catch (e) {
      setPhase({ kind: "failed", error: e as ApiError });
      announce((e as ApiError).message, true);
    }
  }

  // Fetch the file as soon as it's ready, so Save clip can open the share sheet
  // straight from the tap (iOS needs the user gesture to still be fresh).
  const clipId = phase.kind === "ready" ? phase.clipId : null;
  useEffect(() => {
    if (!clipId) return;
    let live = true;
    fetch(clipFileUrl(clipId))
      .then((r) => r.blob())
      .then((b) => live && setFile(new File([b], fileName, { type: "video/mp4" })))
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [clipId, fileName]);

  const share =
    typeof navigator.canShare === "function" &&
    window.matchMedia("(pointer: coarse)").matches &&
    file &&
    navigator.canShare({ files: [file] });

  async function save() {
    if (!clipId) return;
    if (share && file) {
      try {
        await navigator.share({ files: [file], title: movie.title ?? "Clip" });
        return;
      } catch (e) {
        if ((e as DOMException).name === "AbortError") return;
      }
    }
    const a = document.createElement("a");
    a.href = file ? URL.createObjectURL(file) : clipFileUrl(clipId);
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  const len = end - start;
  return (
    <Sheet label="Cut clip" onClose={onClose}>
      <h2>{phase.kind === "ready" ? "Clip ready" : "Cut clip"}</h2>
      <p className="num">
        {fmtPrecise(start, fps)} – {fmtPrecise(end, fps)}
        <span className="fog">&ensp;{fmtLength(len)}</span>
      </p>

      {phase.kind === "ready" ? (
        <>
          <video className="preview" src={clipFileUrl(phase.clipId)} autoPlay loop muted playsInline />
          <div className="sheet-actions">
            <button className="btn btn-primary btn-block" data-autofocus onClick={save}>
              Save clip
            </button>
            <button
              className="btn btn-block"
              onClick={async () => {
                try {
                  await navigator.clipboard.writeText(new URL(clipFileUrl(phase.clipId), location.href).href);
                  toast("Link copied");
                } catch {
                  toast("Couldn't copy the link");
                }
              }}
            >
              Copy link
            </button>
          </div>
        </>
      ) : (
        <>
          <fieldset>
            <legend>Format</legend>
            <label className="radio">
              <input type="radio" name="format" checked readOnly />
              <span>Video (MP4)</span>
            </label>
          </fieldset>
          <fieldset disabled={phase.kind === "cutting"}>
            <legend>Cutting</legend>
            <label className="radio">
              <input
                type="radio"
                name="mode"
                checked={effectiveMode === "accurate"}
                onChange={() => setMode("accurate")}
              />
              <span>
                Exact <span className="hint">cuts on the frame</span>
              </span>
            </label>
            <label className="radio">
              <input
                type="radio"
                name="mode"
                checked={effectiveMode === "fast"}
                disabled={crop}
                onChange={() => setMode("fast")}
              />
              <span>
                Quick{" "}
                <span className="hint">
                  {crop ? "not available when cropping to the face" : "instant, may start a moment early"}
                </span>
              </span>
            </label>
          </fieldset>
          {faceBox && (
            <fieldset disabled={phase.kind === "cutting"}>
              <label className="radio">
                <input type="checkbox" checked={crop} onChange={(e) => setCrop(e.target.checked)} />
                <span>
                  Crop to the face <span className="hint">portrait, framed on the face at the match</span>
                </span>
              </label>
            </fieldset>
          )}
          {phase.kind === "failed" && (
            <div role="alert">
              <p className="message danger">{phase.error.message}</p>
              <ErrorDetails detail={phase.error.detail} />
            </div>
          )}
          <div className="sheet-actions">
            <button
              className="btn btn-primary btn-block cutting"
              disabled={phase.kind === "cutting"}
              onClick={cut}
              data-autofocus
            >
              {phase.kind === "cutting" && (
                <span className="fill" style={{ width: `${Math.round(phase.pct * 100)}%` }} />
              )}
              <span className="label">{phase.kind === "cutting" ? "Cutting…" : "Cut clip"}</span>
            </button>
          </div>
        </>
      )}
    </Sheet>
  );
}
