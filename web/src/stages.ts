// Pipeline stages grouped and named for people (ui-design.md §4.2).
import type { Stage, StageState } from "./api";

interface GroupDef {
  label: string;
  stages: Record<string, number>; // stage key → weight within the group
}

export const GROUPS: GroupDef[] = [
  { label: "Downloading", stages: { downloading: 0.6, probing: 0.05, transcoding_proxy: 0.35 } },
  { label: "Getting a GPU", stages: { provisioning_gpu: 1 } },
  { label: "Sending the movie", stages: { staging_media: 0.2, fetch: 0.8 } },
  { label: "Finding shots", stages: { shots: 1 } },
  { label: "Looking at frames", stages: { frames: 0.3, visual: 0.7 } },
  { label: "Finding faces", stages: { faces: 1 } },
  { label: "Listening to dialogue", stages: { audio: 0.85, sentences: 0.15 } },
  {
    label: "Packing the index",
    stages: { thumbs: 0.3, bundle: 0.3, downloading_index: 0.2, importing: 0.2 },
  },
];

export interface Group {
  label: string;
  state: StageState;
  pct: number;
  eta_s: number | null;
}

export function groupStages(stages: Stage[]): Group[] {
  const out: Group[] = [];
  for (const g of GROUPS) {
    const members = stages.filter((s) => s.stage in g.stages && s.state !== "skipped");
    if (!members.length) {
      // A wholly skipped group (e.g. download reused) still happened: show it done.
      if (stages.some((s) => s.stage in g.stages)) {
        out.push({ label: g.label, state: "done", pct: 1, eta_s: 0 });
      }
      continue;
    }
    let weight = 0;
    let done = 0;
    for (const m of members) {
      const w = g.stages[m.stage];
      weight += w;
      done += w * (m.state === "done" ? 1 : m.pct || 0);
    }
    const states = members.map((m) => m.state);
    let state: StageState = "pending";
    if (states.includes("failed")) state = "failed";
    else if (states.every((s) => s === "done")) state = "done";
    else if (states.includes("running") || states.includes("done")) state = "running";
    const etas = members
      .filter((m) => m.state === "running" && m.eta_s != null)
      .map((m) => m.eta_s as number);
    out.push({
      label: g.label,
      state,
      pct: weight ? done / weight : 0,
      eta_s: etas.length ? Math.max(...etas) : null,
    });
  }
  return out;
}

/** How far a running job is through its current group, as the movie page shows it. */
export function activeGroupPct(stage: string | null, stages: Stage[] | undefined): number | null {
  const def = GROUPS.find((g) => stage && stage in g.stages);
  if (!def || !stages) return null;
  return groupStages(stages).find((g) => g.label === def.label)?.pct ?? null;
}

/** The phrase a library row shows for a movie in progress: "Finding faces 62%". */
export function activeLabel(stage: string | null, pct: number | null): string {
  const def = GROUPS.find((g) => stage && stage in g.stages);
  if (!def) return "Waiting to start";
  return pct ? `${def.label} ${Math.floor(pct * 100)}%` : def.label;
}
