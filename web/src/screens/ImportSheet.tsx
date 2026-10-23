// Pick a movie file from the server's import folder, grouped by the folder it's in,
// like the release picker's film cans (ui-design.md §4.1).
import { useEffect, useMemo, useState } from "react";
import { ApiError, api, type ImportFile } from "../api";
import { ErrorDetails } from "../components/ErrorDetails";
import { Sheet } from "../components/Sheet";
import { fmtBytes } from "../format";
import { navigate } from "../router";

const folderOf = (path: string) => (path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "");

/** Files at the top of the folder first, then folders in the order of their newest
 * file (the list arrives newest first). */
function groupByFolder(files: ImportFile[]): [string, ImportFile[]][] {
  const groups = new Map<string, ImportFile[]>([["", []]]);
  for (const f of files) {
    const k = folderOf(f.path);
    groups.set(k, [...(groups.get(k) ?? []), f]);
  }
  return [...groups].filter(([, g]) => g.length);
}

const fmtDay = (mtime: number) =>
  new Date(mtime * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });

export function ImportSheet({
  busy,
  error: addError,
  onPick,
  onClose,
}: {
  busy: boolean;
  error: ApiError | null;
  onPick: (path: string) => void;
  onClose: () => void;
}) {
  const [files, setFiles] = useState<ImportFile[] | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [q, setQ] = useState("");
  const [picked, setPicked] = useState<string | null>(null);

  useEffect(() => {
    api.imports().then(
      (r) => setFiles(r.files),
      (e) => setError(e as ApiError),
    );
  }, []);

  const shown = useMemo(() => {
    const term = q.trim().toLowerCase();
    return (files ?? []).filter((f) => !term || f.path.toLowerCase().includes(term));
  }, [files, q]);

  return (
    <Sheet label="Pick a movie file from the server" onClose={onClose}>
      <h2>Pick a movie file from the server.</h2>
      {files && files.length > 0 && (
        <>
          <label htmlFor="import-q" className="visually-hidden">
            Search the import folder
          </label>
          <input
            id="import-q"
            className="field"
            type="search"
            placeholder="Search the import folder"
            autoComplete="off"
            autoCorrect="off"
            spellCheck={false}
            enterKeyHint="search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
        </>
      )}
      <p className="note num" aria-live="polite">
        {!files && !error
          ? "Looking…"
          : files && files.length > 0
            ? shown.length < files.length
              ? `${shown.length.toLocaleString()} of ${files.length.toLocaleString()} movie files`
              : `${files.length.toLocaleString()} movie file${files.length === 1 ? "" : "s"}`
            : ""}
      </p>
      {error && (
        <div role="alert">
          <p className="message danger">{error.message}</p>
          <ErrorDetails detail={error.detail} />
        </div>
      )}
      {files && files.length === 0 && (
        <p className="fog measure">
          The import folder has no movie files. Copy one into it on the server, then open this list again.
        </p>
      )}
      {files && files.length > 0 && !shown.length && <p className="fog">No files in the import folder match that.</p>}
      {shown.length > 0 && (
        <div className="finds" role="radiogroup" aria-label="Movie files">
          {groupByFolder(shown).map(([folder, group]) => (
            <section className="film-group" key={folder}>
              {folder && <h3 className="folder">{folder}</h3>}
              <ul className="choices">
                {group.map((f) => (
                  <FileRow
                    key={f.path}
                    file={f}
                    checked={picked === f.path}
                    onChange={() => setPicked(f.path)}
                  />
                ))}
              </ul>
            </section>
          ))}
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
          onClick={() => picked && onPick(picked)}
        >
          {busy ? "Adding…" : "Add movie"}
        </button>
      </div>
    </Sheet>
  );
}

function FileRow({ file: f, checked, onChange }: { file: ImportFile; checked: boolean; onChange: () => void }) {
  const done = !!f.movie_id;
  return (
    <li className="import-file">
      <label className={`radio release${done ? " disabled" : ""}`}>
        <input type="radio" name="import" checked={checked} disabled={done} onChange={onChange} />
        <span className="release-body">
          <span className="release-name file-name" title={f.path}>
            {f.name}
          </span>
          <span className="release-facts file-facts num">
            <span>{fmtBytes(f.bytes)}</span>
            <span>{done ? "Already added" : fmtDay(f.mtime)}</span>
          </span>
        </span>
      </label>
      {done && (
        <button type="button" className="btn btn-text" onClick={() => navigate(`/m/${f.movie_id}`)}>
          Open it
        </button>
      )}
    </li>
  );
}
