// The server's internals only ever appear here, folded away (ui-design.md §7).
export function ErrorDetails({ detail }: { detail?: string | null }) {
  if (!detail) return null;
  return (
    <details className="tech">
      <summary>Details</summary>
      <pre>{detail}</pre>
    </details>
  );
}
