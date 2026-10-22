// Two routes (ui-design.md §3): "/" and "/m/{id}", with query state for search.
import { useEffect, useState } from "react";

const listeners = new Set<() => void>();

export function navigate(to: string, opts: { replace?: boolean } = {}) {
  if (opts.replace) history.replaceState(null, "", to);
  else history.pushState(null, "", to);
  listeners.forEach((l) => l());
  if (!opts.replace) window.scrollTo(0, 0);
}

/** Update query params in place (no new history entry). Best effort: Safari throws once
 * replaceState is called more than 100 times in a short window, and a stale URL beats a crash. */
export function setQuery(params: Record<string, string | null>) {
  const url = new URL(location.href);
  for (const [k, v] of Object.entries(params)) {
    if (v == null || v === "") url.searchParams.delete(k);
    else url.searchParams.set(k, v);
  }
  try {
    history.replaceState(null, "", url.pathname + url.search);
  } catch {
    // throttled; the next write catches up
  }
}

export function useLocation() {
  const [, force] = useState(0);
  useEffect(() => {
    const on = () => force((n) => n + 1);
    listeners.add(on);
    window.addEventListener("popstate", on);
    return () => {
      listeners.delete(on);
      window.removeEventListener("popstate", on);
    };
  }, []);
  return { path: location.pathname, query: new URLSearchParams(location.search) };
}

export function linkProps(to: string) {
  return {
    href: to,
    onClick: (e: React.MouseEvent) => {
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
      e.preventDefault();
      navigate(to);
    },
  };
}
