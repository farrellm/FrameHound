// One URL, two states: Indexing until the movie is ready, then Search in place.
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, type Movie } from "../api";
import { useAnnouncer } from "../components/Announcer";
import { fmtDuration } from "../format";
import { linkProps, navigate } from "../router";
import { Indexing } from "./Indexing";
import { Search } from "./Search";

export function MovieScreen({ id }: { id: string }) {
  const [movie, setMovie] = useState<Movie | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const { toast } = useAnnouncer();
  const wasReady = useRef<boolean | null>(null);
  const readyWhileHidden = useRef(false);

  const load = useCallback(async () => {
    try {
      const m = await api.movie(id);
      if (m.status === "merged" && m.merged_into) {
        navigate(`/m/${m.merged_into}`, { replace: true });
        return;
      }
      setMovie(m);
      setError(null);
    } catch (e) {
      setError(e as ApiError);
    }
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  // Turning ready while the page was in the background earns a toast on return.
  useEffect(() => {
    if (!movie) return;
    const ready = movie.status === "ready";
    if (wasReady.current === false && ready) {
      if (document.visibilityState === "hidden") readyWhileHidden.current = true;
      else toast("Ready to search");
    }
    wasReady.current = ready;
  }, [movie, toast]);

  useEffect(() => {
    const onVis = () => {
      if (document.visibilityState !== "visible") return;
      load();
      if (readyWhileHidden.current) {
        readyWhileHidden.current = false;
        toast("Ready to search");
      }
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, [load, toast]);

  useEffect(() => {
    if (movie?.title) document.title = `${movie.title} · FrameHound`;
    return () => {
      document.title = "FrameHound";
    };
  }, [movie?.title]);

  if (error) {
    return (
      <main className="page">
        <Back />
        <p className="message danger" role="alert">
          {error.status === 404 ? "This movie isn't in your library any more." : error.message}
        </p>
      </main>
    );
  }
  if (!movie) {
    return (
      <main className="page" aria-busy="true">
        <Back />
      </main>
    );
  }
  if (movie.status === "ready") return <Search movie={movie} onChanged={load} />;
  return (
    <main className="page">
      <Back />
      <h1 className="movie-title">{movie.title || "Untitled movie"}</h1>
      {movie.duration_s ? <p className="fog num">{fmtDuration(movie.duration_s)}</p> : null}
      <Indexing movie={movie} onChanged={load} />
    </main>
  );
}

export function Back() {
  return (
    <nav className="topbar">
      <a className="back" {...linkProps("/")}>
        ‹ Library
      </a>
    </nav>
  );
}
