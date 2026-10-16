# FrameHound

A self-hosted webapp that downloads a movie, finds a moment in it by description,
picture, or a line of dialogue, and cuts a clip. See [docs/design.md](docs/design.md)
(system) and [docs/ui-design.md](docs/ui-design.md) (UI).

**Status: milestone M1 (skeleton).** Implemented:

- Docker Compose stack: `web` (nginx + React SPA), `api` (FastAPI), `redis`,
  `worker-io` and `worker-index` (RQ).
- Ingest from plain HTTP(S) and archive.org, including the collection picker,
  with an SSRF guard, resumable downloads, and checksum checks.
- ffprobe validation, content-hash deduplication, and a 720p proxy for playback.
- `LocalIndexBackend` running the `framehound_indexer` pipeline on this host,
  transcript only: faster-whisper with word timestamps, plus bge-small sentence
  embeddings.
- Dialogue search: fuzzy lexical and semantic matching, word-accurate bounds,
  and the heat-map series.
- Accurate (re-encoded) and quick (stream copy) MP4 clips.
- The UI: library, indexing progress over SSE, search and timeline strip, and
  the export sheet, at iPhone and desktop sizes.

Not yet: Colab indexing, shots, visual and face search (M2, M3); torrents, GIF
and subtitle export, the reranker, eviction, and token auth (M4). The Describe it
and Show a picture modes are shown but unavailable until a movie has a visual index.

## Run it

```sh
cp .env.example .env        # optional; every setting has a default
docker compose up -d --build
open http://localhost:8080  # bound to 127.0.0.1 only
```

Movies, indexes, and model weights live in the `fh-data` volume. The first index
downloads Whisper large-v3 (about 3 GB). On a CPU, large-v3 takes roughly as long
as the film, so for trying things out set `FH_MODELS_LOCK=models.tiny.lock` in
`.env`. An index can only be searched by an api running the same lock.

## Develop

```sh
uv sync --all-packages --all-extras
DATA_DIR=./data FH_MODELS_DIR=./data/models FH_MODELS_LOCK=models.tiny.lock \
  uv run framehound-api                 # :8000; with no REDIS_URL, jobs run in threads
cd web && pnpm install && pnpm dev      # :5173, proxies /api to :8000
```

Tests and lint:

```sh
uv run pytest
uv run ruff check server indexer
cd web && pnpm typecheck
```

Indexing uses CUDA when ctranslate2 finds a GPU, and falls
back to the CPU if the CUDA libraries aren't usable.

## Layout

| Path | What |
|---|---|
| `indexer/` | `framehound_indexer`: the pipeline (`python -m framehound_indexer job.json`), query encoders, `models.lock` |
| `server/` | `framehound`: api, resolver, downloads, ffmpeg, jobs, index backends, search |
| `web/` | React + Vite SPA |
| `docker/` | Dockerfiles and the nginx config |
