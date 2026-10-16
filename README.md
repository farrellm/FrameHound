# FrameHound

A self-hosted webapp that downloads a movie, finds a moment in it by description,
picture, or a line of dialogue, and cuts a clip. See [docs/design.md](docs/design.md)
(system) and [docs/ui-design.md](docs/ui-design.md) (UI).

**Status: milestone M2 (Colab + visual).** Implemented:

- Docker Compose stack: `web` (nginx + React SPA), `api` (FastAPI), `redis`,
  `worker-io` and `worker-index` (RQ).
- Ingest from plain HTTP(S) and archive.org, including the collection picker,
  with an SSRF guard, resumable downloads, and checksum checks.
- ffprobe validation, content-hash deduplication, and a 720p proxy for playback.
- `ColabIndexBackend` (the default): a Colab GPU per movie through the `colab`
  CLI, with GPU fallback, a compute-unit budget, a detached run polled every
  10 s, retries on a fresh VM, re-attach after a worker restart, and a reconciler
  that stops orphaned `fh-*` sessions. `LocalIndexBackend` runs the same
  pipeline on this host.
- The pipeline: shot detection, SigLIP 2 frame embeddings at 2 fps with a
  thumbnail per shot, and (in parallel) faster-whisper with word timestamps plus
  bge-small sentence embeddings.
- Describe it (text) search: prompt-ensembled SigLIP 2 scores turned into
  shot-snapped candidate segments and a heat-map.
- Dialogue search: fuzzy lexical and semantic matching, word-accurate bounds,
  and the heat-map series.
- Accurate (re-encoded) and quick (stream copy) MP4 clips.
- The UI: library, indexing progress over SSE, search and a timeline strip with
  real shot thumbnails, and the export sheet, at iPhone and desktop sizes.

Not yet: faces and Show a picture search (M3); torrents, GIF and subtitle
export, the reranker, eviction, and token auth (M4). Movies indexed by M1 need a
re-index (the models lock changed) before Describe it works.

## Run it

```sh
cp .env.example .env        # optional; every setting has a default
docker compose up -d --build
open http://localhost:8080  # bound to 127.0.0.1 only
```

Indexing runs on Colab by default. Sign the `colab` CLI in on the host first
(`colab usage` prompts the first time); compose mounts `~/.config/colab-cli`
into the containers. Set `INDEX_BACKEND=local` to index on this machine.

Movies, indexes, and model weights live in the `fh-data` volume. The first index
downloads Whisper large-v3 (about 3 GB). On a CPU, large-v3 takes roughly as long
as the film, so for trying things out set `FH_MODELS_LOCK=models.tiny.lock` in
`.env`. An index can only be searched by an api running the same lock.

## Develop

```sh
uv sync --all-packages --all-extras
DATA_DIR=./data FH_MODELS_DIR=./data/models FH_MODELS_LOCK=models.tiny.lock \
  INDEX_BACKEND=local uv run framehound-api                 # :8000; with no REDIS_URL, jobs run in threads
cd web && pnpm install && pnpm dev      # :5173, proxies /api to :8000
```

Tests and lint:

```sh
uv run pytest                          # Colab is faked; no GPU or account needed
FH_SLOW_TESTS=1 uv run pytest indexer   # the real pipeline with the tiny models
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
