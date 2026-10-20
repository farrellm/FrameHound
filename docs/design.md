# FrameHound — Design Doc

| | |
|---|---|
| **Status** | Draft |
| **Author** | farrellm |
| **Last updated** | 2026-10-16 |

## 1. Overview

FrameHound is a self-hosted webapp that does three things in order:

1. **Downloads** a movie from an HTTP(S) URL, an archive.org item, or a BitTorrent magnet/.torrent.
2. **Searches** the movie for a *target*, which can be:
   - a **text description** ("a man in a red hat climbing a ladder"),
   - a **reference image** of an object or a person's face, or
   - a line of **spoken dialogue**.
3. **Produces a clip** (MP4, GIF, or MP4 with burned-in subtitles) of the matching segment.

The design rests on one split. The machine-learning work that has to see every frame is done **once per movie, on a rented Colab GPU** driven by the [Colab CLI](https://github.com/googlecolab/google-colab-cli). The resulting index is small and gets copied back to the host. Everything after that runs on the host's CPU: searching (which embeds only the query) and clipping (ffmpeg on the original file). The host needs no GPU.

```
 Ingest (host) ──► Index (Colab GPU, once) ──► Search (host CPU, many) ──► Clip (host)
```

### Goals
- Index a 2-hour feature in about 15–20 minutes end to end, with GPU time limited to the indexing step.
- Return search results in under 1 s on a CPU-only host once the movie is indexed.
- Support all three query types against the same index.
- Keep a person in the loop: the system proposes ranked candidate segments, and the user confirms and adjusts them before anything is cut.
- Keep Colab spend bounded and visible. A Colab session must never be left running after its job ends.

### Non-goals
- Multi-tenant SaaS, horizontal scaling, and user accounts.
- Live streams, or videos longer than about 4 hours (a configurable limit).
- Hosting or redistributing content. Users are responsible for the rights to what they download. The UI defaults to, and the docs point to, public-domain sources such as the Internet Archive.

## 2. User flow

The screens, visual language, and iPhone behavior are specified in [ui-design.md](ui-design.md).

1. **Add a movie.** The user pastes a URL, an archive.org link, or a magnet link, or uploads a `.torrent` file. If the archive.org link points to a *collection* rather than a single item, the UI lists the collection's items for the user to pick from. The example in the brief, `archive.org/details/bittorrent`, is a collection named "Archive Torrents" and contains no video file of its own.
2. **Watch progress.** A single progress bar steps through: downloading → probing → provisioning GPU → staging media → indexing (shots / frames / faces / transcript) → importing → ready.
3. **Search.** The user picks a query type and enters text, drops in an image, or types a quote.
4. **Review candidates.** Results appear as a ranked list, each with a thumbnail strip, a time range, a score, and the matching reason (visual, face, or dialogue). Clicking a candidate seeks the player to it.
5. **Refine.** The user drags the in/out handles on a timeline that shows shot boundaries and the score as a heat-map.
6. **Export.** The user chooses MP4, GIF, or MP4 with burned-in subtitles, and fast or accurate cutting, then downloads the clip.

## 3. Architecture

```
┌──────────────────────── Self-hosted machine (docker compose) ───────────────────────┐
│                                                                                     │
│  ┌──────────┐   REST/SSE   ┌──────────┐      ┌───────┐                              │
│  │  web     │◄────────────►│  api     │◄────►│ redis │  (RQ queues: io, index)      │
│  │ React SPA│              │ FastAPI  │      └───┬───┘                              │
│  └──────────┘              │ + search │          │                                  │
│                            │ encoders │   ┌──────┴───────┐   ┌──────────────────┐   │
│                            └────┬─────┘   │ worker-io    │   │ worker-index     │   │
│                                 │         │ download,    │   │ ColabIndexBackend│   │
│                                 │         │ probe, proxy,│   │ (shells out to   │   │
│                                 │         │ clip (ffmpeg)│   │  `colab` CLI)    │   │
│                                 │         └──────┬───────┘   └────────┬─────────┘   │
│                       ┌─────────┴─────────────────┴────────────────────┤             │
│                       │  /data  (SQLite WAL + movie files + indexes)   │             │
│                       └────────────────────────────────────────────────┘             │
└───────────────────────────────────────────────────────────────────────┼─────────────┘
                                                                        │ colab new / install /
                                                                        │ upload / exec / download / stop
                                                         ┌──────────────▼───────────────┐
                                                         │ Colab VM (L4/T4/A100 GPU)    │
                                                         │  framehound_indexer          │
                                                         │  /content/fh/{movie_id}/...  │
                                                         └──────────────────────────────┘
```

### Components

| Service | Responsibility |
|---|---|
| `web` | React + Vite SPA, with a video player (native `<video>` playing an HTTP-range-served proxy), a timeline, and the search UI. |
| `api` | FastAPI app. Provides REST plus SSE for job progress. Search runs **in process**: the query encoders are loaded once at startup and the per-movie FAISS indexes are memory-mapped. |
| `redis` | Backs the RQ job queues. |
| `worker-io` | Downloads (HTTP, archive.org, libtorrent), runs `ffprobe`, transcodes the proxy, and cuts clips. A pool of `MAX_DOWNLOADS` (default 3) RQ workers, so that many jobs run at once; each torrent claims its own listen port (`TORRENT_PORT` + slot). |
| `worker-index` | Runs `IndexBackend` jobs. It is a thin orchestrator that does no ML itself. `ColabIndexBackend` is the default and `LocalIndexBackend` is the fallback. |

### Shared indexer package

`framehound_indexer/` is a standalone Python package. The host ships it to a Colab VM as a reproducible tarball (fixed mtimes and modes, named by its hash) that is unpacked onto `PYTHONPATH`, so the image needs no build step and a re-upload is skipped when nothing changed. It contains:

- the indexing pipeline (the entry point is `python -m framehound_indexer job.json`), and
- the **query encoders**: `encode_text`, `encode_image`, `encode_face`, and `encode_sentence`. The `api` imports these. Every model runs on ONNX Runtime (CUDA on the VM, CPU on the host), so the `api` needs no torch, and both sides run the same exported weights.

All models are pinned in one `models.lock` file (model id, revision or commit hash, and preprocessing parameters). A query embedding is only comparable to the index embeddings if both come from the *exact* same model and preprocessing. Each index bundle therefore records the hash of its `models.lock`, and `api` refuses to search a bundle whose hash doesn't match its own.

### Storage layout

```
/data/
  framehound.sqlite
  models/                      # HF / insightface weight cache (volume)
  movies/{movie_id}/
    source.{mp4,mkv,...}       # original, used for clipping
    proxy.mp4                  # 720p H.264, for browser playback + Colab upload
    index/
      manifest.json            # models.lock hash, fps, counts, durations
      clip.faiss               # frame embeddings (IndexFlatIP)
      faces.faiss              # face embeddings
      sentences.faiss          # transcript segment embeddings
      thumbs/{shot_idx}.jpg
    clips/{clip_id}.{mp4,gif}
```

## 4. Stage 1 — Ingest (host, `worker-io`)

### Source resolution

`SourceResolver.resolve(input) -> ResolvedSource{kind, url|torrent, filename, expected_size, sha?}`

- **archive.org.** Match `archive.org/details/{id}` or `/download/{id}/...`, then call `GET https://archive.org/metadata/{id}`.
  - If `metadata.mediatype == "collection"`, list the collection's members with `GET https://archive.org/advancedsearch.php?q=collection:{id} AND mediatype:movies&fl[]=identifier,title&output=json` and return them for the user to choose from.
  - Otherwise, choose a video file by format preference: `h.264` / `h.264 HD` / `MPEG4` > `512Kb MPEG4` > `Ogg Video` > `Matroska`. Prefer `source: original` over derivatives, and among equally preferred files take the larger one. The download URL is `https://archive.org/download/{id}/{name}`. The item's `md5`/`sha1` from the metadata is kept so the download can be verified.
  - Every item also exposes `{id}_archive.torrent`, but HTTP is used for archive.org because it is simpler and fast enough.
- **Plain HTTP(S).** Downloaded with httpx in streaming mode. Interrupted downloads resume using `Range` requests. The resolver checks `Content-Type`/`Content-Length` first.
- **Magnet / .torrent.** Downloaded with libtorrent (`server/framehound/torrent.py`) on the **host**:
  - A magnet's `source_ref` is `btih:{v1 info hash}` (hex or base32 in the link), so the same torrent pasted twice, or as a magnet and as a file, is a duplicate. An uploaded `.torrent` (at most `MAX_TORRENT_KB`) is parsed at once and kept as `source.torrent`, so a re-download needs nothing else.
  - Only the largest file with a video extension is downloaded (all other files get priority 0: up front for a `.torrent`, as soon as a magnet's metadata arrives). It lands in `torrent/` and moves to `source.{ext}` when complete; an interrupted download resumes there after libtorrent rechecks the pieces.
  - Seeding is configurable (`SEED_RATIO`, default 0 = stop when complete; at most `SEED_MAX_S`).
  - By default a torrent waits indefinitely for peers; the user can cancel it. An optional stall timeout makes the job fail with `torrent_stalled` when there are no peers, or no progress, for `TORRENT_STALL_S` seconds (default 0 = off), including while waiting for a magnet's metadata. The Downloading stage shows the peer count, or "Waiting for peers".
  - The SSRF rule holds for peers too: an IP filter refuses loopback, RFC1918, link-local, CGNAT and multicast peers and trackers, and libtorrent's `ssrf_mitigation` is on.
- **Finding a torrent by name (Bitmagnet).** This is optional and on when `BITMAGNET_URL` is set. [Bitmagnet](https://bitmagnet.io) (`docker/bitmagnet`) crawls the DHT and matches torrents to TMDB films.
  - `server/framehound/bitmagnet.py` sends one GraphQL `torrentContent.search` (content type `movie`, relevance order, 40 results).
  - It groups the releases under the film they matched, in the order of each film's best-ranked release, with releases sorted by seeders. Torrents with no film match come last.
  - Releases over `MAX_BYTES` are flagged `too_big`.
  - The user's pick is added as an ordinary magnet (`POST /movies` with the film's title), so everything above applies unchanged.
  - The URL is set by the operator, so the call skips the SSRF guard. In compose, `docker-compose.bitmagnet.yml` joins the api to Bitmagnet's network, and Bitmagnet's unauthenticated port stays on localhost.

### Validation and normalization

- `ffprobe -show_format -show_streams` must find a video stream and a duration of at most `MAX_DURATION`. The file must be no larger than `MAX_BYTES`.
- **Deduplication.** The host computes the SHA-256 of `source.*` as the movie's content key. If a ready index with the same key and the same `models.lock` hash already exists, the job skips straight to *ready*.
- **Proxy transcode.** `ffmpeg -i source -vf scale=-2:720 -c:v libx264 -preset veryfast -crf 26 -c:a aac -b:a 96k -movflags +faststart proxy.mp4`. A 2-hour film comes out at roughly 1–1.5 GB. The proxy serves two purposes: browser playback, and the upload to Colab when one is needed (see below). With `TRANSCODE_HWACCEL=nvenc` (`docker-compose.gpu.yml`) it is decoded, scaled and encoded on an NVIDIA GPU instead (`scale_cuda`, `h264_nvenc -cq 33`, about the same size and quality); if that fails for any reason, the proxy is redone with libx264.

### Getting media onto the Colab VM

| Source kind | How the VM gets the bytes |
|---|---|
| archive.org / public HTTP(S) | **The VM downloads the file itself** from the source URL. Datacenter bandwidth is much faster than uploading from home. The VM checks the SHA-256 against the host's value before indexing. |
| BitTorrent, user upload, auth-gated URL | The host runs `colab upload proxy.mp4 /content/fh/{id}/media.mp4`. **Torrents never run on the VM**, because Colab's usage policy forbids peer-to-peer file sharing. |

Indexing works on the proxy or on the source, whichever the VM has. Detection quality at 720p is sufficient. Clip *output* always comes from the original `source.*`.

> Staging through Google Drive or GCS was considered. `colab drivemount` and `colab auth` both require an interactive browser step, so they are not suitable for an unattended worker. `colab upload` is the only staging path for v1.

### Security

- **SSRF guard.** Before connecting, the host resolves DNS and rejects loopback, RFC1918, link-local, CGNAT, and metadata addresses (`169.254.169.254`). It connects to the vetted IP, and re-validates on every redirect (at most 5 redirects).
- **Media handling.** ffmpeg/ffprobe run as an unprivileged user under `ulimit` CPU, memory, and file-size limits, with a protocol whitelist (`-protocol_whitelist file`). Container and codec are allowlisted.
- **Filenames.** Every filename is derived by the server. User-supplied names are never used as paths.

## 5. Stage 2 — Index (Colab GPU, `worker-index`)

### Orchestration: `ColabIndexBackend`

The backend is a Python class that shells out to the `colab` CLI (v0.7.x; `--auth` from `COLAB_AUTH`, default the CLI's own oauth2, or `adc`). All calls go through one adapter (`server/framehound/colab.py`). Every command passes an explicit `-s fh-{movie_id}-{attempt}`. Omitting `-s` would auto-generate a random name and make later commands ambiguous.

```
 1. colab usage                                  # check compute-unit balance vs. budget
 2. colab new -s S --gpu $GPU                    # GPU from config, default L4
      └─ on HTTP 400 (no entitlement/quota): fall back along GPU_FALLBACK = [L4, T4]
 3. colab exec -f util.py (mkdir)                # the contents API won't create directories
    colab upload framehound_indexer-{hash}.tar; colab exec -f util.py (unpack)
    colab install -s S -r requirements-colab.txt
    colab upload -s S job.json /content/fh/{id}/job.json
    [colab upload proxy.mp4 in 64 MB parts; util.py joins]   # only for non-public sources
 4. colab exec -s S -f launch.py --env FH_JOB=… --env FH_PKG=…
      # launch.py: Popen(run.py JOB, start_new_session=True, → run.log); prints PID; returns.
      # run.py swaps in onnxruntime-gpu if needed, puts the image's CUDA 12 pip libraries
      # on LD_LIBRARY_PATH, then execs python -m framehound_indexer JOB
 5. loop every 10 s:
      colab exec -s S -f poll.py --env FH_OFFSET=n   # new progress.jsonl lines + PID liveness
      → each line becomes a job progress event → SSE to the browser
 6. on {"stage":"done"}:
      colab exec -f util.py (split) + colab download of each 64 MB part → index.tar.zst
 7. finally: colab log -s S -o /data/logs/{session}.jsonl   (on failure only)
             colab stop -s S
```

`colab upload`/`download` send a whole file as one base64 JSON request, so anything over 64 MB moves in parts. Exit codes are unreliable (`exec` exits 0 when the code raises), so results are judged from marker lines our scripts print (`@@fh-launched`, `@@fh-poll`, `@@fh-util`) and from the CLI's own messages.

The reasons behind the main choices:

- **Detached process with polling, not one long `colab exec`.** `colab exec` has a default timeout of 30 s and holds a websocket open. A 15-minute exec would be fragile if the network blipped or the worker restarted. With a detached process, the indexer keeps running on the VM even if the host worker dies. On restart, the worker sees the job's `colab_session` in SQLite and **re-attaches by polling** instead of starting over.
- **Not `colab run`.** It combines new, exec, and stop, but it cannot stage inputs, it gives no incremental progress, and it cannot be re-attached. It is still useful for the manual smoke test (§11).
- **GPU name validation.** An unrecognized `--gpu` value silently falls back to A100, so the backend validates the configured GPU against the supported set before calling `colab new`.
- **CUDA on the image.** As of 2026-10 the Colab image's system CUDA is 13 (torch cu130), while ctranslate2 and onnxruntime-gpu link CUDA 12. The image also carries CUDA 12 cuBLAS/cuDNN as pip wheels; `run.py` puts their `nvidia/*/lib` directories on `LD_LIBRARY_PATH` before the pipeline starts. Without that, Whisper silently falls back to the CPU.

### Pipeline on the VM (`framehound_indexer`)

Each stage writes its output and a `.done` marker under `/content/fh/{id}/work/`. A crashed process that is relaunched in the same session resumes at the first stage without a marker. Every stage appends lines like `{"stage": "...", "pct": 0.42, "eta_s": 310}` to `progress.jsonl`.

| # | Stage | Implementation | Output |
|---|---|---|---|
| 0 | fetch | httpx download (public sources), SHA-256 verify | `media.*` |
| 1 | shots | numpy port of PySceneDetect's `AdaptiveDetector` (HSV content change vs. its neighbours, ratio ≥ 3) over every frame decoded at 96 px by ffmpeg (NVDEC when available) | `shots.parquet (idx, start, end)` |
| 2–3 | visual | one ffmpeg decode at **2 fps** (frames, visual and thumbs fused): squash to 384², **SigLIP 2 so400m** vision tower (ONNX fp16, CUDA), L2-normalize; each shot's thumbnail is the sampled frame nearest its middle | `clip.faiss` (IndexFlatIP), `frames.parquet (row, t, shot_idx)`, `thumbs/` |
| 4 | faces | **InsightFace** `buffalo_l` on ONNX Runtime: SCRFD detect over the same 2 fps frames decoded at 720p (det ≥ 0.6, min face 40 px) + ArcFace 512-d | `faces.faiss`, `faces.parquet (row, frame_row, x, y, w, h, det_score)` (box as fractions of the frame) |
| 5 | audio | ffmpeg → 16 kHz mono; **faster-whisper large-v3**, `word_timestamps=True`, VAD filter | `transcript.parquet (seg, start, end, text, words_json)` |
| 6 | sentences | **bge-small-en-v1.5** over transcript segments (plus 3-segment sliding windows) | `sentences.faiss` |
| 8 | bundle | write `manifest.json`, `tar` + `zstd -19` | `index.tar.zst` |

The visual stage decodes each sampled frame once for the embeddings and the thumbnails. The faces stage decodes the same 2 fps samples again at a higher resolution (720p rather than 384²), so its frame `k` is `frames.parquet` row `k`. The audio lane (audio, sentences) runs in a thread alongside the visual lane (shots, visual, faces). On a GPU, the GPU-bound stages (audio, visual, faces) take turns: each one saturates an L4 by itself, so sharing only slowed both down and made their ETAs wrong. Shot detection runs on the CPU and NVDEC, so it still overlaps Whisper.

**Visual model.** SigLIP v1 so400m has no ONNX export, so the lock pins `onnx-community/siglip2-so400m-patch14-384-ONNX` (SigLIP 2, same size, better retrieval): the VM runs `vision_model_fp16.onnx`, the host `text_model.onnx` (about 2.2 GB of RAM; a query takes about 0.6 s with the 3-prompt ensemble). Text is lowercased and padded to 64 tokens, as SigLIP was trained. `models.tiny.lock` uses SigLIP base (224 px) for development.

**Face models.** The lock pins `public-data/insightface`'s copy of the `buffalo_l` pack: `det_10g.onnx` (SCRFD, the detector with five landmarks) and `w600k_r50.onnx` (ArcFace). Both run directly on ONNX Runtime, without the `insightface` package (which needs a compiler); `framehound_indexer/faces.py` ports its SCRFD decoding, NMS and five-point alignment, with PIL's affine transform in place of OpenCV. The index and the query encoder share that code. Boxes are stored as fractions of the frame, so a face crop applies to the source at any resolution. InsightFace's pretrained models are licensed for non-commercial research use.

**Size and time estimates for a 2-hour film.** 2 fps gives about 14.4k frames.

- **Embeddings:**
  - SigLIP: about 14.4k × 1152 × 4 B ≈ 66 MB. Stored as fp16 that is about 33 MB.
  - Faces: typically 10–40k × 512 × 2 B ≈ 10–40 MB.
- **Compute on an L4** (measured on a 10-min clip and a 2h11m film, Oct 2026):
  - Whisper large-v3 (fp16, batched): about 10–12 min (49–60 s per 10 min of audio). Only shot detection (3–5 min) overlaps it.
  - Decode plus SigLIP: about 6 min at 40 frames/s, with the vision tower in PyTorch with SDPA attention. The ONNX export manages 22–25 frames/s. The 4.3 GB PyTorch checkpoint downloads in the background while Whisper runs.
  - Faces: about 4 min (estimate).
- **Overhead:** provisioning plus installation adds about 2–3 min.
- **Total:** about **22–25 min** of wall time and a bundle of **about 60–120 MB**.

### `LocalIndexBackend`

This backend runs the same package in a subprocess on the host, on a CPU or a local GPU, behind the same `IndexBackend` interface (`start`, `poll`, `fetch`, `cancel`, `cleanup`). It is selected with `INDEX_BACKEND=colab|local`. Uses:

- development and CI, with tiny models via `models.lock` overrides,
- a fallback when Colab is unavailable or over budget, and
- privacy-sensitive media.

On a CPU it is slow: about 1–2 h for a 2-hour film with the small model variants.

### Import

The host verifies `manifest.json`: the `models.lock` hash matches, and the row counts match the parquet files. It then unpacks into `index/` and loads the parquet tables into SQLite in one transaction, then marks the movie `ready`. The FAISS files stay on disk and are memory-mapped by `api`.

### Robustness

- **Leak prevention.**
  - `colab stop` runs in a `finally` block.
  - On startup, and every 15 min, a **reconciler** runs `colab sessions` and stops every `fh-*` session that no `running` job owns.
  - Sessions the CLI lists with no local record (`[?]`) are reported, and are stopped if they match the `fh-` prefix.
- **Timeouts.** A job has a hard wall-clock limit of `INDEX_TIMEOUT` (default 60 min). Each stage also has a no-progress watchdog of 10 min.
- **Retries.**
  - When the VM is preempted or disconnects (poll fails three times in a row, or `colab status` reports the session gone), the backend starts a fresh session `…-{attempt+1}`. It retries up to 2 times.
  - Stage checkpoints only survive within one session. A new VM starts from scratch. Making them survive across VMs would mean uploading partial artifacts, which v1 does not do.
- **Credentials.** The host mounts a gcloud ADC volume minted with the scopes `openid`, `cloud-platform`, `userinfo.email`, and `colaboratory`. A health check runs `colab sessions`, and the UI shows a "Colab auth broken" banner when it returns 401 or 403.

## 6. Stage 3 — Search (host CPU, in `api`)

The query encoders load at startup: SigLIP text and image towers, ArcFace (with SCRFD), and bge-small, about 2 GB of RAM in total. A brute-force inner-product search over 14k vectors takes under 5 ms. Encoding dominates: measured on an 8-core host (2026-10), a Describe it query takes about 0.6–1.5 s (the 3-prompt ensemble), a face about 0.6 s, but the so400m vision tower takes about 5 s per picture, so **Show a picture runs about 4–6 s**, past the 1 s goal. The fp32 export is no faster than fp16 on a CPU, and the int8 export disagrees with fp16 (cosine 0.37), so neither helps; a smaller vision model would need a re-index of every movie.

### Query types

**Text** (`type=text`):
1. Prompt ensemble (SigLIP 2 text tower). Embed `"{q}"`, `"a photo of {q}"`, and `"a film still of {q}"`, then average and normalize.
2. Score every frame by cosine similarity to the averaged embedding.
3. If the text reads like a quote (it has quotation marks, or the user toggled "also search dialogue"), also run the dialogue path and merge the results.

**Image** (`type=image`, multipart with an `image` file of at most `MAX_QUERY_IMAGE_MB`):
1. Visual score: the SigLIP image embedding against `clip.faiss`, robust-z normalized as below.
2. If SCRFD finds a face in the query image (retrying at 576 down to 320 px, since a close-up can fill the detector), the largest one's ArcFace embedding is scored against `faces.faiss`. Each frame takes the maximum over the faces in it, and similarity ≥ 0.40 (`match` in the lock) counts as a match.
3. Fuse the two: `z = max(z_visual, z_face)`, where a matching frame's `z_face = Z_MIN + FACE_BONUS + (sim − 0.40)·FACE_SCALE` (defaults 2 and 20). Raw ArcFace similarities don't suit the median/MAD normalization (most frames have no face), so this calibrated map stands in for `z_face + face_bonus`: every true face match clears the threshold, and a face match is far more specific than general visual similarity, so it dominates when present.
4. A candidate whose peak frame holds a matching face has `reason: face` and that face's `face_bbox`. The response's `face_found` says whether the picture had a face (null when the movie has no face index), so the UI can explain a looks-only result.

**Dialogue** (`type=dialogue`):
1. **Lexical.** `rapidfuzz.fuzz.partial_ratio` of the normalized query against each segment and each 3-segment window, keeping matches at 85 or above.
2. **Semantic.** bge-small embedding of the query against `sentences.faiss`, which catches paraphrases ("the line about the horse").
3. **Precise bounds.** When a lexical match is found, align the query to the segment's word timestamps to get exact start and end times.

### From frame scores to candidate segments

Each frame's raw score `s(t)` (sampled at 2 Hz) becomes a list of candidate segments:

1. **Normalize per movie.** Compute `z(t) = (s(t) − median) / MAD`. Raw CLIP cosines drift with the query, so a fixed threshold doesn't work.
2. **Smooth.** Apply a moving max over a 1 s window to fill single-frame dropouts.
3. **Threshold.** Keep frames with `z ≥ max(Z_MIN, z_(top 1%))`, where `Z_MIN` defaults to 3.0.
4. **Merge.** Join kept frames into runs, merging across gaps under 1.5 s.
5. **Snap and pad.** Extend each run to the nearest shot boundary if one is within 1 s, and pad by 0.5 s on each side.
6. **Rank.** Score each segment as `peak_z × (1 + 0.1·log(1 + duration_s))`, apply NMS on time overlap, and return the top K (default 10).

Each candidate returns: `start`, `end`, `score`, `peak_t`, `reason` (visual, face, or dialogue), `thumbs[]` (shot thumbnails), `has_face` (whether a face box exists at `peak_t`, which enables face-crop export), and for face hits `face_bbox` at `peak_t`. Dialogue candidates also return `matched_words[]` (`{word, start, end}` from the transcript), so the UI can quote the line. The API also returns the `z(t)` series, max-pooled into `heat_bins` buckets (default 512, range 64–4096), so the UI can draw the heat-map without downloading every 2 Hz sample. Max-pooling keeps short spikes visible at coarse resolution.

### Optional reranker (off by default; deferred past M4)

When `RERANK=on`, the top K candidates' peak thumbnails are sent to a vision-language model (default Claude, `claude-opus-5-5`). The prompt is "Does this frame show: {q}? Answer yes/no with confidence". Candidates the model rejects drop in the ranking. This sends frames to a third party, so it is opt-in and the UI says so.

## 7. Stage 4 — Clip (host, `worker-io`)

Clips are always cut from the original `source.*`, never from the proxy.

| Mode | Command sketch | Trade-off |
|---|---|---|
| **Accurate** (default) | `ffmpeg -ss {start} -i source -t {dur} -c:v libx264 -crf 20 -preset medium -c:a aac -b:a 160k -movflags +faststart out.mp4` | Frame-accurate; re-encodes (seconds for a short clip). |
| **Fast** | `ffmpeg -ss {start} -i source -t {dur} -c copy -avoid_negative_ts make_zero out.mp4` | Instant, lossless; start snaps to previous keyframe. |

Options:

- **GIF.** One ffmpeg run whose filtergraph splits the stream: `fps=12,scale=min(480,iw):-2:flags=lanczos`, then `palettegen` over the clip on one branch and `paletteuse=dither=sierra2_4a` on the other. No audio. Capped at `MAX_GIF_S` (15 s), and always cut accurately.
- **Burn-in subtitles.** The worker writes the transcript words whose midpoint is inside `[start, end]` to `clips/{id}.srt`, with timestamps shifted so the clip starts at 0, and applies `-vf subtitles=filename={id}.srt` (ffmpeg runs in the clips directory, so the server-derived name needs no filtergraph escaping). Cues break at segment boundaries, at pauses over 0.8 s, past two 42-character lines, or past 4 s (`server/framehound/subtitles.py`). MP4 only, needs the dialogue index, and always re-encodes.
- **Face crop.** For face hits, crop to the face's bounding box expanded 2.5× around its centre, grown to portrait 9:16, or to 1:1 when 9:16 doesn't fit the frame's height (shrunk to fit if neither does), kept inside the frame, with even dimensions. It uses `crop=` with a fixed box taken at `peak_t`, which the client sends as `face_bbox`. Cropping re-encodes, so it always cuts in accurate mode. Tracking a moving face is out of scope for v1.
- **Limits.** Clips are at most `MAX_CLIP_S` (default 120 s). Outputs are served with HTTP Range support and garbage-collected after `CLIP_TTL`.

## 8. API

| Method & path | Body / params | Returns |
|---|---|---|
| `POST /movies` | `{source: url \| magnet, title?}` or multipart `.torrent` (`title` overrides the source's own, e.g. a magnet's release name) | `{movie_id, job_id}`, or `{collection, choices: [...]}` for an archive.org collection, or `409 {code: duplicate, movie_id}` |
| `GET /collections/{identifier}` | `?q=` (title search) | `{collection: {identifier, title, total}, choices: [...]}`: the picker sheet's search, top 100 by downloads |
| `GET /torrents/search` | `?q=` (2–200 chars) | `{films: [{title, year, runtime_min, releases: [...]}], unmatched: [...], total, max_bytes}` from Bitmagnet. Each release has `{info_hash, magnet, name, size, files, seeders, leechers, resolution, source, codec, too_big}`. `404 not_configured` without `BITMAGNET_URL`, `502 bitmagnet_unreachable` |
| `GET /movies` / `GET /movies/{id}` | — | movie metadata + status, latest ingest/index job, `capabilities` (`dialogue`, `visual`, `faces`), `models_ok` |
| `DELETE /movies/{id}` | — | `204`; cancels its jobs and removes its files (**Remove movie**) |
| `POST /movies/{id}/retry` | — | `202 {job_id, kind}`: restarts at the first missing step (download/proxy, else indexing). Also **Re-index** |
| `GET /jobs/{id}/events` | — | SSE stream of the job (`state`, `stage`, `pct`, `eta_s`, `error`, per-stage `stages`); event ids are the job's `seq`, so `Last-Event-ID` resumes |
| `POST /jobs/{id}/cancel` | — | `202` (also stops the Colab session) |
| `GET /movies/{id}/index-progress` | — | snapshot of every index stage `{stage, state: pending\|running\|done\|failed, pct, eta_s}` plus `gpu` and `compute_units_used`; the UI calls it when a page resumes, then reopens SSE |
| `GET /movies/{id}/shots` | `?from=&to=` (seconds; default whole movie) | `[{idx, start, end, thumb_url}]` for the strip at the current zoom |
| `POST /movies/{id}/search` | `{type: text\|image\|dialogue, q?, image? (multipart), top_k?, also_dialogue?, heat_bins?}` | `{candidates: [...], heat: [...], face_found?}` (see §6 for candidate fields) |
| `POST /movies/{id}/clips` | `{start, end, mode: accurate\|fast, format: mp4\|gif, subs?, crop?, face_bbox?}` (`subs` is MP4 only; subs, GIF and crop force accurate) (`face_bbox` is the candidate's, required with `crop`) | `{clip_id, job_id}` |
| `GET /clips/{id}` | — | clip metadata, `ready`, and its job |
| `GET /clips/{id}/file` | — | the clip (Range-enabled) |
| `GET /jobs/{id}` | — | job state with per-stage progress |
| `GET /media/{id}/proxy.mp4` | — | proxy for the player (Range-enabled) |
| `GET /health` | — | redis, disk, `colab` auth status, compute-unit balance, `bitmagnet.state` (`ok` \| `not_configured`, from config only); only `{auth_required, signed_in}` without the token |
| `POST /auth` / `DELETE /auth` | `{token}` | sets / clears the sign-in cookie (`401 bad_token` for a wrong token) |

## 9. Data model (SQLite)

```sql
movies(id TEXT PK, sha256 TEXT UNIQUE, title, source_kind, source_ref,
       source_url, source_md5, source_file,
       duration_s REAL, width, height, fps, bytes, models_lock_hash,
       status TEXT /* ingesting|indexing|ready|failed|cancelled|merged|evicted */,
       error TEXT, merged_into TEXT, created_at, last_accessed_at)
jobs(id TEXT PK, movie_id, kind TEXT /* ingest|index|clip */, state TEXT,
     stage TEXT, pct REAL, eta_s REAL, message TEXT, error TEXT, error_code TEXT,
     attempt INT, colab_session TEXT, gpu TEXT, compute_units_used REAL,
     backend_handle TEXT, cancel_requested INT, seq INT /* bumps on every write */,
     started_at, finished_at, heartbeat_at, created_at)
job_stages(job_id, stage, ord INT, state TEXT /* pending|running|done|failed|skipped */,
           pct REAL, eta_s REAL, PRIMARY KEY(job_id, stage))
shots(movie_id, idx INT, start REAL, end REAL, PRIMARY KEY(movie_id, idx))
frames(movie_id, row INT, t REAL, shot_idx INT, PRIMARY KEY(movie_id, row))
faces(movie_id, row INT, frame_row INT, x, y, w, h, det_score REAL,
      PRIMARY KEY(movie_id, row))
transcript_segments(movie_id, seg INT, start REAL, end REAL, text TEXT,
                    words_json TEXT, PRIMARY KEY(movie_id, seg))
queries(id PK, movie_id, type, q, image_sha, created_at, latency_ms)
clips(id PK, movie_id, start REAL, end REAL, mode, format, path, job_id, created_at,
      crop_box TEXT /* JSON [x, y, w, h], fractions of the frame */, subs INT)
```

`transcript_segments` also gets an FTS5 virtual table, which is used for fast exact-phrase prefiltering before the fuzzy match.

**Deduplication and `merged`.** When an ingest's SHA-256 matches a movie that is ready with the current `models.lock`, or still in progress, the new row becomes `status = merged` with `merged_into` set, its files are deleted, and the UI redirects to the existing movie. A failed or stale movie gives up its `sha256` instead, and the new one proceeds.

**Liveness.** A running job heartbeats every 15 s. One with no heartbeat for `STALE_JOB_S` (default 300 s) is marked failed ("The worker stopped…"), so a crashed worker never leaves a job stuck as running.

### Job state machine

```
ingest:  queued → downloading → probing → transcoding_proxy → (dedup hit ? ready : enqueue index)
index:   queued → provisioning_gpu → staging_media → indexing:{fetch|shots|visual|faces|audio|sentences|bundle}
                → downloading_index → importing → ready
         any → failed (error recorded) | cancelled
clip:    queued → cutting → ready | failed
```

## 10. Non-functional requirements

- **Cost control.**
  - Before calling `colab new`, the backend reads `colab usage` and refuses to start if the balance is below `MIN_BALANCE_CU`, or if this month's recorded spend would pass `MONTHLY_BUDGET_CU`.
  - Only one Colab session runs at a time by default (`MAX_COLAB_SESSIONS=1`). Further index jobs wait in the queue.
  - Each job's compute units used (from the before/after difference in `colab usage`) and its GPU type are recorded and shown in the UI.
- **Disk.**
  - Quota by `MAX_DATA_GB` (0 = none) over `movies/`, checked before each download (counting its expected size), after each proxy, and hourly by the api (`server/framehound/evict.py`, under a file lock so the api and worker-io don't both evict for one overage).
  - LRU eviction by `last_accessed_at` (a search touches it), skipping movies with a queued or running job. Each pass runs over every movie before the next starts: clip files, then a ready movie's `source.*` (it stays searchable, plays from the proxy, and shows "re-download to clip"), then the rest of a movie (`status = evicted`, index rows dropped, `source.torrent` kept).
  - **Re-download** (`POST /movies/{id}/retry`) on a movie whose index is current downloads the source again, checks its SHA-256 against the recorded one, and goes straight back to ready without indexing. An evicted movie goes through ingest and indexing.
- **Auth.** v1 binds to localhost by default. When `FH_TOKEN` is set, every request but `/health` and `/auth` needs it, as `Authorization: Bearer` or as the `fh_auth` cookie that `POST /auth` sets (HttpOnly, SameSite=Lax, Secure behind https). `<video>`, `<img>`, EventSource and clip links can't send headers, hence the cookie. It holds an HMAC of the token, so changing `FH_TOKEN` signs every browser out.
- **Observability.**
  - Logs are structured JSON.
  - Every `colab` invocation is logged with its argv, exit code, and duration.
  - For failed index jobs, `colab log -o` is saved next to the job.
- **Config.** Everything above is an env var with a documented default (`.env.example`).

## 11. Testing strategy

- **Unit tests:**
  - segment builder (normalization, merge, snap, NMS) on synthetic score curves,
  - `SourceResolver` with recorded archive.org metadata JSON (item, collection, item with no video) and a mocked advancedsearch,
  - SSRF guard (private IPs, redirect to a private IP, DNS rebinding),
  - SRT time-shifting and cue grouping,
  - torrent ingest against an in-process libtorrent seeder on 127.0.0.1 (magnet and `.torrent`), the stall timeout, and the private-peer filter,
  - eviction order, and re-download with a matching and a mismatched file,
  - token auth (bearer, cookie, rotation).
- **`ColabIndexBackend` tests:** a **fake `colab` executable** (`server/tests/fake_colab.py`) is put on `PATH`. Each session is a directory standing in for the VM, so the real launch/poll/util scripts run and start the fake indexer. It records argv and simulates these scenarios:
  - success,
  - `new` returning 400 (should trigger the GPU fallback),
  - a stalled progress file (should trip the watchdog),
  - a session vanishing mid-run (should retry),
  - an orphaned `fh-*` session (the reconciler should stop it),
  - a worker restart mid-job (should re-attach),
  - budget refusal, no GPU at all, broken auth, and a chunked proxy upload.
- **Integration tests:** `LocalIndexBackend` with tiny models runs on a 2-minute public-domain clip with known ground truth: one text target, one face (a reference still), and one quote. Each query's correct segment must rank in the top 3. The clip output is checked with `ffprobe`: duration within ±1 frame in accurate mode.
- **Manual smoke test** (not in CI because it costs compute units): `colab run --gpu T4 scripts/colab_smoke.py` installs the indexer wheel and indexes the same 2-minute clip. Its bundle must match the local bundle's manifest, with embeddings within cosine 0.999.

## 12. Milestones

| Milestone | Scope |
|---|---|
| **M1 — Skeleton** (done) | Compose stack, HTTP + archive.org ingest (incl. collection picker), proxy, `LocalIndexBackend` with transcript only, dialogue search, accurate MP4 clip. |
| **M2 — Colab + visual** (done) | `ColabIndexBackend` (provision → detached run → poll → fetch → stop, reconciler, budget), shots + SigLIP, text search, timeline/heat-map UI. |
| **M3 — Faces & images** (done) | InsightFace stage, image/face search and fusion, face crop export. |
| **M4 — Torrents & polish** (done) | libtorrent ingest + proxy upload path, GIF and subtitle export, eviction, token auth. |
| **Later** | Optional VLM reranker (§6). |

## 13. Risks and open questions

| Risk | Mitigation |
|---|---|
| Colab GPU not available for the account tier / quota exhausted | GPU fallback chain; queue with clear "waiting for GPU" state; `LocalIndexBackend`. |
| VM preemption or idle reclamation mid-index | Detached process + checkpointing; re-attach on worker restart; bounded retries on a fresh VM. |
| Colab policy (no P2P, no long-running non-interactive abuse) | Torrents only on host; sessions are short, single-purpose, and always stopped. Revisit if Colab terms change. |
| Upload bandwidth for torrent/uploaded media (≈1–1.5 GB proxy) | Lower-bitrate "index proxy" option (480p, ≈500 MB) — index quality is still adequate for search. |
| `colab` CLI is young (v0.7.x); flags/outputs may change | All calls isolated in one adapter class with the fake-binary test suite; pin the CLI version in the image. |
| CLIP misses small/brief objects | Raise fps for short films; optional tiled (2×2 crop) embeddings stage; reranker. |
| Face matching on profiles, low light, makeup | Show confidence; let users add multiple reference images (average embeddings). |
| Whisper hallucinations over music / silence | VAD filter; drop segments with high `no_speech_prob`; fuzzy match tolerates errors. |
| Model drift between index and query encoders | `models.lock` hash in every bundle; refuse mismatches; offer "re-index". |

**Open questions**

1. Should indexes be shareable across installs, for example by publishing a bundle keyed by the source SHA so other users can skip indexing?
2. Is per-frame object detection (for example OWL-ViT) worth adding for "find every appearance of X" style queries, beyond whole-frame embeddings?
3. Should the face path support *naming* (clustering faces into identities across the film) so a user can pick a character from a gallery instead of uploading a photo?
