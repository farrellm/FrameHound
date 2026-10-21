# FrameHound — Models for Describe it

| | |
|---|---|
| **Status** | Research note |
| **Author** | farrellm |
| **Last updated** | 2026-10-20 |
| **Companion to** | [design.md](design.md) §5 (visual stage) and §6 (text query) |

> **Adopted (2026-10-20):** this install runs Qwen3-VL-Embedding-2B (section D), via `models.qwen.lock`. It was chosen for detailed queries over the cheaper steps below, accepting slower queries. SigLIP 2 stays the default lock. Measured costs are in [design.md §5](design.md#5-stage-2--index-colab-gpu-worker-index).

## 1. Question

Describe it misses or mis-ranks moments. Would a different model fix that, and at what cost? This note compares candidates on paper. It changes no code, and it ends with the small evaluation that should settle the choice.

## 2. Today's baseline

- **Model:** SigLIP 2 so400m/14 at 384² (`onnx-community/siglip2-so400m-patch14-384-ONNX`), 1152-d.
- **Index:** one whole-frame embedding per frame at 2 fps, about 14.4k vectors (33 MB fp16) for a 2-hour film.
- **Query:**
  - a 3-prompt ensemble (`"{q}"`, `"a photo of {q}"`, `"a film still of {q}"`);
  - each frame scored with cosine similarity, robust-z, then segments (design §6).
- **Published quality:** COCO text→image R@1 55.8 and Flickr 85.7 [[1]](#sources). That is already near the top of the open dual encoders.

Where it fails, by type of query:

| Failure | Example | Why a whole-frame embedding struggles |
|---|---|---|
| Small or brief object | "a key on the table" | The object is a few percent of 384² pixels and barely moves the global vector. |
| Several attributes or a relation | "a man in a red hat climbing a ladder" | Dual encoders behave like bags of concepts. Red, hat, man and ladder each match somewhere. |
| Action or event | "someone slams a door" | A single frame can't show motion. The index has no temporal signal. |
| Long, specific description | a whole sentence | The text tower truncates at 64 tokens, and SigLIP was trained on short alt-text. |

## 3. Constraints

Each candidate is judged against these:

1. **Text tower on the host CPU.** agda has 8 cores and 15 GB of RAM, and runs ONNX Runtime with no torch. Today a query takes 0.6–1.5 s and the tower holds about 2.2 GB. The goal is under 1 s.
2. **Vision tower once per film on a Colab L4.** so400m takes about 6 min for a 2-hour film in PyTorch. Indexing goals are 15–20 min end to end.
3. **Changing the frame model means re-indexing everything.** The `models.lock` hash changes and every bundle is refused (design §4). Query-side changes avoid this.
4. **Show a picture shares `clip.faiss`.** A new vision tower changes image search too. That is a side benefit if the tower is smaller, since image search takes 4–6 s on the CPU today.
5. **ONNX export.** It must be the same weights on the VM and on the host.
6. **Licence.** A permissive licence is preferred. The face models are already non-commercial, so this is not a hard gate.

## 4. Candidates

The candidates are grouped by what they cost to adopt.

### A. Query side only (no re-index)

| Option | What it does | Cost |
|---|---|---|
| **VLM reranker, Claude** (already specced in design §6, not built) | Sends the top-K peak thumbnails to Claude with "Does this frame show: {q}?" and demotes rejects. | No local hardware. About 1 call per candidate. Frames leave the host, so it is opt-in. |
| **VLM reranker, local: Qwen3-VL-Reranker-2B** [[5]](#sources) | A cross-encoder over (query, image) from Qwen3-VL-2B. Apache-2.0, January 2026. | 2.1B params, so fp16 (about 4.2 GB) doesn't fit the GTX 1050's 4 GB. An int4/GGUF build does. On the CPU, estimate several seconds per frame, so K=10 is too slow for interactive use without the GPU. |
| **Temporal pooling at query time** | Scores a sliding 2–4 s mean of the frame embeddings as well as single frames. This is how PE evaluates video with an image model [[2]](#sources). | Trivial on the existing index. Helps actions a little. Doesn't help small objects. |
| **Prompt ensemble tuning** | More or different templates. Negative prompts such as "a photo" can be subtracted to cancel the generic component. | Free. Small gains. |

A reranker is the only option here that addresses **multi-attribute** and **relation** queries. A generative VLM reads the whole query against the image, which a dual encoder can't do. It can only fix ordering among candidates that recall has already surfaced. It can't find a moment that isn't in the top K.

### B. Drop-in dual encoders (same pipeline, re-index)

All of these keep the frame → vector → FAISS design. Only `models.lock`, the export, and the dimension change.

| Model | Vision / text params | Res | Dim | COCO T→I R@1 | Fine-grained (FG-OVD hard) | Long caption (DCI T→I) | Licence | ONNX |
|---|---|---|---|---|---|---|---|---|
| **SigLIP 2 so400m/14** (today) | ≈0.4B / ≈0.4B | 384 | 1152 | 55.8 [1] | 26.0 (So/16) [4] | 47.1 (So/16) [4] | Apache-2.0 | ✅ onnx-community |
| SigLIP 2 g-opt/16 | ≈1B / ≈0.4B | 384 | 1536 | 56.1 [1] | n.r. | n.r. | Apache-2.0 | export (same arch family) |
| **PE-Core L/14** [[2]](#sources) | 0.32B / 0.31B | 336 | 1024 | 57.1 | n.r. | n.r. | Apache-2.0 | export via open_clip/timm |
| PE-Core G/14 [[2]](#sources) | 1.88B / 0.47B | 448 | 1280 | 58.1 | n.r. | n.r. | Apache-2.0 | export via open_clip/timm |
| **FG-CLIP 2 So/16** [[4]](#sources) | ≈0.4B / ≈0.4B (So400m) | adaptive 128–1024 patches | n.r. | 56.7 | 54.0 | 72.1 | Apache-2.0 tag (with caveats) | export; `trust_remote_code` |
| FG-CLIP 2 L/16 [[4]](#sources) | ≈1.0B total | adaptive | n.r. | 58.6 | 55.5 | 71.6 | as above | as above |
| Meta CLIP 2 H/14 (worldwide) [[3]](#sources) | ≈1.8B total | 224/378 | 1024 | 47.7 [4] | 16.5 [4] | 50.2 [4] | CC-BY-NC | export via open_clip |

n.r. means not reported in the sources read. FG-OVD and DCI are from the FG-CLIP 2 paper's own tables [[4]](#sources). Its SigLIP 2 rows are So/16, a close sibling of the deployed So/14.

What the table shows:

- **On generic retrieval (COCO) every candidate lands within about 3 points of today's model.** Swapping encoders for "better CLIP" buys little and costs a full re-index.
- **The large gaps are on fine-grained and long-caption retrieval.** FG-CLIP 2 roughly doubles SigLIP 2 on FG-OVD hard (26 → 54) and gains 25 points on DCI long-caption T→I [[4]](#sources). Those benchmarks look most like the "man in a red hat climbing a ladder" failure.
- **FG-CLIP 2 So/16 keeps today's costs.** It is the So400m shape with 16-px patches, so the L4 indexing time and the host text-tower size should stay close to today's, and its text limit rises from 64 to 196 tokens. The risk is the export: it ships as custom code, with an adaptive-patch (NaFlex-style) image processor that has to be pinned to a fixed patch count for a reproducible lock.
- **PE-Core L/14 is the cheaper option.** It has a smaller vision tower (0.32B at 336 px, 576 tokens against 729), so it would index faster and make Show a picture faster on the CPU. Its retrieval is a little better than today's. But its text context is only 32 tokens, and there is no evidence it helps on fine-grained queries.
- **The bigger towers are poor value here.** By rough FLOPs, PE-Core G/14 at 448 is about 6–7× so400m, which puts a film at roughly 40 min on an L4 (an estimate) for about 1 point on COCO. Meta CLIP 2 scores below SigLIP 2 on every English number, and its licence is non-commercial. Rule both out.

### C. Video-text encoders (actions and events)

Clip-level encoders (InternVideo2, VideoPrism, PE's video variant) embed a few seconds of frames at once and are trained against video captions, so they can see motion. Adopting one changes the index layout: one vector per 2–4 s window instead of per frame. It needs a second FAISS file next to `clip.faiss`, and segments would fuse two score series. A model with a separate video tower but a shared text space would keep the host cost at one extra text encode.

The cheaper first step is query-time temporal pooling (section A). For an image model, PE reports its video retrieval by averaging 8 frames [[2]](#sources), which FrameHound can do with no new model. Revisit this group only if the evaluation shows that actions are a common miss.

### D. MLLM embedders

**Qwen3-VL-Embedding-2B/8B** [[6]](#sources) (January 2026) leads MMEB-V2. It embeds text, images and video into one space with Matryoshka dimensions, and its understanding of compositional queries is far beyond a dual encoder's. But:

- **The query side is an LLM.** A 2B model on 8 CPU cores is estimated at several seconds per query, with 4–8 GB of RAM in fp16/fp32. That breaks the 1 s goal unless the GTX 1050 runs it quantized.
- **Indexing grows by about an order of magnitude** (estimate). Each of about 14.4k frames passes through a 2B VLM.
- There is no ONNX path. The host would gain torch or llama.cpp, which goes against the design's ONNX-only `api`.

This is not worth it as the first-stage index. Its companion reranker (section A) delivers most of the benefit at a fraction of the cost.

### E. Structural changes (any model)

| Change | Targets | Cost |
|---|---|---|
| **Tiled embeddings:** a 2×2 grid of crops plus the full frame; score each frame as the max over 5 vectors | Small objects | 5× vision compute and 5× index size (about 165 MB fp16 per film); no new model. Already listed in design §13. |
| Higher fps (4 Hz) for short films | Brief objects | 2× compute and index size |
| Open-vocabulary detection (OWL-ViT/OWLv2, Grounding DINO) | "every appearance of X" | Detection per frame is expensive. At query time it would only rerank the top-K. This is design open question 2. |

No global encoder in section B fixes small objects. Tiling does, whatever the backbone.

## 5. Recommendation

In order of cost:

1. **Build the evaluation set first** (section 6). The published generic numbers barely separate the candidates. Only FrameHound's own misses can show which failure dominates.
2. **Reranker.** Implement the Claude reranker already in design §6, behind its opt-in, and measure the precision gain on the evaluation set. It needs no re-index and covers multi-attribute queries. If it pays off and frames shouldn't leave the host, try Qwen3-VL-Reranker-2B (int4) on the GTX 1050.
3. **If the misses are recall misses on multi-attribute queries:** try **FG-CLIP 2 So/16** on a couple of re-indexed films. It is the one encoder with a large published gain on the right kind of benchmark, at about today's compute.
4. **If the misses are small objects:** add tiled embeddings on whatever backbone wins step 3.
5. **If the misses are actions:** try query-time temporal pooling first, then group C.
6. **Hold** PE-Core G/14, Meta CLIP 2 and the MLLM embedders.

PE-Core L/14 is the fallback if FG-CLIP 2's custom code won't export cleanly. It costs less but gains less.

## 6. Deciding empirically

Design only, with nothing built yet:

- **Queries.** 30–50 (query, ground-truth span) pairs, tagged by failure type from the table in §2: small object, multi-attribute, action, long description, plain. One film is indexed so far (`data/live/movies/`). Add 2–3 public-domain films with varied looks.
- **Metrics.** Recall@10 (is a candidate overlapping the true span in the top 10?) and the rank of the first hit, per tag. Quality matters more than latency, but record the query latency too.
- **Running it.**
  - Score the query-side options (reranker, pooling, prompts) on the current index.
  - Score the encoder candidates by re-indexing the films on an ad-hoc Colab session. Name it `bench-*`: the worker reaps `fh-*` sessions.
  - Keep the candidate locks out of `models.lock` until one wins.

## Sources

1. SigLIP 2 paper, Table 1 (zero-shot retrieval R@1): <https://arxiv.org/abs/2502.14786>
2. Perception Encoder model card (PE-Core L/14, G/14: params, res, dim, COCO-T2I, licence): <https://huggingface.co/facebook/PE-Core-G14-448>. Paper: <https://arxiv.org/abs/2504.13181>
3. Meta CLIP 2 (worldwide): <https://huggingface.co/docs/transformers/model_doc/metaclip_2>. NeurIPS 2025: <https://neurips.cc/virtual/2025/poster/117255>
4. FG-CLIP 2 paper, Tables 1 and 3 (FG-OVD, COCO/Flickr, ShareGPT4V/DCI, including SigLIP 2 and Meta CLIP 2 baselines): <https://arxiv.org/html/2510.10921v2>. Model: <https://huggingface.co/qihoo360/fg-clip2-so400m>
5. Qwen3-VL-Reranker-2B: <https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B>. Repo: <https://github.com/QwenLM/Qwen3-VL-Embedding>
6. Qwen3-VL-Embedding report: <https://arxiv.org/abs/2601.04720>. Model: <https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B>

Params for SigLIP 2 are approximate (≈) because the sources give only the size name. Indexing-time and CPU-latency figures marked "estimate" are scaled from today's measured so400m numbers by parameters and tokens. They have not been measured.
