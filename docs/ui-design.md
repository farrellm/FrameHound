# FrameHound — UI Design

| | |
|---|---|
| **Status** | Draft |
| **Author** | farrellm |
| **Last updated** | 2026-10-15 |
| **Companion to** | [design.md](design.md) (system and pipeline) |

## 1. Purpose

This document specifies the web UI for FrameHound: what each screen shows, how it behaves on an iPhone and on a desktop, and the visual language that ties them together. It does not cover implementation. The `web` service in [design.md §3](design.md#3-architecture) builds to this spec.

- **Audience.** One person running their own install. They often reach it from a phone: on the couch, mid-conversation, trying to find "that bit where…".
- **Primary job.** Get from *I remember a moment* to *a clip saved on my device* in as few taps as possible.
- **Parity principle.** The iPhone is not a reduced version of the desktop. Every task, including precise in/out trimming, must be comfortable with one thumb. The desktop layout spends its extra space on seeing more at once, and adds no extra capability.

## 2. Design direction

### Concept: the editor's light table

Before nonlinear editing, film editors found a moment by laying strips of film on a backlit table, squinting at frames, and marking cut points with a china marker (a grease pencil). FrameHound does the same job with a GPU doing the squinting, so the UI borrows that table:

- The **timeline is a strip of film** lying on a lit surface: an Emulsion-colored body, sprocket holes along both edges, and frame thumbnails in the gate.
- **Search results are grease-pencil marks** drawn on the strip. Their height and density show how strongly each moment matches. The heat-map is just the editor's marks.
- **In and out points are china-marker brackets**, the same yellow.

That strip is the one bold element. Everything around it (lists, sheets, forms) stays quiet, flat, and typographic.

### Color tokens

| Token | Light | Dark | Role |
|---|---|---|---|
| `--lightbox` | `#E8EEF1` | `#151C26` | Page background. Light mode is a cool, backlit white. Dark is a blue slate, the cutting room with the lights down. |
| `--diffuser` | `#F7FAFB` | `#1F2835` | Raised surfaces: sheets, the query bar, the add-movie field. |
| `--table` | `#E8EEF1` | `#C9D3DA` | The lit panel the film strip lies on. **In dark mode the table stays lit**, which is the concept and also the contrast fix (see below). |
| `--emulsion` | `#1C2533` | `#1C2533` | Film-strip body (both modes), and body text in light mode. |
| `--ink` | `#1C2533` | `#F7FAFB` | Body text. |
| `--fog` | `#5B6878` | `#9AA7B6` | Secondary text: durations, timecode ranges, captions. |
| `--marker` | `#F2B705` | `#F2B705` | China marker: heat marks, in/out brackets, primary button fill. |
| `--splice` | `#23688A` | `#6BB8D8` | Links, the selected candidate, focus rings. |
| `--danger` | `#B3261E` | `#F2867E` | Destructive actions (Cancel indexing, Delete movie) and error text. |

**Rules that come from the contrast checks** (WCAG 2.x, computed and recorded in §2.6):

- `--marker` is **never placed directly on `--lightbox`** (1.55:1). It appears only on the Emulsion strip (8.5:1), or as a button fill with `--emulsion` text on it (8.5:1).
- Sprocket holes are drawn in `--table` color, as light showing through the film, rather than in `--fog`, which only reaches 2.7:1 on Emulsion.
- Focus inside the strip uses a double ring: a 2 px `--diffuser` inner ring and a 2 px `--splice` outer ring. Splice alone on Emulsion only reaches 2.5:1 in light mode.
- Heat is never shown by color alone. Mark **height** and **density** carry the score (§4.3).

### Type

| Role | Face | Notes |
|---|---|---|
| Screen titles, movie titles, the large playhead timecode | **Big Shoulders Display** 800 | A condensed signage face that reads like the lettering on a clapperboard slate. Used sparingly. |
| Everything else | **Atkinson Hyperlegible Next** 400 / 700 | Designed for legibility at small sizes and at a glance on a phone. All timecodes use `font-variant-numeric: tabular-nums` (feature verified present). |

The large timecode is the one place Big Shoulders shows numbers. Big Shoulders **has no `tnum` feature, and its digits are proportional**, so the timecode would shimmy as it counts. To prevent that, each digit sits in a fixed cell (`inline-block`, width `0.6em`, centered), like the digit wheels on a counter. This is deliberate, and it is the second most recognizable element after the strip.

No other monospace face is used: tabular numerals do that job.

**Scale.** Ratio 1.25 (major third), base 17 px to match iOS body text. Sizes are set in `rem`, so iOS Dynamic Type and browser zoom scale them.

| Step | px (phone) | px (desktop) | Use |
|---|---|---|---|
| −2 | 11 | 11 | Sprocket-edge frame numbers on the strip only |
| −1 | 14 | 14 | Captions, timecode ranges |
| 0 | 17 | 17 | Body, list rows, buttons |
| +1 | 21 | 21 | Section headings inside sheets |
| +2 | 27 | 27 | Movie title on the Movie screen |
| +3 | 33 | 42 | Screen titles (Big Shoulders) |
| +4 | 42 | 66 | Playhead timecode (Big Shoulders, fixed cells) |

- **Line height:** 1.45 for body text, 1.1 for Big Shoulders.
- **Measure:** at most 68 ch for any running text.
- **Case:** sentence case everywhere. There are no all-caps labels, no eyebrow labels above headings, no middle-dot meta strings, and no arrows appended to buttons or links.

### Spacing, radius, elevation

- **Spacing:** 4 px base, using the steps 4 / 8 / 12 / 16 / 24 / 32 / 48. The phone side gutter is 16 px. The desktop gutter is 32 px.
- **Radius follows hierarchy**, not one value everywhere:
  - Film frames and the strip: **0**. Film has square corners.
  - Buttons and inputs: **10**.
  - Sheets: **16** on their top corners only. They rise from the bottom like iOS sheets.
- **Elevation.** Only sheets cast a shadow, because they are the only things that float. Everything else is flat on the table and separated by color, not by lines.

### Motion

There is **one** orchestrated moment. When search results arrive, the grease-pencil marks draw onto the strip in a single left-to-right sweep (600 ms, ease-out, staggered by position). Then the candidate list fills in at once.

Every other motion answers a user action:

- Sheets slide up.
- The strip zooms toward a selected candidate (250 ms).
- In/out brackets snap to the playhead.

With `prefers-reduced-motion: reduce`, all of these are instant cuts. Nothing loops, pulses, or shimmers, except the single progress fill on the active indexing stage.

### Contrast record

| Pair | Ratio | Needs |
|---|---|---|
| ink on lightbox (light) | 13.2 | 4.5 ✓ |
| fog on lightbox (light) | 4.85 | 4.5 ✓ |
| splice on lightbox (light) | 5.25 | 4.5 ✓ (darkened from `#2E7DA1`, which was 3.9) |
| emulsion on marker (button text) | 8.5 | 4.5 ✓ |
| marker on emulsion (heat marks, brackets) | 8.5 | 3.0 ✓ |
| ink on lightbox (dark) | 16.3 | 4.5 ✓ |
| fog on diffuser (dark) | 6.1 | 4.5 ✓ |
| splice on diffuser (dark) | 6.7 | 4.5 ✓ |
| table vs. lightbox (dark), strip's container | 11.3 | 3.0 ✓ |
| emulsion strip on table (dark) | 10.1 | 3.0 ✓ (an Emulsion strip on the dark lightbox was 1.1, which is why the table stays lit) |

### Checked against generic defaults

These are choices I made, then revised because they matched the defaults any similar brief would produce:

- **Dark "screening room" with a single amber accent.** This was my first instinct for a movie tool, and it is the near-black-plus-one-accent default. I replaced it with the light table, which comes from the subject's own work practice. Dark mode is derived from that table, not the other way around.
- **Candidates as rounded cards with soft shadows.** I replaced these with contact-sheet strips: square frames on film, with no shadow.
- **One border radius on everything.** I replaced it with radius by hierarchy: 0, 10, 16.
- **Monospace for timecodes.** I replaced it with tabular numerals in the body face, plus the fixed-cell big timecode.
- **Numbered markers.** These are used only on the indexing stages, which really are a sequence.

## 3. Navigation model

```
Library ──(Add movie)──► Movie ──┬── Indexing   (until ready)
   ▲                             └── Search     (once ready)
   └──────── back ───────────────────────┘          │
                                         Export sheet (over Search)
```

| Route | Screen |
|---|---|
| `/` | Library |
| `/m/{id}` | Movie. Shows Indexing or Search depending on the movie's state. |
| `/m/{id}?q=…&mode=describe\|picture\|quote` | Search with a query restored. Shareable, and it survives a reload. |
| `/m/{id}?in=…&out=…` | A selection restored, ready to export. |

The **Movie** screen is one URL with two states, so a phone left on the Indexing view turns into Search by itself when the movie is ready. There is no tab bar: the app has one primary path, and a back button covers the rest.

## 4. Screens

Phone frames are drawn at about 36 columns. Desktop frames are wider. `▓` marks a China-marker fill, and `░` marks a grease-pencil heat mark.

### 4.1 Library and adding a movie

**Job:** paste a link and leave, or pick up a movie added earlier.

```
iPhone                                Desktop
┌────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
│ FrameHound                     │    │ FrameHound                                           │
│                                │    │                                                      │
│ ┌────────────────────────────┐ │    │ ┌──────────────────────────────────┐ ┌──────────────┐│
│ │ Paste a movie link or      │ │    │ │ Paste a movie link or magnet link│ │▓▓Add movie▓▓▓││
│ │ magnet link                │ │    │ └──────────────────────────────────┘ └──────────────┘│
│ └────────────────────────────┘ │    │ or choose a .torrent file                            │
│ ▓▓▓▓▓▓▓▓ Add movie ▓▓▓▓▓▓▓▓▓▓ │    │                                                      │
│ or choose a .torrent file      │    │ ▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚ │
│                                │    │ [▣][▣][▣][▣][▣]  Night of the Living Dead           │
│ ▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚ │    │                  1:36:12  Ready to search            │
│ [▣][▣][▣]  Night of the        │    │ ▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚ │
│            Living Dead         │    │ [▣][▣][▣][▣][▣]  His Girl Friday                    │
│            1:36:12  Ready      │    │                  1:32:04  Finding faces  62%         │
│ ▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚ │    │ ▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚ │
│ [▣][▣][▣]  His Girl Friday     │    └──────────────────────────────────────────────────────┘
│            Finding faces 62%   │
└────────────────────────────────┘
```

- **The field** is a single input that accepts an `https://` URL, an archive.org link, or a `magnet:` link. Its placeholder says exactly that.
  - On iPhone the keyboard is `inputmode="url"`, and on focus a **Paste** chip appears above the keyboard. Most links arrive by copying from Safari.
  - "Choose a .torrent file" opens the Files picker.
  - The button is disabled until the field parses as one of the three source types.
- **Each movie row is a short film strip**: three (phone) or five (desktop) shot thumbnails, then the title, duration, and state. Rows are separated by the strip's own sprocket edge, with no divider lines. Tapping a row opens the Movie screen.
  - A row in progress shows the current stage name and a percentage.
  - A failed row shows **Indexing failed** in `--danger` and the one-line cause.
- **archive.org collection.** If the link is a collection (as `archive.org/details/bittorrent` is), a sheet opens instead of starting a job:

```
┌────────────────────────────────┐
│ This is a collection.          │
│ Pick a movie to add.           │
│ ┌────────────────────────────┐ │
│ │ Search this collection     │ │
│ └────────────────────────────┘ │
│ ○ Plan 9 from Outer Space      │
│ ○ Detour                       │
│ ○ The Stranger                 │
│   …                            │
│ ▓▓▓▓▓▓▓▓ Add movie ▓▓▓▓▓▓▓▓▓▓ │
└────────────────────────────────┘
```

**States**

| State | Copy |
|---|---|
| Empty library | Title: "Nothing on the table yet." Body: "Paste a link to a movie. Public-domain films from the Internet Archive are a good place to start." |
| Unrecognized link | "That isn't a web address or magnet link. Paste a link that starts with https:// or magnet:." |
| Already added | "You've already added this movie." Button: **Open it** (goes straight to Search, with no second indexing). |
| Collection with no videos | "This collection has no movies FrameHound can index. Try a link to a single item." |
| Blocked address (SSRF guard) | "FrameHound can't download from private network addresses. Use a public link." |

### 4.2 Indexing progress

**Job:** show the movie is moving, and make it obvious the user can leave.

```
iPhone                                Desktop
┌────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
│ ‹ Library                      │    │ ‹ Library                                            │
│ His Girl Friday                │    │ His Girl Friday                                      │
│ 1:32:04                        │    │ 1:32:04                                              │
│                                │    │                                                      │
│ 1 Downloading            done  │    │ 1 Downloading        done    ┃ You can close this    │
│ 2 Getting a GPU     L4   done  │    │ 2 Getting a GPU  L4  done    ┃ page. Indexing keeps  │
│ 3 Sending the movie      done  │    │ 3 Sending the movie  done    ┃ going, and the movie  │
│ 4 Finding shots          done  │    │ 4 Finding shots      done    ┃ will be ready in      │
│ 5 Looking at frames      done  │    │ 5 Looking at frames  done    ┃ about 6 minutes.      │
│ 6 Finding faces           62%  │    │ 6 Finding faces       62%    ┃                       │
│   ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░        │    │   ▓▓▓▓▓▓▓▓▓▓▓░░░░░░          ┃ GPU: Colab L4         │
│ 7 Listening to dialogue   48%  │    │ 7 Listening to dialogue 48%  ┃ Used so far:          │
│ 8 Packing the index            │    │ 8 Packing the index          ┃ 1.3 compute units     │
│                                │    │                              ┃                       │
│ About 6 minutes left.          │    │ [▣][▣][▣][▣][▣][▣][▣][▣] shots found so far           │
│ You can close this page.       │    │                              ┃ Cancel indexing       │
│ Indexing keeps going.          │    └──────────────────────────────────────────────────────┘
│                                │
│ GPU: Colab L4                  │
│ Used so far: 1.3 compute units │
│                                │
│ Cancel indexing                │
└────────────────────────────────┘
```

- **The stages are numbered because they really are a sequence.** Each one is named in plain words, never with the system's stage keys:

  | Shown | Pipeline stage ([design.md §5](design.md#5-stage-2--index-colab-gpu-worker-index)) |
  |---|---|
  | Downloading | ingest: downloading, probing, transcoding_proxy |
  | Getting a GPU | provisioning_gpu |
  | Sending the movie | staging_media, fetch |
  | Finding shots | shots |
  | Looking at frames | frames, visual |
  | Finding faces | faces |
  | Listening to dialogue | audio, sentences |
  | Packing the index | thumbs, bundle, downloading_index, importing |

- Stages that run at the same time (6 and 7) each show their own percentage.
  - Only the active stage has a progress bar: a china-marker fill on an Emulsion track.
  - Completed stages say **done** in `--fog`.
- **Shots found so far** (desktop, and on phone below the GPU line) shows a strip that fills with thumbnails as stage 4 produces them. It is the only decoration on the screen, and it is real data.
- **Cost** shows the GPU and the compute units used so far, so spending is never a surprise.
- **Cancel indexing** is a `--danger` text button. Tapping it opens a confirmation sheet: "Stop indexing His Girl Friday? The GPU is released and the download is kept." The buttons are **Stop indexing** and **Keep going**.
- **When indexing finishes**, the screen becomes Search in place. If the page was in the background, a toast reads **Ready to search** on return, and an optional push notification is sent (§5).

**Errors** name the cause and the fix, and keep the completed stages visible:

| Cause | Copy | Action |
|---|---|---|
| Colab auth broken | "Colab sign-in has expired. On the server, run `gcloud auth application-default login` with the Colab scopes, then retry." | **Retry** |
| No GPU available | "Colab has no L4 or T4 available for this account right now. FrameHound will try again in 10 minutes." | **Index on this computer instead** (slow) |
| Budget reached | "This would pass this month's GPU budget (40 compute units). Raise the budget in settings or index on this computer." | **Index on this computer instead** |
| Download stalled | "The torrent has had no peers for 10 minutes." | **Retry**, **Remove movie** |

### 4.3 Search and timeline

**Job:** say what you're looking for, see where it is, and get the edges right.

```
iPhone (portrait)                     Desktop (≥ 960 px)
┌────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
│ ‹ Library   His Girl Friday    │    │ ‹ Library   His Girl Friday                           │
│ ┌────────────────────────────┐ │    │ ┌──────────────────────────────┐ Describe it ▸ man in │
│ │                            │ │    │ │                              │ a hat on the phone   │
│ │          player            │ │    │ │           player             │ ───────────────────  │
│ │                            │ │    │ │                              │ [▣][▣][▣] Looks like │
│ └────────────────────────────┘ │    │ │                              │ 0:14:02–0:14:19      │
│        0 0:1 4:0 6 .1 2        │    │ └──────────────────────────────┘                      │
│ ▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚ │    │         0 0 : 1 4 : 0 6 . 1 2  [▣][▣][▣] Looks like │
│ [▣][▣][▣][▣][▣][▣][▣][▣][▣][▣] │    │                                  0:51:40–0:51:47      │
│   ░  ░░▌▓▓▓▓▓▐   ░       ░░    │    │ ▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚ │
│ ▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚ │    │ [▣][▣][▣][▣][▣][▣][▣][▣][▣][▣][▣][▣][▣][▣][▣][▣][▣] │
│ Set in  ‹‹ ‹  › ››  Set out    │    │  ░  ░░ ▌▓▓▓▓▓▓▐   ░     ░░░          ░      ░░        │
│ ▓▓▓▓▓ Cut clip 0:17 ▓▓▓▓▓▓▓▓▓ │    │ ▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚ │
│                                │    │ Set in  ‹‹ ‹  › ››  Set out          ▓▓ Cut clip ▓▓  │
│ 3 matches                      │    └──────────────────────────────────────────────────────┘
│ [▣][▣][▣]  Looks like          │
│            0:14:02–0:14:19     │
│ [▣][▣][▣]  Looks like          │
│            0:51:40–0:51:47     │
│ [▣][▣][▣]  Same face           │
│            1:02:11–1:02:20     │
│┌──────────────────────────────┐│
││ Describe │ Picture │ Quote   ││
││ ┌─────────────────────┐ Find ││
││ │ man in a hat on the │      ││
││ └─────────────────────┘      ││
│└──────────────────────────────┘│
└────────────────────────────────┘
```

(The segmented control's full labels are **Describe it**, **Show a picture**, and **Quote a line**.)

**Query bar.**

- **iPhone:** the bar is pinned to the bottom, above the home indicator, with `padding-bottom: env(safe-area-inset-bottom)`, so it is always in thumb reach.
- **Desktop:** the bar sits at the top of the right column.

The three modes map to the API's `type`:

| Mode | Input | Placeholder / affordance | API |
|---|---|---|---|
| **Describe it** | Text, 1–3 lines, grows | "Describe what's on screen, like a man in a hat on the phone" | `text` |
| **Show a picture** | `<input type=file accept="image/*">` (iOS offers Photo Library, Take Photo, Choose File), plus paste and drop on desktop | A thumbnail of the chosen image with a ✕ to clear | `image` |
| **Quote a line** | Text | "Type a line of dialogue, even roughly" | `dialogue` |

**Find** runs the search. The last query per movie is kept in the URL (§3).

**The strip** (the signature element):

- **Body.** Emulsion, with sprocket holes along the top and bottom edges in `--table` color. Shot thumbnails sit in the gate at their real positions in time.
- **Heat.**
  - Grease-pencil marks in `--marker` sit under the gate. Each mark's **height** (0–100% of the band) is the smoothed score z(t) from the API's `heat` series.
  - Above threshold, marks thicken into a solid bar. Height and thickness carry the information, so it never depends on color.
  - The marks draw in once per search (§2, Motion).
- **Candidate span and handles.** The selected candidate shows as a span with china-marker brackets `▌ ▐` (in/out), filled at 35% marker over the frames.
- **Playhead.** A 2 px Diffuser line with a downward notch. The big timecode above the strip follows it.
- **Overview and zoom.** At rest the strip shows the whole movie (overview). **Selecting a candidate zooms the strip to the candidate ±10 s** (250 ms). At that scale, one thumb-width (44 pt) is about half a second on a phone, so the brackets can be dragged precisely. Tap **Whole movie** (top-right of the strip when zoomed) or pinch out to return.
- **Gestures.**
  - Horizontal drag on the strip scrubs.
  - Dragging a bracket moves that bracket. Its hit area is 44 × 44 pt even though the visible mark is 6 px wide.
  - On iPhone, a long press on a bracket gives haptic feedback where supported and shows a loupe of the frame.
  - Vertical drags pass through to page scroll. The strip never traps scrolling.

**Precision row.** This row is how the phone matches desktop accuracy without precise dragging:

| Control | Does |
|---|---|
| **Set in** / **Set out** | Move the bracket to the playhead. |
| `‹‹` `››` | Step the playhead by ½ s. |
| `‹` `›` | Step the playhead by one frame. |

The nudges have accessible names: "Back half a second", "Back one frame", and so on.

**Desktop keys** follow editing convention:

| Key | Action |
|---|---|
| J / K / L | Reverse / pause / forward. Repeat J or L to go faster. |
| I / O | Set in / set out. |
| ← / → | One frame. With Shift, half a second. |
| ↑ / ↓ | Previous or next candidate. |
| Enter | Cut clip. |

Pressing ? opens the key list.

**Candidates.**

- Each candidate is a short contact-sheet strip of three thumbnails from its span, followed by:
  - **Reason**, in words: **Looks like** (visual), **Same face** (face), or **Says** (dialogue). A dialogue candidate also shows the matched words, quoted.
  - **Span** in tabular numerals, `0:14:02–0:14:19`.
- There is no numeric score. Order carries rank, and the strip shows strength.
- Tapping a candidate selects it, zooms the strip, seeks the player to its peak, and plays muted on iPhone (unmuted on desktop).
- The selected candidate has a 3 px `--splice` bar on its leading edge.

**Search states.**

| State | Copy |
|---|---|
| Before the first search | "What are you looking for? Describe it, show a picture, or quote a line." |
| Searching (under 1 s) | The Find button shows "Finding…". The strip dims to 60%. There is no spinner. |
| No match above threshold | "No clear matches. The strongest moments are marked on the strip. Try fewer words, or a different mode." The strip still shows weak heat, so the user can browse it. |
| Picture with no face, in a face-like query | Shown as a quiet note under results, not an error: "No face found in this picture, so FrameHound matched how it looks overall." |
| Index built with different models | "This movie was indexed with older models. Re-index it to search." Button: **Re-index**. |

### 4.4 Export

**Job:** turn the span into a file on the device.

Export opens as a **sheet** over Search, so the strip stays visible behind it and the user can see exactly what is being cut.

```
iPhone sheet                          Desktop (sheet anchored right, 400 px)
┌────────────────────────────────┐    ┌───────────────────────────┐
│ ─────                          │    │ Cut clip                  │
│ Cut clip                       │    │ 0:14:02.12 – 0:14:19.04   │
│ 0:14:02.12 – 0:14:19.04  17 s  │    │ 17 seconds                │
│                                │    │                           │
│ Format                         │    │ Format                    │
│ (●) Video (MP4)                │    │ (●) Video (MP4)           │
│ ( ) Video with subtitles       │    │ ( ) Video with subtitles  │
│ ( ) GIF   up to 15 s           │    │ ( ) GIF   up to 15 s      │
│                                │    │                           │
│ Cutting                        │    │ Cutting                   │
│ (●) Exact   cuts on the frame  │    │ (●) Exact                 │
│ ( ) Quick   instant, may start │    │ ( ) Quick                 │
│     a moment early             │    │                           │
│                                │    │ [ ] Crop to the face      │
│ [ ] Crop to the face           │    │                           │
│                                │    │ ▓▓▓▓▓▓ Cut clip ▓▓▓▓▓▓▓▓ │
│ ▓▓▓▓▓▓▓▓ Cut clip ▓▓▓▓▓▓▓▓▓▓▓ │    └───────────────────────────┘
└────────────────────────────────┘
```

- Choices are radio groups with plain names. The *system's* terms (accurate/fast, `-c copy`) never appear.
  - **Exact**: "cuts on the frame".
  - **Quick**: "instant, may start a moment early".
- **Crop to the face** appears only for **Same face** candidates. Cropping re-encodes, so while it is checked **Quick** is disabled with the reason inline: "not available when cropping to the face".
- **GIF up to 15 s.** If the span is longer, the GIF option is disabled with the reason inline: "GIF is limited to 15 seconds. Shorten the clip to use it."
- **The action keeps one name through the flow:**
  - The button is **Cut clip**.
  - While running, the label changes to **Cutting…** with a china-marker fill sweeping across the button, the one progress bar in this sheet.
  - When done, the sheet shows **Clip ready**, a looping muted preview, and a save button.
- **Saving.**
  - **iPhone:** **Save clip** calls `navigator.share({ files: [clip] })`. The iOS share sheet offers *Save Video* (to Photos), Messages, AirDrop, and so on. A plain download is the fallback where `canShare({files})` is false.
  - **Desktop:** **Save clip** downloads the file.
  - In both cases a secondary **Copy link** copies the `/clips/{id}/file` URL.

## 5. iPhone specifics

| Concern | Decision |
|---|---|
| Video | `<video playsinline>`, so it never forces fullscreen. A fullscreen button is offered explicitly. Muted autoplay only on candidate selection. |
| Viewport | `viewport-fit=cover`. Layout uses `100dvh` so the dynamic Safari toolbar doesn't cover the query bar. Safe-area insets on the top bar, query bar, and sheets. |
| Landscape | Rotating on Search gives player and strip only, edge to edge. Candidates and the query bar move into a sheet that a bottom-edge handle pulls up. |
| Keyboard | Opening the keyboard in the query bar collapses the player to a 25%-height letterbox, so results stay visible above the keyboard. |
| Background tabs | iOS kills background connections. On `visibilitychange` → visible, the app refetches movie and job state, then reopens SSE with `Last-Event-ID`. Progress never looks frozen after returning. |
| Notifications | Optional. If FrameHound is added to the Home Screen as a PWA (iOS 16.4+), it can send a push notification: "His Girl Friday is ready to search." The permission prompt is only shown after the user adds their first movie, never on load. |
| Touch | All hit targets are at least 44 × 44 pt. Nothing relies on hover. Long-press is an enhancement, never the only path. |
| Haptics | Where `navigator.vibrate` is unsupported (iOS Safari), brackets snap visually only. No features depend on haptics. |

## 6. Accessibility

- **Strip.** The strip is a composite widget.
  - The playhead is `role="slider"` with `aria-valuetext="14 minutes 6 seconds"`.
  - Each bracket is its own slider ("Clip start", "Clip end").
  - Arrow keys work as described in §4.3.
  - The heat band is `aria-hidden`. Its information is repeated in the candidate list.
- **Candidates.** Each is announced as a single item, for example: "Looks like, 14 minutes 2 seconds to 14 minutes 19 seconds, match 1 of 3."
- **Focus.** Focus is always visible: 2 px `--splice` with a 2 px offset, or the double ring on the strip (§2). Focus moves into sheets when they open and returns to the trigger when they close.
- **Color independence.** Heat uses height and thickness. Status uses words ("done", "Indexing failed"), not only color.
- **Text scaling.** Every size is in `rem`. The layout reflows down to a 320 px width at 200% zoom with no horizontal scroll.
- **Motion.** `prefers-reduced-motion` removes the sweep, zooms, and sheet slides.
- **Live regions.**
  - Stage changes and "Ready to search" are announced politely.
  - Errors are announced assertively.
  - Percentages are *not* announced on every tick. They are announced every 25%.

## 7. Copy glossary

One name per thing, used the same way everywhere:

| Thing | Always called | Never called |
|---|---|---|
| Starting ingest | **Add movie** | Import, Submit, Upload |
| Running a search | **Find** | Search, Go, Submit |
| The three modes | **Describe it**, **Show a picture**, **Quote a line** | Text query, Image query, Dialogue search |
| A result | **match** | candidate, hit, result |
| Clip boundaries | **in** / **out** (Set in, Set out) | start/end handles, trim points |
| Making the file | **Cut clip** → **Cutting…** → **Clip ready** | Export, Render, Generate |
| Getting the file | **Save clip** | Download, Export |
| Stopping a job | **Cancel indexing** → **Stop indexing** | Abort, Kill |

Tone rules:

- Errors say what happened and what to do, in the interface's voice. They don't apologize.
- Empty states invite the next action.
- Numbers come with units ("17 s", "1.3 compute units").
- The system's internals (stage keys, ffmpeg, FAISS) never appear, except in an expandable **Details** under an error. That is for the person who runs the server, who is also the user.

## 8. Mapping to the API

| Screen | Calls ([design.md §8](design.md#8-api)) |
|---|---|
| Library | `GET /movies`, `POST /movies` (returns `{choices}` for a collection, which opens the picker sheet) |
| Indexing | `GET /movies/{id}`, `GET /jobs/{id}/events` (SSE), `POST /jobs/{id}/cancel` |
| Search | `POST /movies/{id}/search` (`candidates` populate the list, `heat` draws the strip), `GET /media/{id}/proxy.mp4` |
| Export | `POST /movies/{id}/clips`, `GET /jobs/{id}/events`, `GET /clips/{id}/file` |
| Errors / banners | `GET /health` (Colab auth, balance) |

**API additions this UI relies on.** These have been added to design.md (§6 candidate fields, §8 endpoints):

1. `GET /movies/{id}/shots?from=&to=`, which returns thumbnail URLs for the strip at the current zoom.
2. Search `heat` returned at a requested resolution (`heat_bins`), so the phone overview doesn't download 14k points.
3. Each candidate gains `matched_words` (dialogue) and `has_face` (to enable Crop to the face).
4. `GET /movies/{id}/index-progress` returns a snapshot of the stage list, for the refetch-on-resume described in §5.
