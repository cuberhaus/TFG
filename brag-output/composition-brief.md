# Hyperframes Composition Brief: Polyp Detection Dashboard (TFG)

## Objective
Create a short launch-style brag video for the Polyp Detection Dashboard, a FIB-UPC bachelor's thesis (polyp detection in colonoscopy with generative data augmentation).

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 21.2 seconds

## Source Material
- Project root: `/home/pol/cuberhaus/TFG`
- Primary files read: `README.md`, `AGENTS.md`, `frontend/src/App.tsx`, `frontend/src/components/*` (Data Augmentation / Generative page, InferenceUI), `frontend/src/index.css`, `backend/main.py` (`/api/predict` box drawing), thesis defence slides in `docs/entrega/` (detection figure)
- Product name: Polyp Detection Dashboard
- Tagline / strongest claim: "Deep Learning Polyp Detection with Generative Data Augmentation"; the app's own card: "Medical datasets for polyps are often small."
- Key UI or visual moment to recreate: the Data Augmentation "Prepare dataset" step (boxes → binary mask PNGs), then the Inference tab flow ending in red detection boxes + the "Detection Results" table.
- Copy that must appear verbatim:
  - Not enough polyps.
  - So we taught a GAN to draw more.
  - Bounding box → binary mask
  - Detection Results / 2 found
  - Polyp Detection Dashboard
  - Deep Learning Polyp Detection with Generative Data Augmentation
  - Most theses end as a PDF.
  - This one has ten tabs.
  - FIB-UPC Bachelor's Thesis · github.com/cuberhaus/TFG

## Creative Direction
- Tone preset: default
- Creative direction: dry research-lab launch, the thesis that grew a control room
- Interpretation: calm, clean pacing; one dry joke at each end, the real product in the middle; no hype adjectives; no claims about accuracy or image realism (the thesis lists getting SPADE to work as future work).
- Angle: The thesis problem is a genuinely odd sentence — there aren't enough polyps. State it straight, answer in one dry line, let the real dashboard do the talking (box → mask preparation, then detection), and end on the format joke: most theses end as a PDF, this one has ten tabs.
- Hook: dimmed real colonoscopy frame on screen at frame 0; "Not enough polyps." then, in accent blue, "So we taught a GAN to draw more."
- Outro / punchline: "Most theses end as a PDF. This one has ten tabs."
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals
  - Unrelated visual redesign
  - Any claim about generated-image quality or detection accuracy
  - Real patient data, names, emails, hostnames (none are used)

## Visual Identity
- Background: `#111827` (gray-900)
- Text: `#ffffff`, secondary `#9ca3af`; muted app text lifted to `#8b95a5` for contrast
- Accent: blue-400 `#60a5fa`; sidebar icon colours per `App.tsx`; Inference CTA cyan-700 `#0e7490`
- Display font: Inter (bundled)
- Body font: Inter; code snippets in JetBrains Mono
- Visual references from the project: the dark dashboard shell with grouped sidebar, Data Augmentation prep card with the (mask, image) folder diagram, Train CycleGAN task card, Inference configuration column + result column, red-only detection boxes with `polyp 0.xx` labels.

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. Hook — 4.6 s — "Not enough polyps." / "So we taught a GAN to draw more." over a dimmed real frame
2. Dashboard — 12.6 s (4.0–16.6) — Prepare dataset + box→mask callout, Select Task card, Inference upload → Run Detection → boxes + "2 found" table
3. Outro — 5.1 s (16.1–21.2) — title, tagline, dry line, footer with repo URL

## Audio
- Audio role: warm bed with sparse, motion-matched accents
- Audio arc: bed fades in under the hook, UI clicks carry the demo, one bell when the boxes land, soft hit on the outro title, bed fades under the final footer
- Music: `assets/music/happy-beats-business-moves-vol-9-by-ende-dot-app.mp3`
- Music treatment: volume 0.34, 0.8 s fade-in, 1.4 s fade-out
- Music cue guidance: bundled cue preset for vol-9 (114.84 BPM). Strong cues locked: 4.23 s (dashboard reveal), 12.65 s (boxes land). Beat grid: 16.34 s (outro title), 17.91 s / 18.44 s (joke lines).
- Audio-reactive treatment: subtle — baked bass energy breathes the hook glow and the outro title halo
- Audio-coupled moments:
  - Dashboard reveal — soft impact on the 4.23 s cue
  - Four simulated clicks (5.02 / 9.07 / 10.02 / 10.82 s) — UI click at each press frame
  - Boxes landing — bell accent at 12.6 s
  - Outro title — soft impact at 16.3 s
- SFX selection guidance: soft, low high-frequency-risk interface clicks and impacts; no stinger per text line
- Exact SFX choice: made after the animation existed (see `index.html`)
- Audio files: music and SFX are copied under `composition/assets/`

## Hyperframes Instructions
Built with the Hyperframes `hyperframes`, `hyperframes-cli` and `media-use` guidance (paused GSAP timeline, scene `<template>`s per host, `fromTo` entrances, no `repeat:-1`/`Date.now`/network fetches). `npx hyperframes check` is the single pre-render gate.
