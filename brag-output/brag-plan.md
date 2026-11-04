# Brag Plan: Polyp Detection Dashboard (TFG)

## 9-question rubric (Step 1)

1. **What is the app?** A FIB-UPC bachelor's thesis that detects polyps in colonoscopy frames (Faster R-CNN / RetinaNet / SSD), trained on data augmented by CycleGAN and SPADE, wrapped in a React + FastAPI dashboard with ten sidebar tabs.
2. **Funniest / most impressive claim.** The dashboard's own card: *"Medical datasets for polyps are often small."* The thesis problem is literally "not enough polyps", and the fix is teaching a GAN to draw more.
3. **Visual hook.** A real colonoscopy frame (from the thesis) with a polyp, dimmed behind a blunt line of text; later the same frame with red detection boxes inside the dark gray-900 dashboard.
4. **What to show from the actual UI.** Data Augmentation page: the "Prepare dataset" step (the UI's own words: *rasterises the boxes into binary mask PNGs*) → the Train CycleGAN task card with its real default fields → Inference tab: upload → "Running Model..." → boxes + "Detection Results · 2 found" table.
5. **Shortest satisfying video.** ~21 s.
6. **Tone.** Preset `default`; direction: *dry research-lab launch, the thesis that grew a control room.*
7. **Audio.** Laid-back business-y bed, UI clicks synced to the simulated cursor, one soft reveal hit, one result accent.
8. **Share caption.** "Not enough polyps. So we taught a GAN to draw more." (see `share-copy.txt`)
9. **User flow worth showing.** Prepare dataset (box → mask) → select Train CycleGAN → open Inference → upload a frame → Run Detection → boxes + scores table.

## What is this app?
A bachelor's thesis in polyp detection with generative data augmentation, shipped as a ten-tab web dashboard ("Polyp Detection Dashboard — Deep Learning Polyp Detection with Generative Data Augmentation").

## The angle
The thesis problem is a genuinely odd sentence: *there aren't enough polyps.* The video states it straight, answers it in one dry line, then lets the real dashboard do the talking: the real Prepare-dataset step turning a bounding box into a binary mask (the input the GAN learns from), the real task cards, and the real detection flow on a real colonoscopy frame. Punchline is the format joke: most theses end as a PDF; this one has ten tabs.

Deliberate restraint: nothing in the video claims the generated images are any good (the thesis itself lists getting SPADE to work as future work). The video says the GAN was *taught to draw more*, and shows the pipeline, not generation quality.

## Hook (first 2-3 seconds)
Dark ground with a dimmed real colonoscopy frame already on screen at frame 0. Big line: **"Not enough polyps."** (hold) then, in accent blue: **"So we taught a GAN to draw more."**

## Key moments (the middle)
- The dashboard blur-swaps in on the **Data Augmentation** page. The cursor clicks **Prepare dataset**; a callout lifts out of the card showing the frame with its green ground-truth box → a binary white-rectangle mask, captioned **"Bounding box → binary mask"**.
- Camera scrolls to the **Select Task** card: **Train CycleGAN** selected, with its real default fields (batch 4, epochs 5, lr 0.0002, ResNet 9 blocks, 286 / 256) next to the "Why Generative Data?" card.
- Cursor clicks **Inference** in the sidebar, clicks the upload drop-zone (the frame appears), clicks **Run Detection**; "Processing images through PyTorch..." spinner, then two red boxes land on the polyp, `2 found` badge, scores table.

## Outro / punchline
Title **Polyp Detection Dashboard**, the project's own tagline, then the dry line: **"Most theses end as a PDF. This one has ten tabs."** Footer: `FIB-UPC Bachelor's Thesis · github.com/cuberhaus/TFG`.

## User flow worth showing
Entry: Data Augmentation → Prepare dataset → Inference. Key action: upload a colonoscopy frame, Run Detection. Result: bounding boxes + score table (`2 found`).

## Tone
- Preset: `default`
- Creative direction: dry research-lab launch, the thesis that grew a control room
- Interpretation: Calm, clean pacing; one dry joke at each end, product in the middle; no hype adjectives, no claims about accuracy or realism.

## Format: landscape — 1920x1080
## Duration: 21.2 s

## Visual identity (from the project)
- Background: `#111827` (Tailwind gray-900; `body` in `index.css`)
- Panels: `#1f2937` (gray-800), borders `#374151` (gray-700), inputs `#111827`
- Text: `#ffffff`, secondary `#9ca3af` (gray-400); the app's muted gray-500 text is lifted to `#8b95a5` so it passes WCAG on the dark panels
- Accents (sidebar icon colours from `App.tsx`): blue-400 `#60a5fa`, purple-400 `#c084fc`, purple-300 `#d8b4fe`, fuchsia-300 `#f0abfc`, fuchsia-400 `#e879f9`, teal-400 `#2dd4bf`, emerald-400 `#34d399`, green-400 `#4ade80`, indigo-400 `#818cf8`, cyan-400 `#22d3ee`; Inference CTA cyan (cyan-700 `#0e7490` here instead of the app's cyan-600, for white-text contrast)
- Display font: Inter (the app's `font-family`: Inter, system-ui)
- Body font: Inter
- Strongest visual element: real colonoscopy frame with red detection boxes + the multicolour sidebar

## Real vs. stand-in material (nothing personal on screen)
- **Real:** all dashboard copy, layout and default field values (from `frontend/src/`), the colonoscopy frame (from the thesis defence slides in `docs/entrega/`, LDPolypVideo), the box geometry (measured from that figure), the red-only box style and `polyp 0.xx` labels (what `/api/predict` draws), repo URL (README).
- **Stand-ins (disclosed):** the Inference *preview* is the thesis frame with its drawn boxes inpainted away and the white figure border trimmed; the two scores (`0.91`, `0.64`) are placeholder values; the weights name in the select (`FasterRCNN_LR-0.0053_EP-3`) is a placeholder in the app's naming style; the mask in the callout is derived from the box the way the project's mask step does it. No patient data, no names, no emails, no hostnames.

## Share copy (draft)
Not enough polyps. So we taught a GAN to draw more. My bachelor's thesis shipped as a ten-tab dashboard: CycleGAN, Faster R-CNN, and an Inference tab that boxes the polyp. github.com/cuberhaus/TFG

## Audio direction
- Role: warm bed with sparse, motion-matched accents
- Music: `happy-beats-business-moves-vol-9` (114.84 BPM, mid-energy, laid-back; fits `default` + dry)
- Music treatment: fades in over 0.8 s at ~0.34 volume, stays under UI sounds, fades out over the last ~1.4 s under the outro
- Music cue guidance: preset read (`vol-9` cue preset). Strong cues used: 4.23 s (dashboard reveal), 12.65 s (detection boxes land), plus beat grid 16.34 s (outro title) and 17.91 / 18.44 s (the two joke lines).
- Audio-reactive treatment: subtle; music bass energy (baked per-0.1 s from the track) breathes the glow behind the hook and the halo behind the outro title. No waveform/equalizer visuals.
- SFX posture: moderate, motion-matched. Two soft drops on the hook lines, one soft impact on the reveal, a UI click on each of the four simulated clicks, one bell accent when the boxes land, one soft impact on the outro title.
- Restraint rule: no stinger on every text line; nothing louder than the UI clicks except the single result accent.

## Storyboard

### Scene 1 — Hook — 0–4.6 s
Dimmed real colonoscopy frame behind (visible at frame 0 so the first frame is shareable). "Not enough polyps." lands at 0.25 s (word-by-word blur-in) and holds. At 1.4 s "So we taught a GAN to draw more." lands below in accent blue and holds ~2.2 s. Exits (blur-out) at 3.9 s so the dashboard reveal lands clean.
Audio: bed fades in; a soft drop under each line.
Transition mood: blur swap → Scene 2

### Scene 2 — The dashboard working — 4.0–16.6 s
One continuous stage: the app recreated at 1372×772 and scaled to 1080p, with a virtual camera (scale / pan only, no CSS filters on footage).
1. (4.15–4.6 s) Dashboard shell blur-swaps in on **Data Augmentation**. // beat-locked: 4.23 s.
2. (5.0 s) Cursor clicks **Prepare dataset**; button → "Preparing…". Callout rises: real frame with green GT box → arrow "prepare" → white-rectangle mask `trainA/ · PNG mask`, caption **"Bounding box → binary mask"**. Text underneath dims while the callout is up. Done pill appears; callout leaves.
3. (7.0 s) Page scrolls to **Select Task**: Train CycleGAN selected with its real defaults, beside "Why Generative Data?".
4. (9.05 s) Cursor clicks **Inference** in the sidebar.
5. (10.0 s) Click drop-zone → frame preview appears, **Run Detection** turns cyan. (10.8 s) Click it → "Running Model..." + spinner "Processing images through PyTorch...".
6. (11.9–12.65 s) Result: frame returns, camera pushes in, two red boxes land. // beat-locked: 12.65 s. "Detection Results · 2 found" badge, score rows arrive one by one (both visible > 1.5 s).
7. (15.3 s) Camera pulls back to the full shell; stage blurs out into the outro at 16.0 s.
Audio: UI click on each simulated click (5.02 / 9.07 / 10.02 / 10.82), soft impact on the reveal, bell accent when the boxes land.
Transition mood: blur swap → Scene 3

### Scene 3 — Outro — 16.1–21.2 s
Ground returns to `#111827`. "Polyp Detection Dashboard" lands large with a blue Activity glyph (from the header) on the 16.34 s beat. Tagline in gray: "Deep Learning Polyp Detection with Generative Data Augmentation". Dry line: "Most theses end as a PDF." (17.91 s) / "This one has ten tabs." (18.44 s, accent blue). Footer: "FIB-UPC Bachelor's Thesis · github.com/cuberhaus/TFG" (18.8 s, held ~2.2 s as the last line).
Audio: one soft impact on the title; music fades out under the footer.

**Music mood for this video:** upbeat-dry, laid-back business bed
**Audio summary:** a quiet bed under a dry hook, crisp UI clicks through the simulated flow, one bell when the boxes land, a soft hit on the title, and a fade under the final joke.
