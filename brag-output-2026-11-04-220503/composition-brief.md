# Hyperframes Composition Brief: TFG, round two (polyp synthesis)

## Objective
Create a short launch-style brag video for the diffusion-based polyp synthesis added to the TFG project: the thesis' failed CycleGAN result, then the repair, then the measurement.

## Output
- Composition directory: `brag-output-2026-11-04-220503/composition/`
- Rendered video: `brag-output-2026-11-04-220503/brag.mp4`
- Format: landscape — 1920x1080 @ 30 fps
- Duration: 24.0 s

## Source Material
- Project root: `/home/pol/cuberhaus/TFG`
- Primary files read: `README.md` (new "Diffusion-based polyp synthesis" section), `frontend/src/index.css`, `code/out/polyp_synth/eval/*/detector_check.json`, `eval/*/metrics.json`, `runs/v1/{args.json,metrics.jsonl}`, `TFG_DOC/tex/experimentacion.tex`, `TFG_DOC/png/cyclegan_result.png`
- Product name: TFG — Polyp Detection (bachelor's thesis), "round two"
- Tagline / strongest claim: a detector trained only on real polyps finds 87.3% of the synthetic ones (real frames: 85.0%; stock Stable Diffusion: 6.4%)
- Key visual moment: a real polyp-free frame; the real mask outline traces; the polyp paints itself in; the exact box draws on ("Label comes free")
- Copy that must appear verbatim:
  - Rectangle in. / Stripes out.
  - So I went back. / Keep the real colon. / Paint only the polyp. / Label comes free
  - Every polyp here is synthetic.
  - Found by a detector trained on real polyps only
  - Stock Stable Diffusion 6.4% · Real polyps 85.0% · Fine-tuned (LoRA) 87.3% · 160 held-out frames
  - From stripes to polyps. / Stable Diffusion + LoRA · 30 min · one laptop GPU / FIB-UPC Bachelor's Thesis · github.com/cuberhaus/TFG

## Creative Direction
- Tone preset: cinematic
- Creative direction: quiet before/after reveal: the thesis failure, then the repair
- Interpretation: wide frames, big type, long holds on the two images that matter, one dramatic reveal; restraint lives in the claims (no realism or accuracy-gain claims)
- Angle: the honest-redemption arc: the thesis' CycleGAN drew stripes (the thesis says so); round two keeps the real colon and paints only the polyp, then proves it with a real-data detector
- Hook: the thesis' white rectangle and beige stripes, "Rectangle in. Stripes out."
- Outro / punchline: "From stripes to polyps."
- Avoid: generic SaaS language, abstract filler visuals, claims of clinical realism, any frame with endoscope ID/name/date fields

## Visual Identity
- Background: `#111827`; panels `#1f2937`; borders `#374151`
- Text: `#ffffff`, secondary `#9ca3af` / `#d1d5db`
- Accent: emerald `#34d399` (synthetic / found), red `#f87171` (failure)
- Display font: Inter 900 (bundled); labels Inter 700; body Inter 400
- Visual references from the project: the frontend's gray-900/800/700 palette and Inter; the generator's green contact-sheet boxes

## Storyboard
Use the storyboard in `brag-plan.md` as the creative contract.

1. Rectangle in, stripes out — 3.27 s — thesis `real_A` and `fake_B` panels, two 2-word lines
2. The turn — 7.10 s — real frame, three stacked lines, mask outline, paint-in on the 8.74 s cue, box + "Label comes free"
3. Six at once — 4.92 s — 3x2 mosaic, one polyp per beat, caption
4. The proof — 5.46 s — three bars + counters on 16.38 / 17.47 / 18.56
5. From stripes to polyps — 3.25 s — mirrored hook layout, title, sub-line, footer

## Audio
- Audio role: cinematic support
- Audio arc: fade in; small thud on the stripes; bell on the paint-in; six soft ticks; two bar ticks then a bell on 87.3%; soft thud on the title; fade out
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (CC BY 4.0, Sascha Ende, ende.app)
- Music treatment: volume 0.34, fade-in 0.6 s, fade-out 1.6 s; master normalised after render to about -16.6 LUFS
- Music cue guidance: bundled preset (`vol-12.music-cues.json`). Locks: 8.74 (paint-in), 17.47, 18.56 (bars). Beat grid: mosaic 10.93-13.64 (every beat, non-text); bars every second beat (text)
- Audio-reactive treatment: subtle; bass energy of the track (extract-audio-data.py, 8 bands, 30 fps, sampled at 10 Hz) scales/fades the background glow stage by up to 5%. No waveform graphics
- Audio-coupled moments: stripes landing; paint-in; mosaic arrivals; bar landings; title
- SFX: `interface/drop_001-003`, `impact/impactSoft_medium_001`, `impactSoft_medium_002`, `impactBell_heavy_000`, `impactBell_heavy_003`, `casino/card-slide-1` (all CC0, Kenney)
- Audio files: copied into `composition/assets/`

## Hyperframes Instructions
Built with the `hyperframes` skill contract (data-* timing, one paused GSAP timeline per scene, sub-compositions in `compositions/`), no storyline/SPEC/usage/history/feedback/REVIEW steps. `hyperframes check` is the gate: 0 errors, 0 warnings, 0 info; contrast 27/27 text elements audited.
