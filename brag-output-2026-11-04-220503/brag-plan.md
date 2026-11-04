# Brag Plan: TFG, round two — polyp synthesis that finally works

> Sequel to the earlier brag in `brag-output/` (the dashboard video, "Not enough polyps. So we taught a GAN to draw more."). That one deliberately made no claim about image quality. This one is the follow-up: going back to the thesis' weakest result and fixing it.

## 9-question rubric (Step 1)

1. **What is the app?** A FIB-UPC bachelor's thesis on polyp detection (Faster R-CNN / RetinaNet / SSD) whose generative-augmentation part (CycleGAN: bounding-box mask in, colonoscopy frame out) did not work. Round two replaces it with Stable Diffusion inpainting + a LoRA that keeps the real colon pixel-for-pixel and paints only the polyp.
2. **Funniest / most impressive claim.** The thesis' own failure figure: *a white rectangle in, beige stripes out.* And the fix: a held-out check where a detector trained **only on real polyps** finds **87.3%** of the synthetic ones vs **85.0%** of real ones.
3. **Visual hook.** The thesis' real `real_A` (white rectangle on black) next to its `fake_B` (beige stripes with a glowing blob), from `TFG_DOC/png/cyclegan_result.png`, then the same idea done right: a real frame where a polyp paints itself in inside its mask outline.
4. **What to show from the actual output.** Real before/after frames from `code/out/polyp_synth/synth_v1_showcase/` (`Before/` vs `Images/`, with the exact box from `Annotations/`), and the held-out numbers from `eval/*/detector_check.json`.
5. **Shortest satisfying video.** 24 s (hook 3.3 s, turn 7.1 s, mosaic 4.9 s, proof 5.5 s, outro 3.3 s). As built: the proof scene got ~0.6 s more hold so 87.3% can land.
6. **Tone.** Preset `cinematic`; direction: *quiet before/after reveal: the thesis failure, then the repair.* Restraint in claims; scale and holds do the work.
7. **Audio.** Steady, clean music bed (vol-12, ~110 BPM), three big accents only (hero reveal, hero number, title), card/drop ticks on the mosaic arrivals, subtle audio-reactive background glow.
8. **Share caption.** See `share-copy.txt`.
9. **User flow worth showing.** This is a pipeline, not a UI: **real polyp-free frame → mask outline → polyp painted in → box comes free.** Then six results arriving one by one, then the measurement.

## What is this app?
A bachelor's thesis on polyp detection, now with a diffusion-based way to make labelled synthetic polyps: take a real polyp-free colonoscopy frame, choose where the polyp goes, and let Stable Diffusion inpainting (+ a 30-minute LoRA) paint only that region. The bounding box is exact because the mask was chosen.

## The angle
The honest-redemption arc. The thesis' CycleGAN was asked to turn a rectangle into a whole colonoscopy frame and drew stripes (the figure is in the thesis). Round two stops asking the model to invent the colon: keep the real tissue, paint only the polyp. The video states the failure plainly, shows the repair working on real frames, then proves it with the one number that matters (a detector that has only seen real polyps recognizes the synthetic ones).

Deliberate restraint: nothing on screen claims clinical realism or a detection-accuracy gain. The claim is exactly what was measured: *how often a real-data detector finds the polyp.*

## Hook (first 2-3 seconds)
Frame 0 is already on screen: the thesis' white rectangle (`real_A`) on the left. The line **"Rectangle in."** lands; at the 1.64 s beat the `fake_B` panel (beige stripes) and **"Stripes out."** (in red) land. Two words each, held long enough to read.

## Key moments (the middle)
- **The turn (3.3-10.4 s).** A real polyp-free frame (`Before/synth/00296`) comes up on the right while three short lines stack on the left: **"So I went back."**, **"Keep the real colon."**, **"Paint only the polyp."** A green outline traces the real mask, then the polyp paints itself in from the centre of the mask (everything outside the mask is pixel-identical, so only the polyp changes), and the exact box draws on with the chip **"Label comes free"**. This is the strongest moment: the paint-in starts 0.12 s before the 8.74 s strong cue so the cue lands as the polyp becomes visible.
- **Six at once (10.4-15.3 s).** A 3x2 mosaic of six synthetic polyps (16:9 crops) arrives one per beat, each with its green box drawing on. Caption: **"Every polyp here is synthetic."**
- **The proof (15.3-20.75 s).** Kicker **"Found by a detector trained on real polyps only"**; three bars arrive two beats apart and count up: **Stock Stable Diffusion 6.4%**, **Real polyps 85.0%**, **Fine-tuned (LoRA) 87.3%** (the last in emerald, with a bell). Footnote: **160 held-out frames**.

## Outro / punchline
**"From stripes to polyps."** under the same two-panel layout as the hook, resolved: the thesis' stripes panel on the left, an arrow, a synthetic polyp frame on the right. Sub-line: *Stable Diffusion + LoRA · 30 min · one laptop GPU.* Footer: `FIB-UPC Bachelor's Thesis · github.com/cuberhaus/TFG`.

## User flow worth showing
Entry: a real polyp-free frame. Key action: a polyp is painted into a chosen mask. Result: a realistic frame **plus** its exact bounding box, then the measurement that says a real-data detector finds it.

## Tone
- Preset: `cinematic`
- Creative direction: quiet before/after reveal: the thesis failure, then the repair
- Interpretation: wide frames, big type, long holds on the two images that matter, one dramatic reveal (the polyp painting itself in). Short declarative lines, no hype adjectives. The restraint is in the claims, the drama is in the images.

## Format: landscape — 1920x1080
## Duration: 24.0 s

## Visual identity (from the project)
- Background: `#111827` (Tailwind gray-900, `body` in `frontend/src/index.css`)
- Panels: `#1f2937` (gray-800), borders `#374151` (gray-700)
- Text: `#ffffff`; secondary `#9ca3af` (gray-400)
- Accents: emerald `#34d399` (synthetic / found / the box colour used in the generator's contact sheets), red `#f87171` (the failure: stripes, untuned model)
- Display font: Inter 900 (hero lines) / Inter 700 (labels), the app's `font-family`
- Body font: Inter 400
- Strongest visual element: the real colonoscopy frame with the polyp painting itself in, and the thesis' own stripes figure as the "before"

## Real vs. stand-in material (nothing personal on screen)
- **Real:** the thesis figure panels (`real_A`, `fake_B`) from `TFG_DOC/png/cyclegan_result.png`; the generated frames, the polyp-free originals, the boxes and the mask outline (from `synth_v1_showcase`); all numbers (from `eval/v1_euler16/detector_check.json`, `eval/base/detector_check.json`, `runs/v1/metrics.jsonl`).
- **Privacy:** several source frames carry endoscope on-screen text (`ID:`, `Name:`, `Sex:`, exam dates). Only frames/crops without those fields are used; crops are chosen to exclude them.
- **Stand-ins:** none. There are no names, emails, hostnames or credentials in the video.

## Share copy (draft)
My bachelor's thesis drew stripes instead of polyps, so I went back to it: Stable Diffusion inpainting + a 30-minute LoRA now paints polyps a detector trained only on real ones finds 87.3% of the time (real frames: 85.0%).

## Audio direction
- Role: cinematic support: a steady bed, three big accents, small motion-matched ticks
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (steady and clean, 109.96 BPM)
- Music treatment: starts at 0 s, volume ~0.32, 0.6 s fade-in, fade-out across the last 1.4 s under the title
- Music cue guidance: preset read (`vol-12.music-cues.md`). Strong cues locked: **8.74 s** (polyp paints in), **17.47 s** (real-polyps bar), **18.56 s** (fine-tuned bar / 87.3%). Beat grid for sequential events: mosaic items on 10.93, 11.46, 12.02, 12.55, 13.11, 13.64 (non-text, so every beat is fine); bars every two beats on 16.38, 17.47, 18.56 (text labels, so every other beat). Scene cuts land on beats 3.27, 10.37, 15.29; the outro title lands on the 20.75 s beat.
- Audio-reactive treatment: subtle; a radial emerald glow behind the stage breathes with bass energy (build-time sampled from the music, no waveform or equalizer graphics).
- SFX posture: sparse, motion-matched, low high-frequency-risk picks. Chosen after the animation exists.
- Audio-coupled moments: stripes panel landing (soft impact), polyp paint-in (bell), mosaic arrivals (soft card/drop per item), bars landing (tick each, bell on 87.3%), title (soft impact).
- Restraint rule: no SFX on the text lines of scene 2; no stacked cues; nothing above the music's presence except the three accents.

## Storyboard

### Scene 1 — Rectangle in, stripes out — 3.27 s (0-3.27)
Thesis figure panels side by side on the dark ground: `real_A` (white rectangle) on the left, `fake_B` (beige stripes) on the right with a thin arrow between. Small kicker above: "Bachelor's thesis · CycleGAN". **"Rectangle in."** under the left panel at 0.3 s; **"Stripes out."** (red) under the right panel at 1.64 s. Both held until the cut.
Sequential/interaction: yes: the left panel + line land first, the right panel + line land one beat-and-a-half later (two readable beats).
Audio intent: dry and a little deflating; the second hit is the joke.
Audio-coupled idea: soft drop on the first panel, a soft impact when the stripes panel lands.
Music: bed fades in from silence.
Transition mood: soft blur swap → Scene 2

### Scene 2 — The turn — 7.10 s (3.27-10.37)
Left column, three lines in sequence (each replaces/overlaps the prior): **"So I went back."** (3.82), **"Keep the real colon."** (5.34), **"Paint only the polyp."** (7.09). Right: the real polyp-free frame (`Before/synth/00296`) in a rounded panel; a green outline traces the real mask (5.7-6.9 s); at 8.74 s (strong cue) the polyp paints itself in from the mask's centre; the green box draws on and the chip **"Label comes free"** appears (9.3 s). Slow push-in on the frame the whole scene.
Sequential/interaction: yes: three lines stacking, then outline → paint-in → box → chip.
Audio intent: the held breath, then the release.
Audio-coupled idea: nothing on the lines (restraint); a bell exactly on the 8.74 s paint-in; a soft tick when the box draws.
Music: steady bed, slightly under.
Transition mood: whip (horizontal smear, the mosaic arrives from the right) → Scene 3

### Scene 3 — Six at once — 4.92 s (10.37-15.29)
A 3x2 mosaic of six synthetic polyps (16:9 crops, 560x315 each) arrives one per beat (10.93, 11.46, 12.02, 12.55, 13.11, 13.64), each with its green box drawing on half a beat later. Caption **"Every polyp here is synthetic."** arrives with the first cell and holds to the end.
Sequential/interaction: yes: six cells one by one on consecutive beats.
Audio intent: momentum; the pace picks up.
Audio-coupled idea: a soft card/drop on every arrival (non-text, rhythmic by intent).
Music: bed carries.
Transition mood: dip then blur swap → Scene 4

### Scene 4 — The proof — 5.46 s (15.29-20.75)
Kicker **"Found by a detector trained on real polyps only"**. Three horizontal bars arrive two beats apart and count up: **Stock Stable Diffusion 6.4%** (red), **Real polyps 85.0%** (gray, reference), **Fine-tuned (LoRA) 87.3%** (emerald, glow). Footnote: "160 held-out frames".
Sequential/interaction: yes: bars one by one with counters.
Audio intent: measured, then a payoff.
Audio-coupled idea: a tick per bar start; a bell when 87.3% completes.
Music: bed; the reveal at 18.56 s is the strong cue.
Transition mood: slow blur swap → Scene 5

### Scene 5 — From stripes to polyps — 3.25 s (20.75-24.00)
Two panels (thesis stripes left, a synthetic polyp frame right, emerald arrow between) above the title **"From stripes to polyps."** Sub-line: "Stable Diffusion + LoRA · 30 min · one laptop GPU". Footer: "FIB-UPC Bachelor's Thesis · github.com/cuberhaus/TFG".
Sequential/interaction: none (the title lands dead still; everything else holds).
Audio intent: resolve; the music fades under the last line.
Audio-coupled idea: one soft impact on the title landing.
Music: fade-out across the last 1.4 s.
Transition mood: end

## As built (deviations from the first draft)
- Total 24.0 s (was 23.5): proof scene held longer; outro title is "From stripes to polyps." (reads better than an arrow glyph).
- Scene 2 paint-in slowed to ~0.8 s and the box/chip re-timed so "Label comes free" holds >1 s (Hyperframes craft check flagged 0.66 s).
- Audio mastered after render to about -16.6 LUFS (raw mix was -24 LUFS).
- Poster baked as frame 0 (mosaic at 14.3 s).

**Music mood for this video:** steady, clean, cinematic-leaning
**Audio summary:** a quiet bed fades in; the stripes land with a small deflating thud; the polyp's paint-in gets the first bell on a strong beat; six soft ticks carry the mosaic; two bar ticks lead to a bell on 87.3%; one soft thud for the title; the bed fades out.
