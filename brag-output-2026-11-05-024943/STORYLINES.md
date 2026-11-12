# Storylines: round three, "does it actually help detection?"

Sequel to `brag-output-2026-11-04-220503/` (kept untouched; its brag.mp4 stays as the backup). That film proved the
synthetic polyps look right to a detector. This one answers the question a reviewer asks next, honestly.

## The one thing
**Fake polyps a detector recognises did not make it measurably better at full data.** The one thing to remember:
*the control run is what turned a demo into a result.*
Spine: one real synthetic polyp frame travels from the first shot to the scoreboard.

## Facts every claim traces to (`code/out/polyp_synth/downstream/full/report.md`, `eval/*/detector_check.json`)
- Thesis CycleGAN figure: white rectangle in, beige stripes out (`TFG_DOC/png/cyclegan_result.png`).
- Round two: Stable Diffusion 1.5 inpainting + a 30-minute LoRA on one RTX 5090 laptop GPU. Real pixels kept, only the
  polyp is painted, the box is the mask.
- Check 1 (160 held-out frames, detector trained on real polyps only): stock SD 6.4%, real 85.0%, fine-tuned 87.3%.
- Check 2 (this experiment): 9 Faster R-CNN detectors = 3 training sets x 3 seeds, equal 2,666 steps, scored on
  160 validation frames + 200 polyp-free frames.
  - AP50: real only 85.8 +- 0.6 | + unpainted frames as negatives (control) 83.5 +- 1.0 | + synthetic polyps 85.0 +- 1.4
  - synthetic - real: -0.8 pts, 95% interval -2.5 to +1.1 (spans zero: no measurable gain)
  - control - real: -2.3 pts, interval -4.3 to -0.3 (excludes zero)
  - false alarms per polyp-free frame: real 0.24 | control 0.05 | synthetic 0.10
- Scope, to be stated on screen or in the caption: full-data setting only; Kvasir-SEG + CVC-ClinicDB, not clinical video.

## A (recommended): "The fair test", a question with the input, split screen to one
16:9, 30-36 s. The viewer is a reviewer who just saw a fake polyp and asks the obvious thing. The film puts the real
frame inside the question, splits into the three training sets, and collapses them into one scoreboard that says no.
- First two seconds: frame 0 is already the painted-in hero frame with its green box; "This polyp is fake." lands at
  0.3 s; "Does it help?" lands at 1.6 s.
- Beats: hook (fake polyp, the question) -> tension (a 1.5 s flash of last round's stripes) -> reveal (three panes:
  real / real + empty frames / real + synthetic) -> proof (nine runs tick in, panes collapse into three bars) ->
  payoff ("No measurable gain": the synthetic bar settles beside the real one, the interval bracket spans zero) ->
  claim (false alarms 0.24 -> 0.10, with the control's 0.05 beside it) -> end card.
- Look: project ground #111827, Inter 900/700, emerald for synthetic, red only for the stripes, real frames large.
- Sound: Ende vol-12 bed, soft ticks as runs land, the bed drops for half a second before the verdict.

## B: "The pull-back", the result first, one continuous take
16:9, 28-34 s. The finished polyp paints itself in on a tight macro; the camera never cuts. It pulls back to the
1,800-frame contact sheet it came from, back again to last round's stripes, then travels along to the nine-detector
scoreboard and the verdict. Each stop is one idea.
- First two seconds: frame 0 is the polyp, box drawn, camera already easing out; "synthetic" tag at 0.4 s; the
  neighbouring frames of the sheet are revealed at 1.6 s.
- Beats: hook (polyp, result first) -> reveal (1,800 frames) -> tension (the stripes, red) -> proof (travel to the
  scoreboard, bars) -> payoff (verdict) -> claim -> end card.
- Look: the contact sheet is the world's material (real generated frames as the ground), same palette.
- Sound: Ende vol-12 bed, one low swell on each pull-back, ticks on the bars.

## C (the bold one): "The control", number lands, cut on the beat, 9:16
1080x1920, 16-20 s, muted-friendly. The film is about the control arm: the empty frames alone cut false alarms
0.24 -> 0.05, the fake polyps got 0.10, and detection did not improve. Giant type on colour fields cutting on the beat,
each card backed by one real frame or number. Lesson on the last card: always run the control.
- First two seconds: frame 0 is a full-bleed synthetic polyp frame with its box and "FAKE POLYPS." in giant type;
  a hard cut on the first beat to "Better detector?"
- Beats: hook -> question -> "NO." on a red field with "-0.8 pts, inside the noise" -> "But false alarms fell 0.24 -> 0.10"
  -> "Empty frames alone: 0.05" (the number that lands) -> "Run the control." -> end card.
- Look: colour fields (#111827, emerald, red), Inter 900 at roughly 1.35x landscape size, inside the 168-1330 px band.
- Sound: Ende vol-12 bed with hits on the cuts; everything readable without sound.

## Assumptions
Free material only: Ende vol-12 music (CC BY 4.0) and Kenney SFX (CC0), both already in the repo, no voiceover, no
paid items. Each outro says "one RTX 5090 laptop GPU". New files live under this folder; the earlier videos are not touched.
