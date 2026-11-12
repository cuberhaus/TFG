# SPEC: "The fair test" (storyline A)

GOAL:    A viewer leaves knowing that fake polyps a real-data detector recognises did NOT make detection measurably
         better at full data, that the empty-frame control explains the false-alarm drop, and that this was measured on
         one laptop GPU. Honest, not defensive.
OUTPUT:  HyperFrames, 1920 x 1080, 30 fps, delivery quality, no voiceover.
TIME:    36.05 s x 30 = 1082 frames (as built; see "As built" at the end). Music Ende vol-12, 110 BPM, beat = 0.5455 s =
         16.36 frames. Beat times are read from `npx hyperframes beats`. Cuts on beats, layout change on bars.

LOOK:
  worlds   dark world (#111827, the app's own ground) for the hook, the strips and the end card;
           LIGHT world (#f3f4f6, "the lab report") for the scoreboard and the false-alarm number.
           One 0.4 s `power2.inOut` colour flood on the root ground moves between them (16.375 and 29.465).
  ink      dark world #ffffff / #9ca3af / #d1d5db; light world #111827 / #4b5563.
  accent   emerald (#34d399 on dark, #059669 on light) = the synthetic polyp, once per shot at most.
           The three series are fixed everywhere: real = gray (#9ca3af dark / #6b7280 light),
           control = light gray (#d1d5db dark / #9ca3af light), synthetic = emerald.
  type     Inter 900 for lines (hero words 180 px, lines 68-96 px, tracking -3%), Inter 700 labels 28-38 px,
           Inter 400 notes 30-34 px; tabular numerals on every counter. Nothing meant to be read under 28 px.
  medium   real endoscopy frames (HyperKvasir polyp-free background, painted polyp) at >= 40% of the frame width;
           flat colour fields; no gradients, no vignette, no glow, no grain.
  depth    frame shadow 0 34px 90px rgba(0,0,0,.5); borders 2 px #374151; strip dividers 4 px of the ground colour.

OBJECTS: the hero frame (hero-before / hero-after, 633x532, box [342,309,570,517]) is the spine: it opens the film,
         is the third strip's picture, and is carried into the end card. Real data modules only: every number below.
DATA (one module, `downstream/full/report.md`):
         real 85.8 | control 83.5 | synthetic 85.0 (AP50, mean of 3 seeds); synthetic - real = -0.8 pts, 95% interval
         -2.5 to +1.1; false alarms per polyp-free frame 0.24 | 0.05 | 0.10; 1,422 real frames, +1,422 extra, 2,666 steps.
         Earlier check: 87.3% (synthetic) vs 85.0% (real) found by a detector trained on real polyps only.

MOTION:  arrivals `power4.out` 0.35-0.5 s; camera/flood `power2.inOut`; exits `power3.in` 0.2 s (40-50% of entrances).
         Entrances vary: per-word blur (hook line 1, claim), whole-line blur (questions), horizontal whip (strips),
         scaleX with count-up (bars), count-down (false alarms). Titles are dead still once landed.
SHOTS (as built, global seconds, 30 fps):
  S1 open    0.00-6.555   hero frame left (780x655) with the green box drawing on by 0.43 and a "Synthetic polyp" chip;
                          right column stacks "This polyp is fake." (0.1) -> "Real-data detector finds 87.3%." (2.195, note
                          "Real polyps: 85.0%") -> "Does training on them help?" (4.375). Whip out left at 6.3.
  S2 arms    6.555-16.455 "Nine detectors." 180 px (+ "3 training sets x 3 seeds"); three full-height strips whip in on beats
                          8.735 (drop), 9.275, 9.835 with counters 1,422 / 2,844 / 2,844; seed squares 0/1/2 light on
                          12.015, 12.555, 13.105; line "Same recipe. Same 2,666 steps." at 13.645. Whip out at 16.115.
  S3 score   16.375-24.025 ground floods light; title "Detection score"; bars 85.8 / 83.5 / 85.0 land 17.465 / 18.555 / 19.655;
                          music ducks 20.15-21.35; "No measurable gain." at 20.745 with
                          "fake - real: -0.8 pts, 95% interval -2.5 to +1.1".
  S4 alarms  24.015-29.515 "Fewer false alarms." ; 0.24 counts down to 0.10 (25.1-26.4); "Empty frames alone: 0.05." at 27.285
                          with a bell (the turn); nothing else moves.
  S5 end     29.465-36.05 ground floods dark; "Found as often as real polyps." (30.015) + "Didn't improve detection." (31.105);
                          exit 32.5; end card 32.745: hero frame carried in, "Fake polyps, measured.", "Stable Diffusion + LoRA /
                          one RTX 5090 laptop GPU", scope note "Full-data setting - Kvasir-SEG + CVC-ClinicDB", footer; hold 3.3 s.
SOUND:   bed from 0, vol 0.34, 0.6 s fade-in, 1.6 s fade-out. Effects only on actions: box lock, line arrivals, whips,
         strip landings (soft impact on the drop), seed ticks, bar ticks, one bell on the 0.05 turn, soft impact on the title.
         Rotate drop_001/002/003. Bed ducks to 30% for 0.5 s before the verdict (20.15-21.35), then returns.
         Mastered: -16.0 LUFS integrated, true peak -1.2 dBFS.

BANNED:  the defaults in spec.md (void with small floating things, glow, gradients, cards in a row, docked logo,
         corner labels, everything fading up) plus: any claim of clinical realism; any wording that says the synthetic
         data "improves" or "boosts" detection; a bar chart axis that exaggerates a 2-point difference (bars start at 0);
         the real-polyp datasets' images (not shown). Endoscope on-screen text was meant to be banned too, but the hero
         frame still carries the source capture overlay (see REVIEW.md, P2): the earlier claim that it was clear was wrong.
QA:      t=0 (thumbnail = the 5.2 s settled hook frame, baked in), 3.0 (two lines stacked), 7.5 (hero words),
         11.0 (three strips), 14.5 (seeds lit + bottom line), 19.0 (two bars), 22.5 (verdict), 25.5 (count mid-way),
         28.0 (0.05 line), 31.0 (claim), 35.0 (end card). Phone sheet: every line readable at 360 px. check: 0 errors.
