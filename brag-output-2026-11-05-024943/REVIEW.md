# REVIEW: "The fair test" (storyline A)

Final file: `brag.mp4` (1920x1080, 30 fps, 36.1 s, H.264 + AAC, 3.4 MB). Poster: `brag.jpg` (also baked into frame 0).
Project: `composition/` (HyperFrames 0.9.13, pinned). Brief: `SPEC.md`. Options: `STORYLINES.md`. Copy: `share-copy*.{txt,md}`.

## Creative calls

- **Storyline A, "The fair test"**, picked by the owner from three. The film is a controlled experiment told in order:
  the fake polyp, the question, the nine runs, the score, the false-alarm number, the claim, the end card.
- **Honest result up front.** The film says "No measurable gain." and shows the delta with its 95% interval
  (-0.8 pts, -2.5 to +1.1). Bars start at 0, so a 2-point gap is not exaggerated.
- **The one good-looking number is shown with its control.** Synthetic data cut false alarms 0.24 -> 0.10 per polyp-free
  frame, but empty frames alone reached 0.05, so the film ends that scene on the control, with a bell. Nobody reading the
  film can take the false-alarm drop as a win for the polyps. The control also costs 2.3 AP50 points (a significant drop);
  that is visible as the 83.5 bar in the score scene, but it is not called out in words.
- **Closing line is a pair that both hold:** "Found as often as real polyps." (87.3% vs 85.0%, real-only detector) and
  "Didn't improve detection."
- **Scope on screen:** "Full-data setting - Kvasir-SEG + CVC-ClinicDB" on the end card, plus "one RTX 5090 laptop GPU"
  (owner request). The film does not mention the low-data case, which was not tested.
- **Two worlds:** dark (the app's own ground) for the hook, the nine runs and the end card; light ("the lab report") for the
  score and false-alarm numbers. Series colours are fixed: real gray, control light gray, synthetic emerald.
- **Poster:** frame 0 is the settled hook frame from 5.2 s, baked in, so the thumbnail already carries the claim and the
  question. In the composition itself frame 0 is the hero frame alone.
- **Not built:** B "The pull-back" (16:9, one continuous take) and C "The control" (9:16, type-led). Both are written up in
  `STORYLINES.md` and can be built from the same data module.

## What was measured, what was looked at, what nobody checked

Measured:
- `hyperframes check`: 0 errors, 0 warnings on the final timeline (earlier `line_unread` x3 and `dead_stretch` were fixed by
  retiming, not suppressed).
- Rendered file audio: integrated **-16.0 LUFS** (raw mix was -23.6), true peak **-1.2 dBFS**, loudness range 2.0 LU. The
  normaliser had to fall back to dynamic mode because +7.6 dB of gain would have crossed the -1.5 dBFS peak target, so the
  range is tighter than the raw mix (5.1 LU).
- Duck before the verdict: bed RMS drops ~9 dB at 20.25-20.5 s, impact at 20.75 s, bed returns by 21.25 s.
- Frame 0 of the final file matches the 5.2 s poster frame (mean pixel difference 1.3, compression noise); frame 1 is the real opening.

Looked at (contact sheets from `snapshot --review`: beat frames, cut strips, 360 px phone width) and each timed scene at its
start, middle and settled moment. The phone sheet was checked for readable lines.

Not checked:
- **The sound was measured, not heard.** Levels, ducking and hits are right by the numbers; whether the mix feels right is untested.
- Nobody watched it at full speed on a phone, and no one but the model has reviewed it.
- The Studio preview server was started for editing (`http://localhost:3002`); I did not scrub it there, only `check`, snapshots and the rendered file.

## Weakest shot and what I would change

- **P2, t=0-6.5 s, hero frame:** the HyperKvasir background still carries the source capture overlay (`Mbit/s / Drop / Buf`,
  a timestamp at top-left, and the black position box at bottom-left). It is real, small, and illegible at normal size, but
  `SPEC.md` originally claimed the frames were clear of on-screen text, which was wrong. Fix with more time: pick a
  showcase frame without the overlay (several in `code/out/polyp_synth/synth_v1_showcase/` are cleaner) and update the box
  coordinates in `open.html`, `arms.html` and `outro.html`, or crop the frame to its right 420 px.
- **P2, t=29-32 s:** "Found as often as real polyps." follows two scenes about a different test (the AP50 comparison). The 87.3% /
  85.0% figure that supports it appears only in the hook (2-4 s), so the closing claim leans on a number viewers saw 25 s
  earlier. Fix with more time: repeat "87.3% vs 85.0%" as a small note under the claim.
- **P2, t=20.7 s:** the verdict "No measurable gain." and the delta line are read together; at phone width the interval line is small.

## Warnings kept on purpose

None. The final check is clean.

## Known limits of the film's claim (not film defects)

- Full-data only: the generator and the gate that filters synthetic polyps were trained on all the real training frames.
- 3 seeds on 160 held-out frames is noisy; the synth - real interval comfortably includes zero in both directions.
- The film says the data did not help; it does not say it cannot help. The low-data case and the ungated `synth_all` ablation
  were not run.
