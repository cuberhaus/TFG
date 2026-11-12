# Share copy: "The fair test"

All numbers come from `code/out/polyp_synth/downstream/full/report.md` (Faster R-CNN ResNet50-FPN v2, 3 arms x 3
seeds, equal 2,666 steps, 160 held-out frames with polyps + 200 polyp-free frames). Every version states the scope:
**full-data setting only**. None claims that synthetic data improves detection, because the experiment did not show it.

## Short (X / Bluesky / Mastodon, under 280 characters)

Do the fake polyps from my thesis actually help a detector? I trained 9 Faster R-CNNs on one RTX 5090 laptop GPU to
find out. At full data: no measurable gain (AP50 85.0 vs 85.8). Fewer false alarms, but empty frames alone did that
better. github.com/cuberhaus/TFG

## LinkedIn / longer

My thesis can now paint polyps into colonoscopy frames (Stable Diffusion inpainting + a small LoRA). A detector
trained only on real polyps finds 87.3% of the fake ones, versus 85.0% of real ones, so they look real to it.

That is not the same as being useful, so I ran the fair test: 9 detectors (real only / real + empty frames as a control /
real + fake polyps) x 3 seeds, same recipe, same number of steps, all on one RTX 5090 laptop GPU.

At full data the answer is no measurable gain: AP50 85.8 for real only vs 85.0 with the fake polyps (difference -0.8,
95% interval -2.5 to +1.1). The fake polyps did cut false alarms on polyp-free frames from 0.24 to 0.10 per frame, but
the control did better still (0.05), so that effect is just "the model saw more empty frames", not the polyps.

What I did not test: the low-data case, where synthetic data is supposed to matter most. That needs the generator
retrained on a subset, and it is the obvious next step.

github.com/cuberhaus/TFG

## One-liner for a README or CV

Thesis: a polyp generator a real-data detector can't tell from real polyps (87.3% vs 85.0%), plus a controlled
9-run experiment showing it gives no measurable detection gain at full data.
