"""Prepare image assets for the brag composition (run with the tfg-gen venv).

Reads the real thesis figure and the generated showcase frames; writes web-sized
crops + a JSON with box/mask geometry into composition/assets/img/.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]  # repository root (this script lives one level below it)
SHOW = ROOT / "code/out/polyp_synth/synth_v1_showcase"
OUT = Path(__file__).resolve().parent / "composition/assets/img"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- 1. thesis figure panels (real_A rectangle mask, fake_B stripes) ----------
fig = Image.open(ROOT.parent / "TFG_DOC/png/cyclegan_result.png").convert("RGB")
arr = np.asarray(fig).astype(int)
H, W, _ = arr.shape
lum = arr.mean(axis=2)
row = lum[H // 2]  # through the middle of the panels
runs, start = [], None
for x in range(W):
    dark = row[x] < 235
    if dark and start is None:
        start = x
    if not dark and start is not None:
        if x - start > 150:
            runs.append((start, x))
        start = None
print("panel runs (x):", runs[:6], "fig", (W, H))
col = lum[:, (runs[0][0] + runs[0][1]) // 2]
dark = col < 235
best, s0 = (0, 0), None
for y in range(len(dark) + 1):
    d_ = dark[y] if y < len(dark) else False
    if d_ and s0 is None:
        s0 = y
    if not d_ and s0 is not None:
        if y - s0 > best[1] - best[0]:
            best = (s0, y)
        s0 = None
y0, y1 = best
print("panel y:", y0, y1)
for name, (a, b) in zip(["thesis-rect", "thesis-stripes"], runs[:2]):
    crop = fig.crop((a, y0, b, y1))
    side = min(crop.size)
    crop = crop.crop((0, 0, side, side)).resize((680, 680), Image.LANCZOS)
    crop.save(OUT / f"{name}.jpg", quality=92)
    print("wrote", name, crop.size)

# ---------- 2. hero frame ----------
HERO = "00296"
before = Image.open(SHOW / f"Before/synth/{HERO}.jpg").convert("RGB")
after = Image.open(SHOW / f"Images/synth/{HERO}.jpg").convert("RGB")
before.save(OUT / "hero-before.jpg", quality=94)
after.save(OUT / "hero-after.jpg", quality=94)
box = list(map(int, (SHOW / f"Annotations/synth/{HERO}.txt").read_text().split()[1:5]))
mask = np.asarray(Image.open(SHOW / f"Masks/synth/{HERO}.png").convert("L")) > 127
cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
c = max(cs, key=cv2.contourArea)
c = cv2.approxPolyDP(c, 1.2, True)[:, 0, :]
path = "M" + " L".join(f"{x},{y}" for x, y in c) + " Z"
ys_, xs_ = np.where(mask)
geom = {
    "hero": {
        "size": before.size,
        "box": box,
        "mask_path": path,
        "mask_center": [int(xs_.mean()), int(ys_.mean())],
        "mask_radius": int(np.hypot(xs_.max() - xs_.min(), ys_.max() - ys_.min()) / 2) + 14,
        "diff_outside_mask": float(
            np.abs(np.asarray(before).astype(int) - np.asarray(after).astype(int))[~mask].mean()
        ),
    }
}
print("hero geom", {k: v for k, v in geom["hero"].items() if k != "mask_path"})

# ---------- 3. mosaic candidates: 16:9 crops around each polyp box ----------
ids = sys.argv[1:] or ["00027", "00067", "00099", "00132", "00174", "00221"]
RATIO = 16 / 9
for i in ids:
    im = Image.open(SHOW / f"Images/synth/{i}.jpg").convert("RGB")
    x1, y1, x2, y2 = map(int, (SHOW / f"Annotations/synth/{i}.txt").read_text().split()[1:5])
    w, h = im.size
    cw = w  # full width, 16:9 height
    ch = int(round(cw / RATIO))
    cy = (y1 + y2) / 2
    top = int(round(min(max(cy - ch / 2, 0), h - ch)))
    crop = im.crop((0, top, cw, top + ch))
    crop.save(OUT / f"mosaic-{i}.jpg", quality=92)
    geom[i] = {"box": [x1, y1 - top, x2, y2 - top], "crop": [cw, ch], "src": [w, h]}
# square polyp crop for the outro (right edge of the hero frame)
after.crop((101, 0, 633, 532)).resize((680, 680), Image.LANCZOS).save(OUT / "outro-polyp.jpg", quality=92)
json.dump(geom, open(OUT / "geometry.json", "w"), indent=1)
print("done", list(geom))
