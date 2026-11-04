"""Insert synthetic polyps into polyp-free colonoscopy frames.

Usage (from ``code/src``)::

    python -m polyp_synth.generate --lora ../out/polyp_synth/runs/v1/final \
        --frames ../data/polyp_synth/raw/negatives/cecum --n 200 \
        --out ../out/polyp_synth/synth_v1

Output layout mirrors LDPolypVideo so ``clases.custom_dataset.CustomDataset``
can load it directly::

    <out>/Images/synth/00000.jpg
    <out>/Annotations/synth/00000.txt   # "1\\nx1 y1 x2 y2"
    <out>/Masks/synth/00000.png

Because the polyp is painted *inside* a mask we choose, its bounding box is
known exactly -- the label is free and exact.
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image

from polyp_synth.data import (
    CAPTIONS, PolypInpaintDataset, blank_scope_overlay, fov_mask, index_samples, make_hole, mask_bbox, split_samples,
)
from polyp_synth.pipeline import DEFAULT_PROMPT, DEFAULT_STEPS, composite, grid, inpaint_polyp, load_pipeline, save_jpg

SIZE = 512


def shape_ok(mask: np.ndarray, min_solidity: float = 0.85, max_aspect: float = 2.2) -> bool:
    """Reject thin / ragged masks: they yield fold-like blobs rather than polyps."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    if area < 50 or hull <= 0:
        return False
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    aspect = max(w, h) / max(1.0, min(w, h))
    return area / hull >= min_solidity and aspect <= max_aspect


def on_mucosa(crop512: np.ndarray, fov512: np.ndarray, hole: np.ndarray, min_ratio: float = 0.7) -> bool:
    """The hole must sit on lit mucosa, not on the dark lumen."""
    lum = crop512.astype(np.float32).mean(2)
    ref = lum[fov512 > 0].mean()
    return bool(hole.any() and lum[hole > 0].mean() >= min_ratio * ref)


def place_random_mask(mask512: np.ndarray, fov512: np.ndarray, rng: np.random.Generator, scale=(0.8, 1.25), tries=60):
    """Rotate/scale a real polyp mask and drop it at a random valid location.

    Returns a 512x512 uint8 mask, or None if no valid placement was found.
    """
    bb = mask_bbox(mask512)
    if bb is None:
        return None
    m = mask512[bb[1] : bb[3], bb[0] : bb[2]].astype(np.uint8)
    for _ in range(tries):
        ang = rng.uniform(0, 360)
        sc = rng.uniform(*scale)
        h, w = m.shape
        diag = int(np.ceil(np.hypot(h, w) * sc)) + 4
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        M[0, 2] += diag / 2 - w / 2
        M[1, 2] += diag / 2 - h / 2
        r = cv2.warpAffine(m, M, (diag, diag), flags=cv2.INTER_LINEAR)
        r = (r > 0.5).astype(np.uint8)
        rb = mask_bbox(r)
        if rb is None:
            continue
        r = r[rb[1] : rb[3], rb[0] : rb[2]]
        rh, rw = r.shape
        if rh >= SIZE * 0.9 or rw >= SIZE * 0.9:
            continue
        # centre must keep the whole bbox inside the FOV (eroded by a margin)
        margin = max(6, int(0.02 * SIZE))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        safe = cv2.erode(fov512, k)
        ys, xs = np.where(safe > 0)
        if len(xs) == 0:
            return None
        for _ in range(20):
            j = rng.integers(len(xs))
            cx, cy = xs[j], ys[j]
            x0, y0 = cx - rw // 2, cy - rh // 2
            if x0 < 0 or y0 < 0 or x0 + rw > SIZE or y0 + rh > SIZE:
                continue
            # every polyp pixel must be inside the field of view
            if (fov512[y0 : y0 + rh, x0 : x0 + rw][r > 0] == 0).any():
                continue
            out = np.zeros((SIZE, SIZE), np.uint8)
            out[y0 : y0 + rh, x0 : x0 + rw] = r
            return out
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--frames", nargs="+", required=True, help="dir(s) with polyp-free frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--dilate", type=float, default=0.02, help="hole margin around the polyp, fraction of crop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--style", default="kvasir", choices=list(CAPTIONS))
    ap.add_argument("--min-solidity", type=float, default=0.85)
    ap.add_argument("--max-aspect", type=float, default=2.2)
    ap.add_argument("--min-lum-ratio", type=float, default=0.7, help="hole luminance / FOV luminance; rejects dark lumen")
    ap.add_argument("--min-area", type=float, default=0.01, help="min polyp area as a fraction of the 512 crop")
    ap.add_argument("--max-area", type=float, default=0.20, help="max polyp area as a fraction of the 512 crop")
    ap.add_argument("--save-before", action="store_true", help="also save the original polyp-free frame to Before/synth/")
    ap.add_argument("--sheet", type=int, default=8, help="rows in the before/after contact sheet (0 = skip)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files = sorted(sum([glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")) for d in args.frames], []))
    assert files, "no frames found"
    train_s, _ = split_samples(index_samples())
    bank = PolypInpaintDataset(train_s, size=SIZE, train=True)  # random crops of real masks -> realistic size distribution
    pipe = load_pipeline(args.lora)
    prompt = CAPTIONS[args.style]

    for sub in ("Images", "Annotations", "Masks") + (("Before",) if args.save_before else ()):
        os.makedirs(os.path.join(args.out, sub, "synth"), exist_ok=True)
    sheet_rows = []
    done = 0
    while done < args.n:
        f = files[rng.integers(len(files))]
        frame = blank_scope_overlay(np.array(Image.open(f).convert("RGB")))
        H, W = frame.shape[:2]
        side = min(H, W)
        x0 = int(rng.integers(0, W - side + 1))
        y0 = int(rng.integers(0, H - side + 1))
        crop = frame[y0 : y0 + side, x0 : x0 + side]
        crop512 = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        fov = fov_mask(crop512)
        item = bank.get(int(rng.integers(len(bank))), rng)
        if not (args.min_area <= item["mask"].mean() <= args.max_area):
            continue
        if not shape_ok(item["mask"], args.min_solidity, args.max_aspect):
            continue
        m = place_random_mask(item["mask"], fov, rng)
        if m is None:
            continue
        hole = make_hole(m, max(1, int(round(args.dilate * SIZE))))
        if not on_mucosa(crop512, fov, hole, args.min_lum_ratio):
            continue
        gen = inpaint_polyp(pipe, crop512, hole, prompt=prompt, seed=int(rng.integers(1 << 30)), steps=args.steps, guidance=args.guidance)

        # composite at native resolution so everything outside the hole stays bit-exact
        hole_full = cv2.resize(hole.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
        gen_full = cv2.resize(gen, (side, side), interpolation=cv2.INTER_CUBIC)
        region = (hole_full > 0.5).astype(np.uint8)
        out_frame = frame.copy()
        out_frame[y0 : y0 + side, x0 : x0 + side] = composite(crop, gen_full, region, feather=side / SIZE * 2.0)

        mf = cv2.resize(m.astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST)
        bb = mask_bbox(mf)
        bx = (bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0)
        name = f"{done:05d}"
        Image.fromarray(out_frame).save(os.path.join(args.out, "Images", "synth", name + ".jpg"), quality=95)
        if args.save_before:
            Image.fromarray(frame).save(os.path.join(args.out, "Before", "synth", name + ".jpg"), quality=95)
        with open(os.path.join(args.out, "Annotations", "synth", name + ".txt"), "w") as fh:
            fh.write(f"1\n{bx[0]} {bx[1]} {bx[2]} {bx[3]}\n")
        full_mask = np.zeros((H, W), np.uint8)
        full_mask[y0 : y0 + side, x0 : x0 + side] = mf * 255
        Image.fromarray(full_mask).save(os.path.join(args.out, "Masks", "synth", name + ".png"))

        if len(sheet_rows) < args.sheet:
            viz = out_frame.copy()
            cv2.rectangle(viz, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
            t = lambda a: cv2.resize(a, (320, int(320 * a.shape[0] / a.shape[1])), interpolation=cv2.INTER_AREA)
            sheet_rows.append([t(frame), t(out_frame), t(viz)])
        done += 1
        if done % 10 == 0:
            print(f"{done}/{args.n}", flush=True)

    if sheet_rows:
        h = min(r[0].shape[0] for r in sheet_rows)
        sheet_rows = [[im[:h] for im in r] for r in sheet_rows]
        save_jpg(grid(sheet_rows), os.path.join(args.out, "contact_sheet.jpg"))
    print("wrote", args.n, "synthetic frames to", args.out)


if __name__ == "__main__":
    main()
