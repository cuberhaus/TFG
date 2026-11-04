"""Evaluate a polyp-inpainting checkpoint on the held-out validation split.

For every validation frame the real polyp is blanked out and repainted from its
true mask. We then compare:

* visually  -> ``grid.jpg`` (real | input | generated seeds ...)
* FID / KID -> generated composites vs. real polyp crops

Usage (from ``code/src``)::

    python -m polyp_synth.evaluate --lora ../out/polyp_synth/runs/v1/ckpt-4000 --name v1_4000
    python -m polyp_synth.evaluate --name base            # untuned SD-inpainting reference
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from polyp_synth.data import CAPTIONS, PolypInpaintDataset, index_samples, mask_bbox, split_samples
from polyp_synth.pipeline import DEFAULT_STEPS, composite, grid, inpaint_polyp, load_pipeline, save_jpg

OUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "out", "polyp_synth", "eval"))


def to_uint8_tensor(arrs):
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def fid_kid(real, fake, device="cuda"):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(feature=2048, subset_size=min(100, len(fake)), normalize=False).to(device)
    for i in range(0, len(real), 64):
        b = to_uint8_tensor(real[i : i + 64]).to(device)
        fid.update(b, real=True)
        kid.update(b, real=True)
    for i in range(0, len(fake), 64):
        b = to_uint8_tensor(fake[i : i + 64]).to(device)
        fid.update(b, real=False)
        kid.update(b, real=False)
    km, ks = kid.compute()
    return float(fid.compute()), float(km), float(ks)


def polyp_patch(img: np.ndarray, mask: np.ndarray, out: int = 160, expand: float = 1.5) -> np.ndarray:
    """Square patch centred on the polyp (``expand`` x its bbox), resized to ``out``."""
    import cv2

    x0, y0, x1, y1 = mask_bbox(mask)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * expand / 2 + 8
    h, w = img.shape[:2]
    half = min(half, cx, cy, w - cx, h - cy)  # stay inside the frame
    half = max(half, 16)
    a, b = int(round(cx - half)), int(round(cy - half))
    c = int(round(2 * half))
    return cv2.resize(img[b : b + c, a : a + c], (out, out), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=None, help="PEFT adapter dir (omit for the untuned base model)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--grid", type=int, default=10, help="number of validation rows in the grid")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N val items (debug)")
    ap.add_argument("--no-metrics", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(OUT_ROOT, args.name)
    os.makedirs(out_dir, exist_ok=True)
    samples = index_samples()
    train_s, val_s = split_samples(samples)
    val_ds = PolypInpaintDataset(val_s, size=512, train=False)
    train_ds = PolypInpaintDataset(train_s, size=512, train=False)
    idxs = list(range(len(val_ds)))[: args.limit]

    pipe = load_pipeline(args.lora)
    fakes, rows, reals_val = [], [], []
    fake_patches, real_val_patches = [], []
    grid_idx = set(np.linspace(0, len(idxs) - 1, args.grid).astype(int).tolist())
    for n, i in enumerate(idxs):
        it = val_ds.get(i)
        im, hole = it["image"], it["hole"]
        reals_val.append(im)
        real_val_patches.append(polyp_patch(im, it["mask"]))
        comps = []
        for s in range(args.seeds):
            gen = inpaint_polyp(pipe, im, hole, prompt=CAPTIONS[it["source"]], seed=s, steps=args.steps, guidance=args.guidance)
            c = composite(im, gen, hole)
            comps.append(c)
            fakes.append(c)
            fake_patches.append(polyp_patch(c, it["mask"]))
        if n in grid_idx:
            masked = (im * (1 - hole[..., None])).astype(np.uint8)
            rows.append([im, masked] + comps[:3])
        if n % 20 == 0:
            print(f"{n}/{len(idxs)}", flush=True)
    save_jpg(grid(rows), os.path.join(out_dir, "grid.jpg"))

    res = dict(name=args.name, lora=args.lora, steps=args.steps, guidance=args.guidance, n_fake=len(fakes))
    if not args.no_metrics:
        del pipe
        torch.cuda.empty_cache()
        # reference = real *training* crops, so the validation "real floor" is a fair, disjoint comparison
        items = [train_ds.get(i) for i in range(len(train_ds))]
        ref = [it["image"] for it in items]
        ref_patches = [polyp_patch(it["image"], it["mask"]) for it in items]
        res["fid"], res["kid_mean"], res["kid_std"] = fid_kid(ref, fakes)
        res["fid_real_floor"], res["kid_real_floor"], _ = fid_kid(ref, reals_val)
        # polyp-centred patches: far more sensitive to polyp realism than whole frames
        res["patch_fid"], res["patch_kid_mean"], res["patch_kid_std"] = fid_kid(ref_patches, fake_patches)
        res["patch_fid_real_floor"], res["patch_kid_real_floor"], _ = fid_kid(ref_patches, real_val_patches)
    json.dump(res, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
