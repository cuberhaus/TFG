"""Objective realism check: does a detector trained on REAL polyps find the
SYNTHETIC polyps where the conditioning mask says they are?

1. ``train``  : Faster R-CNN on real training frames (Kvasir-SEG + CVC-ClinicDB).
2. ``eval``   : detection rate (recall @ IoU 0.5) on
                 * real validation frames,
                 * the same frames with the polyp repainted by a model,
               so a realistic generator should score close to the real frames,
               whereas a generator that does not paint a polyp scores ~0.

Usage (from ``code/src``)::

    python -m polyp_synth.detector_check train --out ../out/polyp_synth/detector
    python -m polyp_synth.detector_check eval --det ../out/polyp_synth/detector/det.pt \
        --lora ../out/polyp_synth/runs/v1/final --name v1
"""
from __future__ import annotations

import argparse
import json
import math
import os

import cv2
import numpy as np
import torch
from PIL import Image

from polyp_synth.pipeline import DEFAULT_STEPS
from polyp_synth.data import (
    CAPTIONS, PolypSample, add_subset_args, index_samples, load_pair, make_hole, mask_bbox, split_samples, subset_samples,
)

OUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "out", "polyp_synth"))


def boxes_from_mask(mask: np.ndarray, min_frac: float = 0.0005):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if a >= min_frac * mask.size:
            out.append([x, y, x + w, y + h])
    return out or ([list(mask_bbox(mask))] if mask.any() else [])


def build_detector(pretrained=True):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    m = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None, min_size=512, max_size=704)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    return m


class RealDet(torch.utils.data.Dataset):
    def __init__(self, samples, train=True):
        self.samples, self.train = samples, train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img, mask = load_pair(self.samples[i])
        boxes = np.array(boxes_from_mask(mask), np.float32).reshape(-1, 4)
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
                w = img.shape[1]
                boxes = np.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], 1)
            if np.random.rand() < 0.5:
                img = img[::-1]
                h = img.shape[0]
                boxes = np.stack([boxes[:, 0], h - boxes[:, 3], boxes[:, 2], h - boxes[:, 1]], 1)
            img = np.clip(img.astype(np.float32) * np.random.uniform(0.8, 1.2) + np.random.uniform(-15, 15), 0, 255)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return t, dict(boxes=torch.from_numpy(boxes), labels=torch.ones(len(boxes), dtype=torch.int64))


def train(args):
    torch.manual_seed(0)
    np.random.seed(0)
    samples = index_samples()
    train_s, _ = split_samples(samples)
    train_s = subset_samples(train_s, args.frac, args.subset_seed, args.sources)
    ds = RealDet(train_s, True)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=6, collate_fn=lambda b: tuple(zip(*b)), drop_last=True)
    if args.steps:  # tiny regimes: match the optimiser steps the downstream detectors get
        args.epochs = max(args.epochs, math.ceil(args.steps / len(dl)))
    print(f"gate detector: {len(train_s)} real frames, {args.epochs} epochs x {len(dl)} steps", flush=True)
    m = build_detector().cuda()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / total)))
    m.train()
    step = 0
    for ep in range(args.epochs):
        run = 0.0
        for imgs, tgts in dl:
            imgs = [i.cuda() for i in imgs]
            tgts = [{k: v.cuda() for k, v in t.items()} for t in tgts]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = sum(m(imgs, tgts).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item()
            step += 1
        print(f"epoch {ep + 1}/{args.epochs} loss {run / len(dl):.4f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save(m.state_dict(), os.path.join(args.out, "det.pt"))


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


@torch.no_grad()
def detect_rate(det, frames, gts, score_thr=0.3):
    """frames: list of HxWx3 uint8; gts: list of lists of boxes. Returns dict."""
    hits, best_scores, fps = 0, [], 0
    n_gt = 0
    for img, gt in zip(frames, gts):
        t = torch.from_numpy(img).permute(2, 0, 1).float().cuda() / 255.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = det([t])[0]
        boxes, scores = o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy()
        keep = scores >= score_thr
        boxes, scores = boxes[keep], scores[keep]
        matched = set()
        for g in gt:
            n_gt += 1
            best = 0.0
            for j, (b, s) in enumerate(zip(boxes, scores)):
                if iou(g, b) >= 0.5:
                    best = max(best, float(s))
                    matched.add(j)
            hits += best > 0
            best_scores.append(best)
        fps += len(boxes) - len(matched)
    return dict(recall=hits / max(1, n_gt), mean_score=float(np.mean(best_scores)), fp_per_img=fps / max(1, len(frames)), n=len(frames))


def repaint_frame(pipe, sample: PolypSample, seed: int, steps=16, guidance=4.0, dilate=0.02):
    """Blank the real polyp and repaint it from its mask; composite at native resolution."""
    from polyp_synth.pipeline import composite, inpaint_polyp

    img, mask = load_pair(sample)
    H, W = img.shape[:2]
    side = min(H, W)
    bb = mask_bbox(mask)
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    x0, y0 = int(np.clip(cx - side / 2, 0, W - side)), int(np.clip(cy - side / 2, 0, H - side))
    crop = img[y0 : y0 + side, x0 : x0 + side]
    m = mask[y0 : y0 + side, x0 : x0 + side]
    crop512 = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA if side > 512 else cv2.INTER_CUBIC)
    m512 = (cv2.resize(m.astype(np.float32), (512, 512), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
    hole = make_hole(m512, max(1, int(round(dilate * 512))))
    gen = inpaint_polyp(pipe, crop512, hole, prompt=CAPTIONS[sample.source], seed=seed, steps=steps, guidance=guidance)
    hole_full = (cv2.resize(hole.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    gen_full = cv2.resize(gen, (side, side), interpolation=cv2.INTER_CUBIC)
    out = img.copy()
    out[y0 : y0 + side, x0 : x0 + side] = composite(crop, gen_full, hole_full, feather=side / 512 * 2.0)
    return out, img, mask


def evaluate(args):
    from polyp_synth.pipeline import load_pipeline

    _, val_s = split_samples(index_samples())
    det = build_detector(pretrained=False).cuda().eval()
    det.load_state_dict(torch.load(args.det))
    pipe = load_pipeline(args.lora)
    real_frames, fake_frames, gts = [], [], []
    for i, s in enumerate(val_s):
        fake, real, mask = repaint_frame(pipe, s, seed=0, steps=args.steps, guidance=args.guidance)
        real_frames.append(real)
        fake_frames.append(fake)
        gts.append(boxes_from_mask(mask))
        if i % 40 == 0:
            print(i, flush=True)
    res = dict(name=args.name, lora=args.lora, steps=args.steps, guidance=args.guidance,
               real=detect_rate(det, real_frames, gts), synthetic=detect_rate(det, fake_frames, gts))
    out_dir = os.path.join(OUT_ROOT, "eval", args.name)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "detector_check.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


@torch.no_grad()
def score_dir(args):
    """Score a generated folder: does the real-data detector find each polyp at its labelled box?

    Writes ``scores.json`` and ``kept.txt`` (frames whose polyp is found with
    score >= ``--keep-thr``) so weak or ambiguous insertions can be dropped
    before they are used as training data.
    """
    import glob

    det = build_detector(pretrained=False).cuda().eval()
    det.load_state_dict(torch.load(args.det))
    files = sorted(glob.glob(os.path.join(args.dir, "Images", "synth", "*.jpg")))
    scores = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        ann = open(os.path.join(args.dir, "Annotations", "synth", name + ".txt")).read().split()
        gt = list(map(int, ann[1:5]))
        img = np.array(Image.open(f).convert("RGB"))
        t = torch.from_numpy(img).permute(2, 0, 1).float().cuda() / 255.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = det([t])[0]
        best = 0.0
        for b, sc in zip(o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy()):
            if iou(gt, b) >= 0.5:
                best = max(best, float(sc))
        scores[name] = best
    kept = [n for n, v in scores.items() if v >= args.keep_thr]
    json.dump(scores, open(os.path.join(args.dir, "scores.json"), "w"), indent=1)
    open(os.path.join(args.dir, "kept.txt"), "w").write("\n".join(kept))
    vals = np.array(list(scores.values()))
    print(json.dumps(dict(n=len(vals), found_at_0_3=float((vals >= 0.3).mean()), kept_at_thr=len(kept), keep_thr=args.keep_thr,
                          mean_score=float(vals.mean())), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", default=os.path.join(OUT_ROOT, "detector"))
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--steps", type=int, default=None, help="minimum optimiser steps (raises --epochs for small regimes)")
    add_subset_args(t)
    e = sub.add_parser("eval")
    e.add_argument("--det", default=os.path.join(OUT_ROOT, "detector", "det.pt"))
    e.add_argument("--lora", default=None)
    e.add_argument("--name", required=True)
    e.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    e.add_argument("--guidance", type=float, default=4.0)
    sc = sub.add_parser("score")
    sc.add_argument("--dir", required=True, help="folder produced by polyp_synth.generate")
    sc.add_argument("--det", default=os.path.join(OUT_ROOT, "detector", "det.pt"))
    sc.add_argument("--keep-thr", type=float, default=0.5)
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "score": score_dir}[a.cmd](a)
