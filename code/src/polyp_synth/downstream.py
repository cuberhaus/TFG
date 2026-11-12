"""Downstream test: does adding synthetic polyps improve a polyp *detector*?

Everything is evaluated on the held-out real frames that neither the generator
nor any detector ever saw (``split_samples`` validation split), plus held-out
polyp-free frames for the false-positive rate.

Arms (same number of training steps for all of them, so only the data differs)
--------------------------------------------------------------------------
``real``       real training frames only (baseline).
``neg``        real + the *unpainted* polyp-free frames, labelled "no polyp".
               Control: any gain from extra background images alone shows up here.
``synth``      real + the same frames with a polyp painted in (detector-gated).
``synth_all``  like ``synth`` but without the detector gate.

``synth`` and ``neg`` use exactly the same underlying frames, so the only
difference between them is the painted polyp and its free bounding box.

Usage (from ``code/src``)::

    python -m polyp_synth.downstream run --arms real neg synth --seeds 0 1 2 --out ../out/polyp_synth/downstream/full
    python -m polyp_synth.downstream report --dir ../out/polyp_synth/downstream/full
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from polyp_synth.data import PolypSample, blank_scope_overlay, index_samples, load_pair, split_samples, subset_samples
from polyp_synth.detector_check import OUT_ROOT, boxes_from_mask, build_detector

IOUS = [round(0.5 + 0.05 * k, 2) for k in range(10)]  # COCO: 0.50 .. 0.95
SCORE_THR = 0.3  # operating point for recall / false positives
BATCH = 8
DEFAULT_SYNTH = os.path.join(OUT_ROOT, "synth_v1_aug")
DEFAULT_HELDOUT_NEG = os.path.abspath(os.path.join(OUT_ROOT, "..", "..", "data", "polyp_synth", "raw", "negatives_heldout"))


# --------------------------------------------------------------------------
# Metrics (pure numpy, unit-tested)
# --------------------------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in ``a`` (n,4) and ``b`` (m,4), xyxy."""
    a, b = np.asarray(a, np.float64).reshape(-1, 4), np.asarray(b, np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


@dataclass
class ImageMatches:
    """Per-image detection outcome, independent of every other image."""

    scores: np.ndarray  # (n_pred,)
    tp: np.ndarray  # (n_pred, len(IOUS)) bool: pred is a true positive at that IoU
    n_gt: int
    best_iou: np.ndarray  # (n_gt,) best IoU of any pred with score >= SCORE_THR (0 if none)


def match_image(pred_boxes, pred_scores, gt_boxes) -> ImageMatches:
    """Greedy score-ordered matching, one prediction per ground-truth box, at each IoU threshold."""
    pred_boxes = np.asarray(pred_boxes, np.float64).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, np.float64).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, np.float64).reshape(-1, 4)
    order = np.argsort(-pred_scores, kind="stable")
    pred_boxes, pred_scores = pred_boxes[order], pred_scores[order]
    ious = iou_matrix(pred_boxes, gt_boxes)
    tp = np.zeros((len(pred_scores), len(IOUS)), bool)
    for t, thr in enumerate(IOUS):
        used = np.zeros(len(gt_boxes), bool)
        for i in range(len(pred_scores)):
            if len(gt_boxes) == 0:
                break
            cand = np.where(~used & (ious[i] >= thr))[0]
            if len(cand):
                j = cand[np.argmax(ious[i, cand])]
                used[j] = True
                tp[i, t] = True
    keep = pred_scores >= SCORE_THR
    best = ious[keep].max(0) if keep.any() and len(gt_boxes) else np.zeros(len(gt_boxes))
    return ImageMatches(pred_scores, tp, len(gt_boxes), best)


def average_precision(matches: list[ImageMatches], iou_index: int) -> float:
    """All-point interpolated AP over a set of images at IOUS[iou_index]."""
    n_gt = sum(m.n_gt for m in matches)
    if n_gt == 0:
        return float("nan")
    scores = np.concatenate([m.scores for m in matches]) if matches else np.zeros(0)
    tp = np.concatenate([m.tp[:, iou_index] for m in matches]) if matches else np.zeros(0, bool)
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(np.float64)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    # monotone precision envelope, then integrate over recall
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev) * precision))


def summarize(matches: list[ImageMatches]) -> dict:
    """Metrics on frames that contain a polyp."""
    n_gt = sum(m.n_gt for m in matches)
    hits = sum(int((m.best_iou >= 0.5).sum()) for m in matches)
    fps = sum(int(((m.scores >= SCORE_THR) & ~m.tp[:, 0]).sum()) for m in matches)
    return dict(
        ap50=average_precision(matches, 0),
        ap=float(np.mean([average_precision(matches, k) for k in range(len(IOUS))])),
        recall=hits / max(1, n_gt),
        fp_per_img=fps / max(1, len(matches)),
        n=len(matches),
    )


def summarize_negatives(pred_scores: list[np.ndarray]) -> dict:
    """On polyp-free frames every confident detection is a false positive."""
    n = [int((np.asarray(s) >= SCORE_THR).sum()) for s in pred_scores]
    return dict(fp_per_img=float(np.mean(n)), frac_imgs_with_fp=float(np.mean([k > 0 for k in n])), n=len(n))


# --------------------------------------------------------------------------
# Training data
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Item:
    kind: str  # real | synth | neg
    path: str
    sample: PolypSample | None = None
    boxes: tuple = ()


def _read_ann(path: str):
    toks = open(path).read().split()
    return (tuple(int(v) for v in toks[1:5]),)


def synth_items(synth_dir: str, names: list[str], as_negative: bool) -> list[Item]:
    items = []
    for n in names:
        if as_negative:
            items.append(Item("neg", os.path.join(synth_dir, "Before", "synth", n + ".jpg")))
        else:
            ann = os.path.join(synth_dir, "Annotations", "synth", n + ".txt")
            items.append(Item("synth", os.path.join(synth_dir, "Images", "synth", n + ".jpg"), boxes=_read_ann(ann)))
    return items


class MixDet(torch.utils.data.Dataset):
    """Detection dataset over real frames, painted-polyp frames and empty (no-polyp) frames."""

    def __init__(self, items: list[Item]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        rng = np.random.default_rng((torch.initial_seed() + 7919 * i) % (2**32))
        if it.kind == "real":
            img, mask = load_pair(it.sample)
            boxes = np.array(boxes_from_mask(mask), np.float32).reshape(-1, 4)
        else:
            img = np.array(Image.open(it.path).convert("RGB"))
            boxes = np.array(it.boxes, np.float32).reshape(-1, 4)
        if rng.random() < 0.5:
            img = img[:, ::-1]
            w = img.shape[1]
            boxes = np.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], 1) if len(boxes) else boxes
        if rng.random() < 0.5:
            img = img[::-1]
            h = img.shape[0]
            boxes = np.stack([boxes[:, 0], h - boxes[:, 3], boxes[:, 2], h - boxes[:, 1]], 1) if len(boxes) else boxes
        img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15), 0, 255)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return t, dict(boxes=torch.from_numpy(np.ascontiguousarray(boxes)).float().reshape(-1, 4), labels=torch.ones(len(boxes), dtype=torch.int64))


def plan_steps(n_real: int, epochs: int = 15, floor: int = 600) -> int:
    """Same number of optimiser steps for every arm of a regime (so extra data is not extra training)."""
    return max(floor, int(round(epochs * n_real / BATCH)))


def real_subset(train_s: list[PolypSample], frac: float, seed: int, sources=None) -> list[PolypSample]:
    """The real frames of one regime (see ``polyp_synth.data.subset_samples``)."""
    return subset_samples(train_s, frac, seed, sources)


def select_external(samples: list[PolypSample], source: str, train_sources) -> list[PolypSample]:
    """Every frame of a dataset the detectors were never trained on, in frame order.

    Kept in natural order on purpose: consecutive frames of one video are near-duplicates,
    and the block bootstrap in the report resamples contiguous runs of them.
    """
    if train_sources is not None and source in train_sources:
        raise ValueError(f"external set {source!r} is also a training source {list(train_sources)}")
    return [s for s in samples if s.source == source]


def resample_indices(rng: np.random.Generator, n: int, block: int = 1) -> np.ndarray:
    """Bootstrap indices over ``n`` frames; ``block`` > 1 resamples contiguous blocks of frames.

    Frames from the same colonoscopy video are strongly correlated, so resampling single frames
    makes the confidence interval look tighter than the evidence allows.
    """
    if block <= 1:
        return rng.integers(0, n, n)
    nb = math.ceil(n / block)
    picks = rng.integers(0, nb, nb)
    return np.concatenate([np.arange(b * block, min(n, (b + 1) * block)) for b in picks])


# --------------------------------------------------------------------------
# Train + evaluate one run
# --------------------------------------------------------------------------
def train_detector(items: list[Item], steps: int, seed: int, workers: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    dl = torch.utils.data.DataLoader(
        MixDet(items), batch_size=BATCH, shuffle=True, num_workers=workers, collate_fn=lambda b: tuple(zip(*b)),
        drop_last=True, generator=gen, persistent_workers=True,
    )
    m = build_detector().cuda()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    m.train()
    step, run, log = 0, 0.0, []
    while step < steps:
        for imgs, tgts in dl:
            imgs = [i.cuda(non_blocking=True) for i in imgs]
            tgts = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in tgts]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = sum(m(imgs, tgts).values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item()
            step += 1
            if step % 100 == 0:
                log.append((step, run / 100))
                print(f"  step {step}/{steps} loss {run / 100:.4f}", flush=True)
                run = 0.0
            if step >= steps:
                break
    return m, log


@torch.no_grad()
def predict(det, imgs: list[np.ndarray]):
    det.eval()
    out = []
    for img in imgs:
        t = torch.from_numpy(img).permute(2, 0, 1).float().cuda() / 255.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = det([t])[0]
        out.append((o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy()))
    return out


@dataclass
class EvalSets:
    frames: list
    gts: list
    sources: list
    negs: list
    ext_frames: list
    ext_gts: list


def load_eval_sets(neg_dir: str, sources=None, external: str | None = None) -> EvalSets:
    """Held-out real frames, held-out polyp-free frames and (optionally) a whole unseen dataset."""
    samples = index_samples()
    _, val_s = split_samples(samples)
    if sources:  # cross-dataset regime: the in-domain check only uses datasets that were trained on
        val_s = [s for s in val_s if s.source in sources]
    frames, gts, srcs = [], [], []
    for s in val_s:
        img, mask = load_pair(s)
        frames.append(img)
        gts.append(boxes_from_mask(mask))
        srcs.append(s.source)
    ext_frames, ext_gts = [], []
    if external:
        for s in select_external(samples, external, sources):
            img, mask = load_pair(s)
            ext_frames.append(img)
            ext_gts.append(boxes_from_mask(mask))
    neg_files = sorted(glob.glob(os.path.join(neg_dir, "*.jpg")) + glob.glob(os.path.join(neg_dir, "*.png")))
    negs = [blank_scope_overlay(np.array(Image.open(f).convert("RGB"))) for f in neg_files]
    return EvalSets(frames, gts, srcs, negs, ext_frames, ext_gts)


def write_json_atomic(path: str, obj) -> None:
    """Write `obj` so a crash or power loss leaves either the old file or the complete new one.

    A plain open(path, "w") + dump leaves a zero-length file if the machine dies before the
    page cache is flushed, which a resume-by-exists check would then treat as a finished run.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run_is_done(path: str, need=("val", "neg")) -> bool:
    """True only for a readable result file; empty or truncated leftovers from a crash do not count."""
    try:
        with open(path) as f:
            r = json.load(f)
    except (OSError, ValueError):
        return False
    return isinstance(r, dict) and all(k in r for k in need)


def run_one(args, arm: str, seed: int, out_path: str, eval_sets):
    train_s, _ = split_samples(index_samples())
    # Legacy runs redraw the subset per training seed. Regimes with a retrained generator must pin it
    # (--subset-seed) so the detector sees exactly the frames the LoRA and the gate were trained on.
    subset_seed = args.subset_seed if args.subset_seed is not None else seed
    real = real_subset(train_s, args.frac, subset_seed, args.sources)
    n_extra = args.n_extra if args.n_extra is not None else len(real)
    names_kept = open(os.path.join(args.synth_dir, "kept.txt")).read().split()
    names_all = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(args.synth_dir, "Images", "synth", "*.jpg")))
    rng = np.random.default_rng(2000 + seed)
    pool = names_all if arm == "synth_all" else names_kept
    names = [pool[i] for i in sorted(rng.choice(len(pool), size=min(n_extra, len(pool)), replace=False).tolist())]
    items = [Item("real", "", sample=s) for s in real]
    if arm == "neg":
        items += synth_items(args.synth_dir, names, as_negative=True)
    elif arm in ("synth", "synth_all"):
        items += synth_items(args.synth_dir, names, as_negative=False)
    elif arm != "real":
        raise ValueError(arm)
    steps = args.steps or plan_steps(len(real))
    print(f"[{arm} frac={args.frac} subset={subset_seed} sources={args.sources} seed={seed}] "
          f"real={len(real)} extra={len(items) - len(real)} steps={steps}", flush=True)
    t0 = time.time()
    det, log = train_detector(items, steps, seed, args.workers)
    train_s_ = time.time() - t0
    frames, gts, sources, negs = eval_sets.frames, eval_sets.gts, eval_sets.sources, eval_sets.negs
    vp = predict(det, frames)
    npred = predict(det, negs)
    matches = [match_image(b, s, g) for (b, s), g in zip(vp, gts)]
    res = dict(
        arm=arm, frac=args.frac, seed=seed, subset_seed=subset_seed, sources=args.sources,
        n_real=len(real), n_extra=len(items) - len(real), steps=steps,
        train_seconds=train_s_, loss_log=log,
        val=summarize(matches),
        val_by_source={src: summarize([m for m, s in zip(matches, sources) if s == src]) for src in sorted(set(sources))},
        neg=summarize_negatives([s for _, s in npred]),
        preds=dict(
            val=[dict(boxes=b.round(1).tolist(), scores=s.round(4).tolist()) for b, s in vp],
            gts=[[list(map(float, g)) for g in gg] for gg in gts],
            sources=sources,
            neg=[dict(scores=s.round(4).tolist()) for _, s in npred],
        ),
    )
    if eval_sets.ext_frames:  # a whole dataset the detector never saw (cross-dataset regime)
        ep = predict(det, eval_sets.ext_frames)
        ext_matches = [match_image(b, s, g) for (b, s), g in zip(ep, eval_sets.ext_gts)]
        res["ext"] = summarize(ext_matches)
        res["ext_name"] = args.external_eval
        res["preds"]["ext"] = [dict(boxes=b.round(1).tolist(), scores=s.round(4).tolist()) for b, s in ep]
        res["preds"]["ext_gts"] = [[list(map(float, g)) for g in gg] for gg in eval_sets.ext_gts]
    write_json_atomic(out_path, res)
    v, n = res["val"], res["neg"]
    ext = f" | {args.external_eval} AP50 {res['ext']['ap50']:.3f}" if "ext" in res else ""
    print(f"  -> AP50 {v['ap50']:.3f} AP {v['ap']:.3f} recall {v['recall']:.3f} fp/img {v['fp_per_img']:.2f} | neg fp/img {n['fp_per_img']:.2f}{ext} ({train_s_ / 60:.1f} min)", flush=True)
    del det
    torch.cuda.empty_cache()


def run(args):
    eval_sets = load_eval_sets(args.neg_dir, args.sources, args.external_eval)
    print(f"val frames {len(eval_sets.frames)}, held-out polyp-free frames {len(eval_sets.negs)}, "
          f"external ({args.external_eval}) frames {len(eval_sets.ext_frames)}", flush=True)
    need = ("val", "neg") + (("ext",) if args.external_eval else ())
    for seed in args.seeds:  # seed-major so partial results stay balanced
        for arm in args.arms:
            path = os.path.join(args.out, f"{arm}_s{seed}.json")
            if run_is_done(path, need):
                print("skip (done)", path, flush=True)
                continue
            if os.path.exists(path):
                print("redo (unreadable result from an interrupted run)", path, flush=True)
            run_one(args, arm, seed, path, eval_sets)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _load_runs(d: str):
    runs: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(os.path.join(d, "*_s*.json"))):
        if not run_is_done(f):
            print(f"warning: ignoring unreadable result {f}", flush=True)
            continue
        with open(f) as fh:
            r = json.load(fh)
        runs.setdefault(r["arm"], []).append(r)
    return runs


def _matches_of(run_, key: str = "val"):
    p = run_["preds"]
    gts = p["gts"] if key == "val" else p["ext_gts"]
    return [match_image(v["boxes"], v["scores"], g) for v, g in zip(p[key], gts)]


def paired_bootstrap(runs_a: list[dict], runs_b: list[dict], metric, B: int = 2000, seed: int = 0, key: str = "val", block: int = 1):
    """Mean over runs of metric(A) - metric(B) with a 95% CI from resampling the evaluation frames.

    ``block`` > 1 resamples contiguous blocks of frames (for video frames that are near-duplicates).
    """
    ma = [_matches_of(r, key) for r in runs_a]
    mb = [_matches_of(r, key) for r in runs_b]
    n = len(ma[0])
    rng = np.random.default_rng(seed)

    def score(ms, idx):
        return float(np.mean([metric([m[i] for i in idx]) for m in ms]))

    base = score(ma, np.arange(n)) - score(mb, np.arange(n))
    diffs = []
    for _ in range(B):
        idx = resample_indices(rng, n, block)
        diffs.append(score(ma, idx) - score(mb, idx))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, float(lo), float(hi)


def report(args):
    dirs = args.dir
    runs: dict[str, list[dict]] = {}
    for d in dirs:
        for arm, rs in _load_runs(d).items():
            runs.setdefault(arm, []).extend(rs)
    key = args.set  # "val": held-out frames of the training datasets | "ext": a whole unseen dataset
    if key == "ext":
        runs = {a: [r for r in rs if "ext" in r] for a, rs in runs.items()}
        runs = {a: rs for a, rs in runs.items() if rs}
    if not runs:
        raise SystemExit(f"no usable runs for --set {key} in {dirs}")
    block = args.block if key == "ext" else 1
    order = [a for a in ("real", "neg", "synth", "synth_all") if a in runs]
    n_eval = len(runs[order[0]][0]["preds"][key])
    lines = []
    if key == "ext":
        lines.append(f"External dataset: **{runs[order[0]][0]['ext_name']}** ({n_eval} frames, never used for training; "
                     f"CIs resample blocks of {block} consecutive frames).")
        lines.append("")
    lines.append(f"| arm | runs | real | extra | steps | AP50 | AP50:95 | recall@{SCORE_THR} | FP/img (polyp frames) | FP/img (polyp-free frames) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    def ms(rs, getter, pct=True):
        v = np.array([getter(r) for r in rs]) * (100 if pct else 1)
        return f"{v.mean():.1f} ± {v.std(ddof=1) if len(v) > 1 else 0:.1f}" if pct else f"{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}"

    for a in order:
        rs = runs[a]
        lines.append(
            f"| {a} | {len(rs)} | {rs[0]['n_real']} | {rs[0]['n_extra']} | {rs[0]['steps']} | "
            f"{ms(rs, lambda r: r[key]['ap50'])} | {ms(rs, lambda r: r[key]['ap'])} | {ms(rs, lambda r: r[key]['recall'])} | "
            f"{ms(rs, lambda r: r[key]['fp_per_img'], False)} | {ms(rs, lambda r: r['neg']['fp_per_img'], False)} |"
        )
    lines.append("")
    if key == "val":
        for src in ("kvasir", "cvc"):
            if all(src in r["val_by_source"] for a in order for r in runs[a]):
                lines.append(f"{src} frames only (n={runs[order[0]][0]['val_by_source'][src]['n']}):  " + "; ".join(
                    f"{a} AP50 {ms(runs[a], lambda r, s=src: r['val_by_source'][s]['ap50'])}" for a in order))
        lines.append("")
    lines.append("Paired bootstrap over the evaluation frames (95% CI of the difference, mean over runs):")
    lines.append("")
    lines.append("| comparison | ΔAP50 (pts) | ΔAP50:95 (pts) | Δrecall (pts) |")
    lines.append("|---|---|---|---|")
    metrics = dict(
        ap50=lambda m: average_precision(m, 0),
        ap=lambda m: float(np.mean([average_precision(m, k) for k in range(len(IOUS))])),
        recall=lambda m: sum(int((x.best_iou >= 0.5).sum()) for x in m) / max(1, sum(x.n_gt for x in m)),
    )
    for a, b in (("synth", "real"), ("synth", "neg"), ("neg", "real"), ("synth_all", "real"), ("synth_all", "synth")):
        if a in runs and b in runs:
            cells = []
            for k in ("ap50", "ap", "recall"):
                d, lo, hi = paired_bootstrap(runs[a], runs[b], metrics[k], B=args.boot, key=key, block=block)
                sig = "  *" if (lo > 0 or hi < 0) else ""
                cells.append(f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]{sig}")
            lines.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("`*` = the 95% interval excludes 0.")
    text = "\n".join(lines)
    print(text)
    if args.md:
        md = args.md
    elif len(dirs) == 1:
        md = os.path.join(dirs[0], "report.md" if key == "val" else "report_ext.md")
    else:
        md = None  # pooled report over several directories needs an explicit --md
    if md:
        open(md, "w").write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=["real", "neg", "synth"], choices=["real", "neg", "synth", "synth_all"])
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--frac", type=float, default=1.0, help="fraction of the real training frames to use")
    r.add_argument("--subset-seed", type=int, default=None,
                   help="pin which draw of --frac is used (needed when the generator and gate were trained on that draw); "
                        "default: redraw per training seed")
    r.add_argument("--sources", nargs="+", default=None, choices=["kvasir", "cvc"], help="train only on these datasets")
    r.add_argument("--external-eval", default=None, choices=["kvasir", "cvc"],
                   help="also score every frame of this (untrained-on) dataset")
    r.add_argument("--n-extra", type=int, default=None, help="extra frames added (default: as many as real frames)")
    r.add_argument("--steps", type=int, default=None)
    r.add_argument("--synth-dir", default=DEFAULT_SYNTH)
    r.add_argument("--neg-dir", default=DEFAULT_HELDOUT_NEG)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--dir", nargs="+", required=True, help="one or more run directories (pooled)")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--set", default="val", choices=["val", "ext"], help="held-out frames of the training sets, or the external dataset")
    p.add_argument("--block", type=int, default=20, help="bootstrap block length (consecutive frames) for --set ext")
    p.add_argument("--md", default=None, help="where to write the markdown (default: <dir>/report.md)")
    a = ap.parse_args()
    {"run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
