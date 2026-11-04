"""Data utilities for mask-conditioned polyp synthesis.

Sources
-------
* Kvasir-SEG       (1000 frames, real pixel-level masks)
* CVC-ClinicDB     (612 frames, real pixel-level masks)
* LDPolypVideo     (bounding boxes only -> converted to ellipse masks)

Everything here is framework-light (numpy / cv2 / PIL); the torch Dataset at
the bottom is the only part that needs torch.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

DEFAULT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "polyp_synth", "raw")
)

# Short, source-specific captions. The source token lets the model keep the
# (very different) image statistics of each source apart, and lets us pick the
# style at generation time.
CAPTIONS = {
    "kvasir": "endoscopic colonoscopy image of a polyp, kvasir style",
    "cvc": "endoscopic colonoscopy image of a polyp, cvc style",
    "ldpolyp": "endoscopic colonoscopy image of a polyp, ldpolyp style",
}


@dataclass(frozen=True)
class PolypSample:
    image_path: str
    mask_path: str | None
    source: str
    uid: str
    boxes: tuple = ()  # only used when mask_path is None (bbox-only sources)


# --------------------------------------------------------------------------
# Indexing / splitting
# --------------------------------------------------------------------------
def _natural_key(path: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(path))]


def index_samples(root: str = DEFAULT_ROOT) -> list[PolypSample]:
    out: list[PolypSample] = []
    kv_dir = os.path.join(root, "kvasir-seg", "Kvasir-SEG")
    for f in sorted(glob.glob(os.path.join(kv_dir, "images", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        out.append(PolypSample(f, os.path.join(kv_dir, "masks", stem + ".jpg"), "kvasir", f"kv_{stem}"))

    cvc_files = []
    for split in ("train", "validation", "test"):
        cvc_files += glob.glob(os.path.join(root, "CVC-ClinicDB", split, "images", "*.png"))
    for f in sorted(cvc_files, key=_natural_key):
        stem = os.path.splitext(os.path.basename(f))[0]
        mask = os.path.join(os.path.dirname(os.path.dirname(f)), "masks", os.path.basename(f))
        out.append(PolypSample(f, mask, "cvc", f"cvc_{int(stem):04d}" if stem.isdigit() else f"cvc_{stem}"))
    return out


def split_samples(samples: list[PolypSample], n_val_kvasir: int = 100, cvc_blocks=(40, 200, 400), cvc_block_len: int = 20, seed: int = 0):
    """Deterministic train/val split.

    CVC-ClinicDB frames come from only ~29 video sequences with consecutive
    frame ids, so a random split would leak near-duplicates into validation.
    We hold out *contiguous* blocks of ids instead.
    """
    rng = np.random.default_rng(seed)
    kv = [s for s in samples if s.source == "kvasir"]
    val_kv = set(rng.choice(len(kv), size=min(n_val_kvasir, len(kv)), replace=False).tolist())
    cvc = [s for s in samples if s.source == "cvc"]
    val_cvc = set()
    for start in cvc_blocks:
        val_cvc.update(range(start, min(start + cvc_block_len, len(cvc))))
    # Keep a guard band around held-out blocks out of train as well.
    guard = set()
    for i in val_cvc:
        for d in range(-5, 6):
            guard.add(i + d)
    train, val = [], []
    for i, s in enumerate(kv):
        (val if i in val_kv else train).append(s)
    for i, s in enumerate(cvc):
        if i in val_cvc:
            val.append(s)
        elif i in guard:
            continue
        else:
            train.append(s)
    others = [s for s in samples if s.source not in ("kvasir", "cvc")]
    train += others
    return train, val


# --------------------------------------------------------------------------
# Loading / mask helpers
# --------------------------------------------------------------------------
def boxes_to_ellipse_mask(shape_hw, boxes) -> np.ndarray:
    """Rough polyp mask from bounding boxes (inscribed ellipse).

    Used for bbox-only datasets such as LDPolypVideo, where no segmentation
    masks exist. An ellipse is a far better prior than the filled rectangle
    used originally: real polyps are round-ish, not rectangular.
    """
    m = np.zeros(shape_hw, np.uint8)
    for x1, y1, x2, y2 in boxes:
        c = (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))
        ax = (max(1, int(round((x2 - x1) / 2))), max(1, int(round((y2 - y1) / 2))))
        cv2.ellipse(m, c, ax, 0, 0, 360, 1, -1)
    return m


def load_pair(s: PolypSample) -> tuple[np.ndarray, np.ndarray]:
    img = np.array(Image.open(s.image_path).convert("RGB"))
    if s.mask_path is None:
        mask = boxes_to_ellipse_mask(img.shape[:2], s.boxes)
    else:
        m = np.array(Image.open(s.mask_path).convert("L"))
        if m.shape[:2] != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (m > 127).astype(np.uint8)
        # Close pinholes / JPEG noise in the masks.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return img, mask


def fov_mask(img: np.ndarray, thresh: int = 14) -> np.ndarray:
    """Binary mask of the endoscope field of view (non-black region)."""
    g = img.max(axis=2)
    m = (g > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # Keep the largest connected component (drops text overlays etc.).
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    # Fill holes (specular blobs, dark lumen).
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out


def blank_scope_overlay(img: np.ndarray) -> np.ndarray:
    """Black out the teal ScopeGuide box that HyperKvasir frames carry.

    Kvasir-SEG frames (our training data) have that box blanked, so we match
    the training distribution instead of letting the model paint around it.
    """
    h, w = img.shape[:2]
    y_lo, x_hi = int(h * 0.5), int(w * 0.45)
    reg = img[y_lo:, :x_hi].astype(np.int16)
    r, g, b = reg[..., 0], reg[..., 1], reg[..., 2]
    teal = (r < 60) & (g > 90) & (b > 80) & (np.abs(g - b) < 45)
    if teal.sum() < 5000:
        return img
    ys, xs = np.where(teal)
    x0, x1 = int(np.percentile(xs, 0.5)), int(np.percentile(xs, 99.5))
    y0, y1 = int(np.percentile(ys, 0.5)) + y_lo, int(np.percentile(ys, 99.5)) + y_lo
    out = img.copy()
    out[max(0, y0 - 2) : y1 + 3, max(0, x0 - 2) : x1 + 3] = 0
    return out


def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def make_hole(mask: np.ndarray, dilate_px: int, hull: bool = False) -> np.ndarray:
    """Inpainting hole = polyp mask (optionally its convex hull) dilated a bit."""
    m = mask.astype(np.uint8)
    if hull:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            pts = np.concatenate(cnts)
            m = np.zeros_like(m)
            cv2.fillPoly(m, [cv2.convexHull(pts)], 1)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)
    return m


# --------------------------------------------------------------------------
# Cropping
# --------------------------------------------------------------------------
def sample_square_crop(h, w, bbox, rng: np.random.Generator, zoom_min: float = 0.6, deterministic: bool = False):
    """Pick a square crop (x0, y0, side) that contains the polyp bbox.

    side is drawn between zoom_min * short_side (zoomed in) and the full short
    side. When the polyp is bigger than the crop it is simply clipped.
    """
    side_max = min(h, w)
    x0b, y0b, x1b, y1b = bbox
    bw, bh = x1b - x0b, y1b - y0b
    if deterministic:
        side = side_max
    else:
        s_lo = min(side_max, max(side_max * zoom_min, max(bw, bh) * 1.25))
        side = int(round(rng.uniform(s_lo, side_max)))
    cx, cy = (x0b + x1b) / 2, (y0b + y1b) / 2

    def pos(lo_b, hi_b, c, dim):
        # allowed start range so that the bbox is fully inside when it fits
        if hi_b - lo_b <= side:
            lo, hi = max(0, hi_b - side), min(lo_b, dim - side)
        else:
            lo = hi = int(np.clip(c - side / 2, 0, dim - side))
        if hi < lo:
            lo = hi = int(np.clip(c - side / 2, 0, dim - side))
        return int(lo) if deterministic else int(rng.integers(lo, hi + 1))

    if deterministic:
        x0 = int(np.clip(cx - side / 2, 0, w - side))
        y0 = int(np.clip(cy - side / 2, 0, h - side))
    else:
        x0, y0 = pos(x0b, x1b, cx, w), pos(y0b, y1b, cy, h)
    return x0, y0, side


def resize_pair(img, mask, size):
    img_r = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC)
    m_r = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return img_r, (m_r > 0.5).astype(np.uint8)


# --------------------------------------------------------------------------
# torch dataset
# --------------------------------------------------------------------------
def _torch():
    import torch

    return torch


class PolypInpaintDataset:
    """(image, hole, caption) triplets for inpainting fine-tuning.

    A training item is a random square crop around a real polyp, resized to
    ``size``. The hole is the (slightly dilated) real polyp mask, so the model
    learns "given the surrounding mucosa and a hole shape, paint a polyp that
    fills it".
    """

    def __init__(
        self,
        samples: list[PolypSample],
        size: int = 512,
        train: bool = True,
        dilate_frac: tuple[float, float] = (0.012, 0.03),
        p_hull: float = 0.25,
        zoom_min: float = 0.6,
        caption_dropout: float = 0.1,
        seed: int = 0,
    ):
        self.samples = samples
        self.size = size
        self.train = train
        self.dilate_frac = dilate_frac
        self.p_hull = p_hull
        self.zoom_min = zoom_min
        self.caption_dropout = caption_dropout
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    def get(self, idx: int, rng: np.random.Generator | None = None):
        s = self.samples[idx]
        if rng is None:
            rng = np.random.default_rng(self.seed + idx) if not self.train else np.random.default_rng()
        img, mask = load_pair(s)
        bbox = mask_bbox(mask)
        if bbox is None:  # degenerate mask: use whole frame
            bbox = (0, 0, img.shape[1], img.shape[0])
        x0, y0, side = sample_square_crop(img.shape[0], img.shape[1], bbox, rng, self.zoom_min, deterministic=not self.train)
        img_c = img[y0 : y0 + side, x0 : x0 + side]
        m_c = mask[y0 : y0 + side, x0 : x0 + side]
        img_r, m_r = resize_pair(img_c, m_c, self.size)
        if self.train:
            if rng.random() < 0.5:
                img_r, m_r = img_r[:, ::-1], m_r[:, ::-1]
            if rng.random() < 0.5:
                img_r, m_r = img_r[::-1], m_r[::-1]
            img_r, m_r = np.ascontiguousarray(img_r), np.ascontiguousarray(m_r)
            frac = rng.uniform(*self.dilate_frac)
            hull = rng.random() < self.p_hull
        else:
            frac = float(np.mean(self.dilate_frac))
            hull = False
        hole = make_hole(m_r, max(1, int(round(frac * self.size))), hull=hull)
        return dict(image=img_r, mask=m_r, hole=hole, source=s.source, uid=s.uid)

    def __getitem__(self, idx):
        torch = _torch()
        item = self.get(idx)
        rng = np.random.default_rng()
        caption = CAPTIONS[item["source"]]
        if self.train and rng.random() < self.caption_dropout:
            caption = ""
        pix = torch.from_numpy(item["image"]).permute(2, 0, 1).float() / 127.5 - 1.0
        hole = torch.from_numpy(item["hole"]).float()[None]
        return dict(pixel_values=pix, hole=hole, caption=caption)
