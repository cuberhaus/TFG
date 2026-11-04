"""Split the polyp-free (HyperKvasir) frames into a pool for generation and a held-out pool.

The held-out pool is used only to measure a detector's false-positive rate on
frames that contain no polyp, so none of those frames may be used as a
background for synthetic polyps or as a training negative.

    python -m polyp_synth.split_negatives            # from code/src

Creates two folders of *relative* symlinks next to ``negatives/``::

    negatives_gen/       frames to paint synthetic polyps onto
    negatives_heldout/   frames reserved for the false-positive test

The split is deterministic (seeded) and idempotent.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from polyp_synth.data import DEFAULT_ROOT


def split_negatives(root: str = DEFAULT_ROOT, n_heldout: int = 200, seed: int = 0) -> tuple[int, int]:
    files = sorted(glob.glob(os.path.join(root, "negatives", "*", "*.jpg")) + glob.glob(os.path.join(root, "negatives", "*", "*.png")))
    if not files:
        raise SystemExit(f"no negative frames under {root}/negatives; run polyp_synth.download_data first")
    held = set(np.random.default_rng(seed).permutation(len(files))[:n_heldout].tolist())
    counts = {"negatives_gen": 0, "negatives_heldout": 0}
    for name in counts:
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        for old in glob.glob(os.path.join(d, "*")):
            os.remove(old)
    for i, f in enumerate(files):
        name = "negatives_heldout" if i in held else "negatives_gen"
        dst = os.path.join(root, name, os.path.basename(f))
        os.symlink(os.path.relpath(f, os.path.dirname(dst)), dst)
        counts[name] += 1
    return counts["negatives_gen"], counts["negatives_heldout"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--n-heldout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n_gen, n_held = split_negatives(a.root, a.n_heldout, a.seed)
    print(f"negatives_gen: {n_gen} frames, negatives_heldout: {n_held} frames")
