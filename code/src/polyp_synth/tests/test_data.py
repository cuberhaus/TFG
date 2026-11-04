"""CPU-only unit tests for the pure helpers (no GPU, models or datasets needed).

Run from ``code/src``:  ``python -m pytest polyp_synth/tests -q``
"""
import numpy as np

from polyp_synth.data import (
    PolypSample, blank_scope_overlay, boxes_to_ellipse_mask, fov_mask, make_hole, mask_bbox, sample_square_crop,
    split_samples,
)
from polyp_synth.generate import SIZE, on_mucosa, place_random_mask, shape_ok
from polyp_synth.pipeline import composite


def _disc(h, w, cx, cy, r):
    yy, xx = np.mgrid[:h, :w]
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.uint8)


def test_boxes_to_ellipse_is_inside_box_and_not_a_rectangle():
    m = boxes_to_ellipse_mask((100, 100), [(20, 30, 60, 70)])
    x0, y0, x1, y1 = mask_bbox(m)
    assert (x0, y0) >= (19, 29) and (x1, y1) <= (62, 72)
    assert m.sum() < 40 * 40  # an ellipse, i.e. smaller than the filled rectangle


def test_make_hole_contains_mask_and_dilates():
    m = _disc(128, 128, 64, 64, 15)
    h = make_hole(m, 5)
    assert (h >= m).all() and h.sum() > m.sum()
    assert make_hole(m, 0).sum() == m.sum()


def test_make_hole_hull_is_superset_of_mask():
    m = _disc(128, 128, 40, 64, 12) | _disc(128, 128, 80, 64, 12)
    assert (make_hole(m, 0, hull=True) >= m).all()
    assert make_hole(m, 0, hull=True).sum() > m.sum()  # fills the gap between the two blobs


def test_square_crop_contains_bbox_when_it_fits():
    rng = np.random.default_rng(0)
    h, w = 300, 400
    bbox = (150, 100, 230, 180)
    for _ in range(200):
        x0, y0, side = sample_square_crop(h, w, bbox, rng)
        assert 0 <= x0 and 0 <= y0 and x0 + side <= w and y0 + side <= h
        assert x0 <= bbox[0] and bbox[2] <= x0 + side and y0 <= bbox[1] and bbox[3] <= y0 + side


def test_square_crop_deterministic_is_stable():
    a = sample_square_crop(300, 400, (150, 100, 230, 180), np.random.default_rng(1), deterministic=True)
    b = sample_square_crop(300, 400, (150, 100, 230, 180), np.random.default_rng(2), deterministic=True)
    assert a == b


def test_split_has_no_overlap_and_holds_out_cvc_blocks():
    samples = [PolypSample(f"k{i}", f"m{i}", "kvasir", f"kv_{i}") for i in range(200)]
    samples += [PolypSample(f"c{i}", f"m{i}", "cvc", f"cvc_{i:04d}") for i in range(612)]
    tr, va = split_samples(samples)
    assert not ({s.uid for s in tr} & {s.uid for s in va})
    assert sum(s.source == "kvasir" for s in va) == 100
    cvc_val = sorted(int(s.uid.split("_")[1]) for s in va if s.source == "cvc")
    assert len(cvc_val) == 60
    # guard band: frames adjacent to held-out blocks are dropped from train (no near-duplicate leakage)
    train_ids = {int(s.uid.split("_")[1]) for s in tr if s.source == "cvc"}
    assert 39 - 4 not in train_ids and 60 + 4 not in train_ids


def test_composite_leaves_outside_pixels_untouched():
    orig = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    gen = np.full_like(orig, 255)
    region = _disc(64, 64, 32, 32, 10)
    out = composite(orig, gen, region, feather=2.0)
    far = _disc(64, 64, 32, 32, 16) == 0
    assert (out[far] == orig[far]).all()
    assert (out[region > 0].mean() > orig[region > 0].mean())  # interior was replaced


def test_place_random_mask_stays_inside_fov_and_keeps_shape_area():
    rng = np.random.default_rng(0)
    fov = _disc(SIZE, SIZE, SIZE // 2, SIZE // 2, SIZE // 2 - 10)
    src = _disc(SIZE, SIZE, 100, 100, 40)
    for _ in range(10):
        m = place_random_mask(src, fov, rng)
        assert m is not None
        assert (fov[m > 0] > 0).all()
        assert 0.5 < m.sum() / src.sum() < 2.0


def test_fov_mask_ignores_black_border_and_text():
    img = np.zeros((200, 200, 3), np.uint8)
    img[:] = 0
    yy, xx = np.mgrid[:200, :200]
    img[((xx - 100) ** 2 + (yy - 100) ** 2) < 80**2] = (180, 90, 80)
    img[5:9, 5:40] = 255  # text-like overlay in the black corner
    f = fov_mask(img)
    assert f[100, 100] == 1 and f[7, 10] == 0 and f[0, 0] == 0


def test_blank_scope_overlay_blacks_out_teal_box_only():
    img = np.full((300, 400, 3), (170, 90, 80), np.uint8)
    img[150:290, 0:150] = (18, 163, 134)  # HyperKvasir ScopeGuide teal
    out = blank_scope_overlay(img)
    assert out[200, 50].sum() == 0
    assert (out[10, 300] == (170, 90, 80)).all()
    plain = np.full((300, 400, 3), (170, 90, 80), np.uint8)
    assert (blank_scope_overlay(plain) == plain).all()


def test_shape_ok_accepts_blob_rejects_thin_strip():
    blob = _disc(128, 128, 64, 64, 25)
    strip = np.zeros((128, 128), np.uint8)
    strip[60:68, 10:120] = 1
    assert shape_ok(blob)
    assert not shape_ok(strip)


def test_on_mucosa_rejects_dark_lumen():
    img = np.full((SIZE, SIZE, 3), 150, np.uint8)
    fov = np.ones((SIZE, SIZE), np.uint8)
    img[200:300, 200:300] = 10  # dark lumen
    lit = _disc(SIZE, SIZE, 100, 100, 30)
    dark = _disc(SIZE, SIZE, 250, 250, 30)
    assert on_mucosa(img, fov, lit)
    assert not on_mucosa(img, fov, dark)
