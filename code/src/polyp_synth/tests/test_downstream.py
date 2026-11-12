"""CPU-only tests for the detection metrics / planning helpers used by the downstream experiment."""
import argparse
import json

import numpy as np
import pytest

from polyp_synth.data import PolypSample, subset_samples
from polyp_synth.downstream import (
    IOUS, average_precision, iou_matrix, match_image, plan_steps, real_subset, report, resample_indices, run_is_done,
    select_external, summarize, summarize_negatives, write_json_atomic,
)

GT = [[10, 10, 50, 50]]


def test_iou_matrix_identity_disjoint_and_half_overlap():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]])
    got = iou_matrix(a, b)[0]
    assert got[0] == pytest.approx(1.0)
    assert got[1] == 0.0
    assert got[2] == pytest.approx(50 / 150)
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_perfect_detector_has_ap_one():
    m = [match_image([[10, 10, 50, 50]], [0.9], GT)]
    assert average_precision(m, 0) == pytest.approx(1.0)
    assert summarize(m)["recall"] == 1.0


def test_confident_false_positive_ranked_first_halves_ap():
    m = [match_image([[200, 200, 240, 240], [10, 10, 50, 50]], [0.95, 0.6], GT)]
    assert average_precision(m, 0) == pytest.approx(0.5)
    assert summarize(m)["fp_per_img"] == 1.0


def test_missed_polyp_and_empty_predictions():
    m = [match_image([], [], GT)]
    assert average_precision(m, 0) == 0.0
    assert summarize(m)["recall"] == 0.0


def test_duplicate_detection_of_one_polyp_is_a_false_positive():
    m = [match_image([[10, 10, 50, 50], [11, 11, 51, 51]], [0.9, 0.8], GT)]
    assert m[0].tp[:, 0].tolist() == [True, False]
    assert summarize(m)["fp_per_img"] == 1.0


def test_stricter_iou_threshold_drops_loose_boxes():
    m = [match_image([[10, 10, 60, 60]], [0.9], GT)]  # IoU = 1600 / 2500 = 0.64
    assert m[0].tp[0, IOUS.index(0.5)] and not m[0].tp[0, IOUS.index(0.7)]
    s = summarize(m)
    assert s["ap50"] == pytest.approx(1.0) and s["ap"] < s["ap50"]


def test_low_score_detections_do_not_count_for_recall_but_do_for_ap():
    m = [match_image([[10, 10, 50, 50]], [0.1], GT)]  # below the 0.3 operating point
    s = summarize(m)
    assert s["recall"] == 0.0 and s["ap50"] == pytest.approx(1.0)


def test_ap_pools_scores_across_images():
    good = match_image([[10, 10, 50, 50]], [0.9], GT)
    bad_fp_only = match_image([[0, 0, 5, 5]], [0.95], [])  # polyp-free frame can't be in AP's gt, but its FP ranks first
    ap = average_precision([good, bad_fp_only], 0)
    assert ap == pytest.approx(0.5)


def test_summarize_negatives():
    s = summarize_negatives([np.array([0.9, 0.5]), np.array([0.1]), np.array([])])
    assert s["fp_per_img"] == pytest.approx(2 / 3)
    assert s["frac_imgs_with_fp"] == pytest.approx(1 / 3)


def test_plan_steps_has_floor_and_scales_with_data():
    assert plan_steps(100) == 600  # floor for tiny regimes
    assert plan_steps(1600) == 3000


def test_real_subset_is_deterministic_and_proper_subset():
    train = [PolypSample(f"i{i}.jpg", f"m{i}.png", "kvasir", f"u{i}") for i in range(200)]
    a, b = real_subset(train, 0.25, seed=3), real_subset(train, 0.25, seed=3)
    assert a == b and len(a) == 50 and set(a) <= set(train)
    assert real_subset(train, 0.25, seed=4) != a
    assert real_subset(train, 1.0, seed=0) == train


def test_run_is_done_rejects_missing_empty_and_partial_results(tmp_path):
    good = tmp_path / "real_s0.json"
    write_json_atomic(str(good), {"val": {}, "neg": {}})
    assert run_is_done(str(good))
    assert not run_is_done(str(tmp_path / "missing.json"))
    empty = tmp_path / "real_s1.json"
    empty.write_text("")  # what a power loss mid-save leaves behind
    assert not run_is_done(str(empty))
    partial = tmp_path / "neg_s1.json"
    partial.write_text('{"val": {"ap50": 0.8')
    assert not run_is_done(str(partial))
    wrong_shape = tmp_path / "x_s0.json"
    wrong_shape.write_text("[1, 2]")
    assert not run_is_done(str(wrong_shape))


def test_write_json_atomic_replaces_in_place_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "sub" / "r.json"
    write_json_atomic(str(path), {"val": 1, "neg": 1})
    write_json_atomic(str(path), {"val": 2, "neg": 2})
    assert run_is_done(str(path))
    assert [p.name for p in path.parent.iterdir()] == ["r.json"]


def _mixed_train():
    return [PolypSample(f"k{i}.jpg", f"m{i}.png", "kvasir", f"kv_{i}") for i in range(90)] + [
        PolypSample(f"c{i}.png", f"m{i}.png", "cvc", f"cvc_{i}") for i in range(60)
    ]


def test_subset_samples_is_the_single_source_of_truth_for_every_stage():
    train = _mixed_train()
    # the LoRA, the gate, the mask bank and the detectors all call this with the same arguments
    a = subset_samples(train, 0.1, 2, None)
    assert a == subset_samples(train, 0.1, 2, None) == real_subset(train, 0.1, 2)
    assert len(a) == 15 and set(a) <= set(train)
    assert subset_samples(train, 0.1, 3, None) != a  # a different draw is a different subset


def test_subset_samples_can_restrict_to_one_dataset_and_keeps_order_at_full_fraction():
    train = _mixed_train()
    kv = subset_samples(train, 1.0, 0, ("kvasir",))
    assert len(kv) == 90 and {s.source for s in kv} == {"kvasir"} and kv == train[:90]
    half = subset_samples(train, 0.5, 0, ("kvasir",))
    assert len(half) == 45 and {s.source for s in half} == {"kvasir"}
    assert subset_samples(train, 1.0, 0, None) == train


def test_external_set_never_overlaps_the_training_sources():
    samples = _mixed_train()
    ext = select_external(samples, "cvc", ("kvasir",))
    assert [s.uid for s in ext] == [f"cvc_{i}" for i in range(60)]  # whole dataset, frame order kept
    with pytest.raises(ValueError):
        select_external(samples, "cvc", ("kvasir", "cvc"))
    assert select_external(samples, "cvc", None)  # no restriction declared -> allowed


def test_block_bootstrap_resamples_contiguous_runs_of_frames():
    rng = np.random.default_rng(0)
    assert resample_indices(rng, 50, 1).shape == (50,)
    idx = resample_indices(rng, 60, 20)
    assert len(idx) == 60 and idx.min() >= 0 and idx.max() < 60
    # every consecutive pair is either +1 (inside a block) or the start of a new block
    for k, (a, b) in enumerate(zip(idx[:-1], idx[1:]), start=1):
        if k % 20:
            assert b == a + 1
    short_tail = resample_indices(rng, 50, 20)  # 20 + 20 + 10 frames
    assert short_tail.max() < 50


def test_run_is_done_can_require_the_external_result(tmp_path):
    p = tmp_path / "real_s0.json"
    write_json_atomic(str(p), {"val": {}, "neg": {}})
    assert run_is_done(str(p))
    assert not run_is_done(str(p), ("val", "neg", "ext"))  # an old in-domain-only file must be redone
    write_json_atomic(str(p), {"val": {}, "neg": {}, "ext": {}})
    assert run_is_done(str(p), ("val", "neg", "ext"))


def _fake_run(arm: str, seed: int, shift: float):
    """A result file with just enough predictions for `report` (one box per frame, score shifted per arm)."""
    gt = [[10.0, 10.0, 50.0, 50.0]]
    val = [dict(boxes=[[10, 10, 50, 50]], scores=[round(0.9 - shift, 3)]) for _ in range(12)]
    ext = [dict(boxes=[[10, 10, 50, 50] if (i % 4) else [0, 0, 5, 5]], scores=[round(0.8 - shift, 3)]) for i in range(40)]
    m = lambda ds, g: summarize([match_image(d["boxes"], d["scores"], g) for d in ds])  # noqa: E731
    return dict(
        arm=arm, seed=seed, n_real=100, n_extra=100 if arm != "real" else 0, steps=600, frac=0.1,
        val=m(val, gt), val_by_source={"kvasir": m(val, gt)}, neg=dict(fp_per_img=0.1, frac_imgs_with_fp=0.1, n=5),
        ext=m(ext, gt), ext_name="cvc",
        preds=dict(val=val, gts=[gt] * 12, ext=ext, ext_gts=[gt] * 40, neg=[], sources=["kvasir"] * 12),
    )


def test_report_pools_directories_and_scores_the_external_set(tmp_path):
    for d, seed in (("a", 0), ("b", 1)):
        for arm, shift in (("real", 0.0), ("neg", 0.05), ("synth", 0.1)):
            write_json_atomic(str(tmp_path / d / f"{arm}_s{seed}.json"), _fake_run(arm, seed, shift))
    md = tmp_path / "pooled_ext.md"
    args = argparse.Namespace(dir=[str(tmp_path / "a"), str(tmp_path / "b")], boot=30, set="ext", block=20, md=str(md))
    report(args)
    text = md.read_text()
    assert "External dataset: **cvc** (40 frames" in text and "blocks of 20" in text
    assert "| real | 2 |" in text and "| synth | 2 |" in text  # runs from both directories were pooled
    assert "synth − real" in text and "neg − real" in text
    # in-domain mode reads the same files without needing the external keys
    args_val = argparse.Namespace(dir=[str(tmp_path / "a")], boot=30, set="val", block=20, md=None)
    report(args_val)
    assert (tmp_path / "a" / "report.md").exists()
