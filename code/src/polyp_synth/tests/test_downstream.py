"""CPU-only tests for the detection metrics / planning helpers used by the downstream experiment."""
import numpy as np
import pytest

from polyp_synth.data import PolypSample
from polyp_synth.downstream import (
    IOUS, average_precision, iou_matrix, match_image, plan_steps, real_subset, run_is_done, summarize,
    summarize_negatives, write_json_atomic,
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
