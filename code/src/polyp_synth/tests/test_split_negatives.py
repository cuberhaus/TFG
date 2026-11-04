"""CPU-only tests for the held-out polyp-free frame split."""
import os

import pytest

from polyp_synth.split_negatives import split_negatives


def _make_negatives(root, n_a=30, n_b=20):
    for cls, n in (("cecum", n_a), ("bbps-2-3", n_b)):
        d = root / "negatives" / cls
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{cls}_{i:03d}.jpg").write_bytes(b"x")


def _names(root, sub):
    return sorted(os.listdir(root / sub))


def test_split_is_disjoint_complete_and_sized(tmp_path):
    _make_negatives(tmp_path)
    n_gen, n_held = split_negatives(str(tmp_path), n_heldout=12, seed=0)
    assert (n_gen, n_held) == (38, 12)
    gen, held = set(_names(tmp_path, "negatives_gen")), set(_names(tmp_path, "negatives_heldout"))
    assert not gen & held
    assert len(gen | held) == 50


def test_split_is_deterministic_and_idempotent(tmp_path):
    _make_negatives(tmp_path)
    split_negatives(str(tmp_path), n_heldout=12, seed=0)
    first = _names(tmp_path, "negatives_heldout")
    split_negatives(str(tmp_path), n_heldout=12, seed=0)  # re-running must not change or duplicate anything
    assert _names(tmp_path, "negatives_heldout") == first
    assert len(_names(tmp_path, "negatives_gen")) == 38
    split_negatives(str(tmp_path), n_heldout=12, seed=1)
    assert _names(tmp_path, "negatives_heldout") != first


def test_symlinks_are_relative_and_resolve(tmp_path):
    _make_negatives(tmp_path)
    split_negatives(str(tmp_path), n_heldout=5)
    link = tmp_path / "negatives_gen" / _names(tmp_path, "negatives_gen")[0]
    assert not os.path.isabs(os.readlink(link))
    assert link.exists()  # the relative link still points at a real file


def test_missing_negatives_gives_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="download_data"):
        split_negatives(str(tmp_path))
