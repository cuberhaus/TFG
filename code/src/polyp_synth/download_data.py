"""Download the public datasets used for polyp synthesis.

    python -m polyp_synth.download_data            # from code/src

Fetches into ``code/data/polyp_synth/raw`` (git-ignored):

* Kvasir-SEG        1000 polyp frames + real masks   (Simula, research use)
* CVC-ClinicDB       612 polyp frames + real masks   (CC BY 4.0)
* HyperKvasir        polyp-free colon frames used as backgrounds
                     (``bbps-2-3`` and ``cecum`` classes, CC BY 4.0, ~3.9 GB zip,
                     deleted after extraction)

None of these are LDPolypVideo; to train on LDPolypVideo instead, see
``polyp_synth.data.PolypSample`` (bbox-only samples are converted to ellipse
masks automatically).
"""
from __future__ import annotations

import os
import zipfile

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

from polyp_synth.data import DEFAULT_ROOT  # noqa: E402

NEGATIVE_CLASSES = {
    "labeled-images/lower-gi-tract/quality-of-mucosal-views/bbps-2-3/": "bbps-2-3",
    "labeled-images/lower-gi-tract/anatomical-landmarks/cecum/": "cecum",
}


def main(root: str = DEFAULT_ROOT, keep_zip: bool = False):
    os.makedirs(root, exist_ok=True)
    for repo, name in (("berkaytrhn/kvasir-seg", "kvasir-seg"), ("Aeoo/CVC-ClinicDB", "CVC-ClinicDB")):
        print("downloading", repo)
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=os.path.join(root, name), max_workers=16)

    neg_root = os.path.join(root, "negatives")
    if all(os.path.isdir(os.path.join(neg_root, c)) for c in NEGATIVE_CLASSES.values()):
        print("negatives already present")
        return
    print("downloading HyperKvasir labeled images (3.9 GB)")
    zpath = hf_hub_download(
        "SimulaMet-HOST/HyperKvasir", "hyper-kvasir-labeled-images.zip", repo_type="dataset", local_dir=os.path.join(root, "hyperkvasir")
    )
    with zipfile.ZipFile(zpath) as z:
        for member in z.namelist():
            for prefix, cls in NEGATIVE_CLASSES.items():
                if member.startswith(prefix) and member.endswith(".jpg"):
                    dest = os.path.join(neg_root, cls, os.path.basename(member))
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with z.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
    if not keep_zip:
        os.remove(zpath)
    print("done")


if __name__ == "__main__":
    main()
