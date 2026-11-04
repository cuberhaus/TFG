"""Download every pretrained weight the jobs need, on a machine WITH internet.

The BSC clusters cannot reach the internet, so Stable Diffusion (Hugging Face) and the torchvision
detector checkpoints must be cached here and copied over (``bsc.sh build`` / ``bsc.sh push stage``).

    HF_HOME=<dir>/hf TORCH_HOME=<dir>/torch python prefetch_weights.py

After downloading, the Stable Diffusion cache is re-opened in a subprocess with HF_HUB_OFFLINE=1,
exactly like a compute node would, so a missing file fails here and not in a queued job.
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

from polyp_synth.pipeline import BASE_MODEL  # noqa: E402

# Everything polyp_synth.pipeline.load_pipeline and polyp_synth.train load, as they load it.
LOAD_SD = f"""
import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
B = {BASE_MODEL!r}
StableDiffusionInpaintPipeline.from_pretrained(B, variant="fp16", torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
CLIPTokenizer.from_pretrained(B, subfolder="tokenizer")
CLIPTextModel.from_pretrained(B, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16)
AutoencoderKL.from_pretrained(B, subfolder="vae", variant="fp16", torch_dtype=torch.float16)
UNet2DConditionModel.from_pretrained(B, subfolder="unet", variant="fp16", torch_dtype=torch.float16)
DDPMScheduler.from_pretrained(B, subfolder="scheduler")
print("stable diffusion cache complete")
"""


def prefetch_stable_diffusion():
    subprocess.run([sys.executable, "-c", LOAD_SD], check=True)  # online: downloads whatever is missing
    subprocess.run([sys.executable, "-c", LOAD_SD], check=True, env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})


def prefetch_torchvision():
    """DEFAULT weights of every detector the repo builds (polyp_synth + legacy clases.model_utils)."""
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights,
        RetinaNet_ResNet50_FPN_V2_Weights, SSDLite320_MobileNet_V3_Large_Weights,
    )

    for w in (FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights,
              RetinaNet_ResNet50_FPN_V2_Weights, SSDLite320_MobileNet_V3_Large_Weights):
        w.DEFAULT.get_state_dict(progress=False)
        print("torchvision weights cached:", w.DEFAULT.url.rsplit("/", 1)[-1])


if __name__ == "__main__":
    print("HF_HOME   =", os.environ.get("HF_HOME", "(default)"))
    print("TORCH_HOME =", os.environ.get("TORCH_HOME", "(default)"))
    prefetch_stable_diffusion()
    prefetch_torchvision()
