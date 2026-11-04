"""Inference helpers: load the fine-tuned inpainting pipeline and paint polyps."""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from PIL import Image

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
DEFAULT_PROMPT = "endoscopic colonoscopy image of a polyp, kvasir style"


DEFAULT_STEPS = 16


def make_scheduler(config):
    """First-order Euler with few steps.

    Measured on held-out polyps: higher-order samplers (DPM++/UniPC/DDIM) and
    more steps hallucinate ~2.5x more fine texture inside the polyp than real
    frames have; Euler at ~16 steps matches the real texture statistics.
    """
    from diffusers import EulerDiscreteScheduler

    return EulerDiscreteScheduler.from_config(config, timestep_spacing="linspace")


def load_pipeline(lora_dir: str | None = None, base: str = BASE_MODEL, device: str = "cuda", dtype=torch.bfloat16):
    """Load SD-inpainting, optionally with a PEFT LoRA adapter merged in."""
    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base, variant="fp16", torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )
    pipe.scheduler = make_scheduler(pipe.scheduler.config)
    if lora_dir:
        from peft import PeftModel

        peft_unet = PeftModel.from_pretrained(pipe.unet, lora_dir)
        pipe.unet = peft_unet.merge_and_unload().to(dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def inpaint_polyp(
    pipe,
    image512: np.ndarray,
    hole512: np.ndarray,
    prompt: str = DEFAULT_PROMPT,
    seed: int = 0,
    steps: int = DEFAULT_STEPS,
    guidance: float = 4.0,
    size: int = 512,
) -> np.ndarray:
    """Paint a polyp inside ``hole512`` of ``image512``. Returns uint8 HxWx3."""
    gen = torch.Generator(device=pipe.device).manual_seed(int(seed))
    out = pipe(
        prompt=prompt,
        image=Image.fromarray(image512),
        mask_image=Image.fromarray((hole512 > 0).astype(np.uint8) * 255),
        height=size,
        width=size,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen,
        strength=1.0,
        output_type="np",
    ).images[0]
    return (out * 255).round().clip(0, 255).astype(np.uint8)


def composite(orig: np.ndarray, gen: np.ndarray, region: np.ndarray, feather: float = 2.0) -> np.ndarray:
    """Blend ``gen`` into ``orig`` inside ``region`` with a soft edge.

    Pixels outside the (feathered) region remain bit-exact copies of ``orig``.
    """
    a = region.astype(np.float32)
    if feather > 0:
        a = cv2.GaussianBlur(a, (0, 0), feather)
        a = np.clip(a * 1.25, 0, 1)  # keep the interior fully generated
        a = a * (cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)  # never leak far outside
    a = a[..., None]
    return (a * gen.astype(np.float32) + (1 - a) * orig.astype(np.float32)).round().clip(0, 255).astype(np.uint8)


def grid(rows: list[list[np.ndarray]], pad: int = 4, bg: int = 255) -> np.ndarray:
    """Tile a list of rows of equally sized HxWx3 uint8 images."""
    h, w = rows[0][0].shape[:2]
    H = len(rows) * h + (len(rows) + 1) * pad
    W = max(len(r) for r in rows) * w + (max(len(r) for r in rows) + 1) * pad
    canvas = np.full((H, W, 3), bg, np.uint8)
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            y, x = pad + i * (h + pad), pad + j * (w + pad)
            canvas[y : y + h, x : x + w] = im
    return canvas


def save_jpg(arr: np.ndarray, path: str, quality: int = 93):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path, quality=quality)
