"""Fine-tune Stable Diffusion inpainting (LoRA) to paint polyps into masks.

Usage (from ``code/src``)::

    python -m polyp_synth.train --run-name v1 --steps 6000

The model sees a colonoscopy crop with the polyp region blanked out and must
repaint it. At generation time the blanked region is simply a mask of the
polyp we want to insert, so the bounding box of the new polyp is known exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from polyp_synth.data import PolypInpaintDataset, index_samples, split_samples, CAPTIONS
from polyp_synth.pipeline import BASE_MODEL, composite, grid, inpaint_polyp, make_scheduler, save_jpg

OUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "out", "polyp_synth", "runs"))

LORA_TARGETS = [
    "to_q", "to_k", "to_v", "to_out.0",
    "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2",
    "conv1", "conv2", "conv_shortcut", "time_emb_proj",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="v1")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--hole-weight", type=float, default=3.0, help="extra loss weight inside the hole")
    p.add_argument("--p-hull", type=float, default=0.25)
    p.add_argument("--dilate-min", type=float, default=0.012)
    p.add_argument("--dilate-max", type=float, default=0.03)
    p.add_argument("--zoom-min", type=float, default=0.6)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default=None)
    return p.parse_args()


@torch.no_grad()
def sample_grid(core_unet, vae, text_encoder, tokenizer, val_ds, idxs, step, out_dir, device, dtype):
    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=core_unet,
        scheduler=make_scheduler({"num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012,
                                  "beta_schedule": "scaled_linear", "prediction_type": "epsilon"}),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    core_unet.eval()
    rows = []
    # trainable adapter weights are fp32 while the frozen base is bf16 -> autocast
    with torch.autocast("cuda", dtype=dtype):
        for i in idxs:
            it = val_ds.get(int(i))
            im, hole = it["image"], it["hole"]
            masked = (im * (1 - hole[..., None])).astype(np.uint8)
            gen = inpaint_polyp(pipe, im, hole, prompt=CAPTIONS[it["source"]], seed=1, steps=16, guidance=4.0)
            comp = composite(im, gen, hole)
            rows.append([im, masked, comp])
    core_unet.train()
    save_jpg(grid(rows), os.path.join(out_dir, "samples", f"step_{step:06d}.jpg"))


def main():
    args = parse_args()
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTextModel, CLIPTokenizer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device, dtype = "cuda", torch.bfloat16
    out_dir = os.path.join(OUT_ROOT, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"), indent=2)

    kw = dict(root=args.data_root) if args.data_root else {}
    samples = index_samples(**kw)
    train_s, val_s = split_samples(samples)
    print(f"train {len(train_s)}  val {len(val_s)}")
    train_ds = PolypInpaintDataset(
        train_s, size=args.size, train=True, dilate_frac=(args.dilate_min, args.dilate_max),
        p_hull=args.p_hull, zoom_min=args.zoom_min,
    )
    val_ds = PolypInpaintDataset(val_s, size=args.size, train=False)
    # fixed validation items: a spread over both sources
    kv = [i for i, s in enumerate(val_s) if s.source == "kvasir"]
    cvc = [i for i, s in enumerate(val_s) if s.source == "cvc"]
    sample_idxs = kv[:: max(1, len(kv) // 4)][:4] + cvc[:: max(1, len(cvc) // 2)][:2]

    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder", variant="fp16", torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae", variant="fp16", torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet", variant="fp16", torch_dtype=dtype).to(device)
    noise_sched = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)
    text_encoder.eval()
    vae.eval()
    if not args.no_grad_ckpt:
        unet.enable_gradient_checkpointing()

    cfg = LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=LORA_TARGETS, modules_to_save=["conv_in"], lora_dropout=0.0)
    unet = get_peft_model(unet, cfg)
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    unet.print_trainable_parameters()
    unet.train()
    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.999))
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))) * 0.95 + 0.05)
    )

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        drop_last=True, persistent_workers=True, pin_memory=True)
    it = iter(loader)
    log = open(os.path.join(out_dir, "metrics.jsonl"), "a")
    ema, t0 = None, time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        pix = batch["pixel_values"].to(device, dtype, non_blocking=True)
        hole = batch["hole"].to(device, non_blocking=True)
        with torch.no_grad():
            latents = vae.encode(pix).latent_dist.sample() * vae.config.scaling_factor
            masked_lat = vae.encode(pix * (1 - hole).to(dtype)).latent_dist.mode() * vae.config.scaling_factor
            hole_ds = F.interpolate(hole, size=latents.shape[-2:])  # nearest, same as the diffusers pipeline
            ids = tokenizer(list(batch["caption"]), padding="max_length", max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors="pt").input_ids.to(device)
            enc = text_encoder(ids)[0]
        noise = torch.randn_like(latents)
        t = torch.randint(0, noise_sched.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy = noise_sched.add_noise(latents, noise, t)
        inp = torch.cat([noisy, hole_ds.to(dtype), masked_lat], dim=1)
        with torch.autocast("cuda", dtype=dtype):
            pred = unet(inp, t, enc).sample
        w = 1.0 + args.hole_weight * hole_ds
        loss = (((pred.float() - noise.float()) ** 2).mean(1, keepdim=True) * w).sum() / w.sum() / 1.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 25 == 0:
            rec = dict(step=step, loss=round(ema, 5), lr=sched.get_last_lr()[0], sec=round(time.time() - t0, 1),
                       mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 100 == 0:
                print(rec, flush=True)
        if step % args.sample_every == 0 or step == 1:
            sample_grid(unet.get_base_model(), vae, text_encoder, tokenizer, val_ds, sample_idxs, step, out_dir, device, dtype)
        if step % args.save_every == 0 or step == args.steps:
            unet.save_pretrained(os.path.join(out_dir, f"ckpt-{step}"))
    unet.save_pretrained(os.path.join(out_dir, "final"))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
