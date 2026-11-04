"""Mask-conditioned polyp synthesis via fine-tuned diffusion inpainting.

Replaces the bounding-box -> image CycleGAN approach used originally in the
thesis (which could not learn to hallucinate a whole colon from a rectangle)
with a model that keeps the real surrounding tissue and only synthesises the
polyp inside a mask.
"""
