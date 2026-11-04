# LinkedIn post (long form)

I went back to my bachelor's thesis (FIB-UPC) to fix the one part that didn't work.

The detection side worked. The data-augmentation side didn't: I asked a CycleGAN to turn a rectangle (a polyp's bounding box) into a colonoscopy frame, and it drew beige stripes. My thesis says so openly: the change a mask asks the model to make is simply too big.

So I changed the question. Don't generate the colon. Keep the real frame and paint only the polyp: Stable Diffusion inpainting plus a small LoRA (3.6% of the weights, about 30 minutes of training on a laptop GPU). I choose the mask, so the bounding-box label comes for free.

How I checked it isn't just pretty pictures: I held out 160 real polyps, blanked each one, repainted it from its true mask, and asked a Faster R-CNN trained only on real polyps to find it:
• Stock Stable Diffusion inpainting: 6.4%
• Real frames: 85.0%
• My fine-tuned version: 87.3%

What this does not show (yet): it is trained on the public Kvasir-SEG and CVC-ClinicDB datasets, not on my thesis dataset; "a detector finds it" is not "a clinician can't tell"; very large polyps are still weak; and I haven't tested whether adding these images actually improves detection. That is the next experiment.

What I like most: the fix wasn't a bigger model. It was reading my own conclusion and doing the opposite.

Code and numbers: github.com/cuberhaus/TFG

#MedicalAI #ComputerVision #StableDiffusion #DeepLearning #UPC

---

# Shorter alternative

My bachelor's thesis drew stripes instead of polyps. I went back to it with Stable Diffusion inpainting + a 30-minute LoRA: keep the real colonoscopy frame, paint only the polyp. A detector trained only on real polyps now finds 87.3% of the synthetic ones (real frames: 85.0%; stock Stable Diffusion: 6.4%). Not tested yet: whether it improves detection. github.com/cuberhaus/TFG
