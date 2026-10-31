#!/usr/bin/env python3
"""
Qwen-Image 2.1 reference pipeline, for parity checks of the C code.

Runs the official diffusers components (Qwen3-VL text encoding through
QwenImage21Pipeline.encode_prompt, QwenImage21Transformer2DModel with its
KV cache, FlowMatchEulerDiscreteScheduler, AutoencoderKLQwenImage21) the
way QwenImage21Pipeline.__call__ does for text-to-image, in float32, and
dumps the intermediate tensors as raw little-endian float32.

The text encoder is freed before the transformer loads, and the
transformer before the VAE, so a float32 run fits in 64 GB.

Usage:
    python debug/qimage_reference.py -d ~/GenAI/models/Qwen-Image-2.1-Turbo \\
        -p "a fish" -W 256 -H 256 --seed 42 --blocks -o /tmp/qimage_ref

    # Start from the noise iris generates for a seed (see debug/dump_noise.c):
    python debug/qimage_reference.py ... --noise /tmp/noise.bin

    # Only decode a latent [64, H/16, W/16]:
    python debug/qimage_reference.py -d MODEL -W 256 -H 256 --decode latent.bin -o OUT

Outputs in the -o directory:
    tokens.txt, drop_idx.txt  encoder token ids (whole template) and the
                              number of leading system tokens dropped
    emb.bin                   text embeddings after the drop [L, 4096]
    noise.bin                 initial noise [64, H/16, W/16]
    sigmas.txt                scheduler sigmas (num_steps + 1 values)
    rope_cos.bin, rope_sin.bin  first-step RoPE tables [L + N, 64]
    step<i>_v.bin             velocity at step i [N, 64]
    step<i>_x.bin             latent tokens after step i [N, 64]
    block<l>_s<i>.bin         (--blocks) hidden states after block l at steps
                              0 and 1: [L + N, dim] at step 0, [N, dim] after
    latent.bin                final latent [64, H/16, W/16] (normalized)
    image.bin, image.png      decoded image [4, H, W] in [-1, 1]

L is the text length after the drop, N = (H/16) * (W/16). The C side reads
emb.bin and noise.bin with: ./iris -d MODEL -e emb.bin --noise noise.bin
-W W -H H -o out.png. Pass --emb FILE to skip the text encoder.
"""

import argparse
import gc
import json
import os

import numpy as np
import torch


def dump(path, t):
    t.detach().to(torch.float32).contiguous().cpu().numpy().tofile(path)


def encode(model_dir, prompt, device, out):
    """Text embeddings [L, dim] via the official pipeline's encode_prompt."""
    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline.from_pretrained(
        model_dir, transformer=None, vae=None, torch_dtype=torch.float32).to(device)
    emb, mask, image_pad_mask = pipe.encode_prompt(prompt=prompt, device=device)
    if mask is not None and not bool(mask.all()):
        raise SystemExit("unexpected padding in a single-prompt encoding")
    if bool(image_pad_mask.any()):
        raise SystemExit("unexpected image tokens in a text-only prompt")

    text = pipe.prompt_template_t2i.format(prompt if prompt else " ")
    ids = pipe.processor(text=[text], return_tensors="pt").input_ids[0].tolist()
    with open(out("tokens.txt"), "w") as f:
        f.write("\n".join(str(i) for i in ids) + "\n")
    with open(out("drop_idx.txt"), "w") as f:
        f.write(f"{pipe._drop_idx}\n")
    print(f"tokens: {len(ids)}, dropped: {pipe._drop_idx}, embeddings: {tuple(emb.shape[1:])}")

    emb = emb[0].to(torch.float32).cpu()
    del pipe
    gc.collect()
    return emb


def denoise(model_dir, emb, h, w, args, device, out):
    """The QwenImage21Pipeline.__call__ denoising loop (text-to-image, no
    CFG, KV cache on). Returns the final latent tokens [N, 64]."""
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

    transformer = QwenImage21Transformer2DModel.from_pretrained(
        model_dir, subfolder="transformer", torch_dtype=torch.float32).to(device).eval()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
    print("scheduler config:", dict(scheduler.config))
    channels = transformer.config.in_channels
    n = h * w

    if args.noise:
        noise = torch.from_numpy(np.fromfile(args.noise, dtype=np.float32)).view(1, channels, h, w)
    else:
        noise = torch.randn((1, 1, channels, h, w), generator=torch.Generator().manual_seed(args.seed),
                            dtype=torch.float32)[:, 0]
    dump(out("noise.bin"), noise[0])
    latents = noise.view(1, channels, n).transpose(1, 2).to(device)   # _pack_latents

    with open(os.path.join(model_dir, "model_index.json")) as f:
        sigmas = json.load(f).get("sample_sigmas")
    if sigmas is None:
        steps = args.steps or 40
        sigmas = np.linspace(1.0, 1 / steps, steps)
    elif args.steps:
        print(f"note: Turbo's sample_sigmas override --steps {args.steps}")
    mu = calculate_shift(n, scheduler.config.get("base_image_seq_len", 256),
                         scheduler.config.get("max_image_seq_len", 4096),
                         scheduler.config.get("base_shift", 0.5),
                         scheduler.config.get("max_shift", 1.15))
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    np.savetxt(out("sigmas.txt"), scheduler.sigmas.cpu().numpy(), fmt="%.8f")
    timesteps = scheduler.timesteps

    emb = emb.unsqueeze(0).to(device)
    txt_len = emb.shape[1]
    img_mask = torch.cat([torch.zeros(1, txt_len, dtype=torch.bool),
                          torch.ones(1, n // 4, dtype=torch.bool)], dim=1).to(device)
    img_shapes = [[(1, h, w)]]
    kv_cache = QwenImage21KVCache(len(transformer.transformer_blocks))

    step = {"i": 0}
    hooks = []

    def rope_hook(_module, _inputs, output):
        if step["i"] == 0:
            dump(out("rope_cos.bin"), output.real)
            dump(out("rope_sin.bin"), output.imag)
    hooks.append(transformer.pos_embed.register_forward_hook(rope_hook))
    if args.blocks:
        for idx, block in enumerate(transformer.transformer_blocks):
            def block_hook(_module, _inputs, output, idx=idx):
                if step["i"] < 2:
                    dump(out(f"block{idx}_s{step['i']}.bin"), output[0])
            hooks.append(block.register_forward_hook(block_hook))

    scheduler.set_begin_index(0)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            step["i"] = i
            v = transformer(hidden_states=latents, timestep=t.expand(1).to(latents.dtype) / 1000,
                            encoder_hidden_states=emb, encoder_hidden_states_mask=None,
                            img_shapes=img_shapes, img_mask=img_mask, kv_cache=kv_cache,
                            kv_cache_mode="extract" if i == 0 else "cached",
                            return_dict=False)[0][:, -n:]
            latents = scheduler.step(v, t, latents, return_dict=False)[0]
            dump(out(f"step{i}_v.bin"), v[0])
            dump(out(f"step{i}_x.bin"), latents[0])
            print(f"step {i + 1}/{len(timesteps)}: sigma {float(scheduler.sigmas[i]):.6f}")

    for hk in hooks:
        hk.remove()
    del transformer, kv_cache
    gc.collect()
    return latents[0].cpu()


def decode(model_dir, latent, h, w, device, out):
    """latent [64, h, w] (normalized) -> image [4, 16h, 16w] in [-1, 1]."""
    from diffusers import AutoencoderKLQwenImage21
    from diffusers.image_processor import VaeImageProcessor

    vae = AutoencoderKLQwenImage21.from_pretrained(
        model_dir, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    z = latent.view(1, -1, 1, h, w).to(device)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device)
    with torch.no_grad():
        image = vae.decode(z * std + mean, return_dict=False)[0][:, :, 0]
    dump(out("image.bin"), image[0])
    pil = VaeImageProcessor(vae_scale_factor=16).postprocess(image, output_type="pil")[0]
    print(f"image mode: {pil.mode}")
    pil.save(out("image.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--model-dir", required=True)
    ap.add_argument("-p", "--prompt", default="")
    ap.add_argument("-W", "--width", type=int, default=256)
    ap.add_argument("-H", "--height", type=int, default=256)
    ap.add_argument("--steps", type=int, default=None,
                    help="base model steps (default 40; Turbo has a fixed grid)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", help="raw float32 noise [64, H/16, W/16]")
    ap.add_argument("--emb", help="raw float32 text embeddings [L, context_in_dim] (skips the encoder)")
    ap.add_argument("--decode", help="only decode this raw float32 latent [64, H/16, W/16]")
    ap.add_argument("--blocks", action="store_true", help="dump per-block hidden states")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("-o", "--out-dir", default="/tmp/qimage_ref")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    def out(name):
        return os.path.join(args.out_dir, name)

    # The pipeline floors sizes to multiples of 32 (one token is 16x16 px,
    # and the target occupies 2x2 token slots of the encoder sequence).
    width, height = args.width // 32 * 32, args.height // 32 * 32
    h, w = height // 16, width // 16

    if args.decode:
        latent = torch.from_numpy(np.fromfile(args.decode, dtype=np.float32))
        decode(args.model_dir, latent, h, w, args.device, out)
        return

    if args.emb:
        with open(os.path.join(args.model_dir, "transformer", "config.json")) as f:
            dim = json.load(f).get("context_in_dim", 4096)
        emb = torch.from_numpy(np.fromfile(args.emb, dtype=np.float32)).view(-1, dim)
    else:
        emb = encode(args.model_dir, args.prompt, args.device, out)
    dump(out("emb.bin"), emb)

    tokens = denoise(args.model_dir, emb, h, w, args, args.device, out)
    latent = tokens.transpose(0, 1).reshape(-1, h, w)   # _unpack_latents
    dump(out("latent.bin"), latent)
    decode(args.model_dir, latent, h, w, args.device, out)


if __name__ == "__main__":
    main()
