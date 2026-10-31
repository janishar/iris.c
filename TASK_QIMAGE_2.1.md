# Task: add Qwen-Image 2.1 and Turbo support

Status: not started. No C code has been written. Delete this file when the
work is merged.

## Goal

Add text-to-image for `Qwen/Qwen-Image-2.1` (40 steps) and
`Qwen/Qwen-Image-2.1-Turbo` (8 steps) to iris, on MPS, CUDA/ROCm, BLAS and
generic builds. Output must match the official diffusers pipeline within the
tolerance the FLUX.3 tests use.

Base and Turbo are one architecture: their transformer configs are identical.
Turbo adds a fixed sigma grid and needs no separate model family.

Out of scope for this task (see "Phase 2" at the end): reference images and
editing. Also skip CFG (off by default upstream), the prompt-rewrite models
(`Qwen-Image-2.1-PE-*`), flex attention and VAE tiling.

## What is already in place

| Item | Where |
|---|---|
| Base weights | `~/GenAI/models/Qwen-Image-2.1` |
| Turbo weights | `~/GenAI/models/Qwen-Image-2.1-Turbo` |
| FLUX.3 weights, for regression | `~/GenAI/models/flux-3-image-klein` |
| Python env with diffusers 0.42.0.dev0, transformers 5.17 | `~/GenAI/qwen-image-2.1-studio/.venv/bin/python` (use read-only) |

Check these before relying on them:

- The local Turbo `text_encoder/` and `vae/` are symlinks to the base model's
  files, so the local VAE is the base F32 one. A clean Hub download has a
  single `text_encoder/model.safetensors` and a BF16 VAE. Support both.
- FLUX.2 and Z-Image weights were not found locally, so `make test` cannot run
  those tests here. Say so in the final report rather than implying they
  passed. `make mps` built cleanly on the starting commit (0b02b52). No test
  baseline has been recorded yet, so record one first.
- Machine: M5 Pro, 64 GB. A float32 reference run needs the text encoder freed
  before the transformer loads.

Line numbers below were correct at commit 0b02b52.

## Order of work

Do this one component at a time, in one checkout. Parity runs load tens of GB
and all builds share the same object files, so parallel builds or model runs
will collide.

1. Record the baseline, then write `debug/qimage_reference.py` (see below). Run
   it on Turbo at 256x256, seed 42, dumping per-block states.
2. Text encoder: match `tokens.txt` and `emb.bin`.
3. Scheduler: match `sigmas.txt` for Turbo and for base at a few step counts.
4. Transformer: CPU path first, compare `block<l>_s<i>.bin` and `step<i>_v.bin`,
   then the device path.
5. VAE: decode `latent.bin`, compare with `image.bin`.
6. Routing, CLI, downloads, tests and docs.
7. Check end to end, and that FLUX.3 did not regress.

## Reference script

Write `debug/qimage_reference.py`, modelled on `debug/flux3_reference.py`. Run
the official `QwenImage21Pipeline` in float32 on the CPU, encoding text first and
freeing the encoder before loading the transformer. Accept `--noise FILE` to
start from C noise, and a way to only decode a given latent.

Dump as raw little-endian float32:

| File | Contents |
|---|---|
| `tokens.txt`, `drop_idx.txt` | encoder token ids, leading system tokens dropped |
| `emb.bin` | text embeddings `[txt_seq, 4096]` |
| `noise.bin` | initial noise `[64, H/16, W/16]` |
| `sigmas.txt` | schedule, `num_steps + 1` values |
| `rope_cos.bin`, `rope_sin.bin` | first-step RoPE tables `[txt_seq + img_seq, 64]` |
| `step<i>_v.bin`, `step<i>_x.bin` | velocity and latent tokens per step `[img_seq, 64]` |
| `block<l>_s<i>.bin` | hidden states after block l, steps 0 and 1 |
| `latent.bin` | final latent `[64, H/16, W/16]`, before denormalization |
| `image.bin`, `image.png` | decoded image, `[4, H, W]` in [-1, 1] |

With the KV cache on, block dumps cover `[txt_seq + img_seq, 4096]` at step 0
and `[img_seq, 4096]` after it. Capture tensors with forward hooks on the
transformer, its blocks and `pos_embed`, and `callback_on_step_end` for latents.

## 1. Loader and routing

- New flag `is_qimage` in `iris_load_dir` ([iris.c:272](iris.c)). Detect
  `QwenImage21Pipeline` in `model_index.json`. Turbo is marked by a
  `sample_sigmas` key, which is also how upstream tells them apart.
- `model_index.json` has no `is_distilled`, so the current logic would mark
  both as base models and send them down the CFG path. Neither uses CFG.
- Text dim comes from `context_in_dim` (4096), not `joint_attention_dim`.
- Defaults: 40 steps base, 8 Turbo, no guidance, `is_non_commercial = 1`.
- The VAE load at iris.c:502 will fail on this checkpoint, so branch there.
- Add `qimage` cases to every `is_zimage` / `is_flux3` site in `iris.c`,
  `iris_cli.c` and `main.c`, including step previews (`iris_step_image_vae`
  assumes the Flux VAE type).
- Sizes must be multiples of 32, because the transformer groups image tokens
  2x2. Upstream floors them; the Flux paths floor to 16.
- `IRIS_VAE_MAX_DIM` (1792) is below the model's recommended sizes (up to
  2752x1536), so this VAE needs its own limit.

Naming: `_qimage` suffix per CLAUDE.md. New files `iris_transformer_qimage.c`
and `iris_vae_qimage.c`.

## 2. Text encoder

The encoder (Qwen3-VL 8B text model) has the same shape as the 9B Qwen3: hidden
4096, 36 layers, 32 query heads, 8 KV heads, `rope_theta` 5e6, read from config.
Tensors are under `model.language_model.`, already handled.

- Add `QWEN3_EXTRACT_QIMAGE` in `qwen3_output_layers` (iris_qwen3.c:296):
  `layers[0] = num_layers - 1`. Upstream needs the last layer **before** the
  final norm. iris loads that norm but never applies it, so this works as is.
- Template, built as a raw string (no `<think>`):

  ```
  <|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
  ```

- Drop the leading system tokens from the output. Upstream takes the count from
  the tokenized system message; `drop_idx.txt` has the expected value.
- An empty prompt becomes `" "`.
- Only real tokens reach the transformer, as Z-Image does at iris.c:720.
- Variable lengths hit two limits: the GPU causal kernels (CLAUDE.md FLUX.3
  pitfall 2) and the `seq_len <= 512` check at iris_qwen3.c:1136. Suggested
  approach, unproven: pad to a multiple of 80 like FLUX.3, then trim. Causal
  attention should leave real tokens unchanged; confirm against `emb.bin`.
- Loader gaps:
  - Add a `processor/tokenizer.json` fallback (near iris_qwen3.c:2024).
  - Add single-file `model.safetensors` support (iris_qwen3.c:1764 only falls
    back to two shards). Needed for Hub Turbo downloads.

## 3. Scheduler

Euler: `x += (sigma_next - sigma) * v`. The timestep passed to the model is the
sigma itself.

- Turbo: read the 8 values from `model_index.json`, append 0, no shift.
  `--steps` must not change them:

  `1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568`

- Base: `linspace(1, 1/n, n)`, exponential time shift (reuse
  `generalized_time_snr_shift`, iris_sample.c:160), stretch to terminal 0.02,
  append 0.
  - `mu = 0.5 + (seq - 256) * 0.4 / 7936`, where `seq` is the target token
    count. It is not clamped, so it exceeds 0.9 at high resolution.
  - The stretch is `1 - (1 - s) / ((1 - s_last) / (1 - 0.02))`. That formula is
    from memory of diffusers, so check it against `sigmas.txt`.

## 4. Transformer

Config: dim 4096, heads 32, head_dim 128, 32 blocks, mlp 12288, in/out 64,
patch 1, RoPE axes `[16,56,56]`, theta 10000 (not in the config), eps 1e-6.
All linears have no bias.

Sequence is `[TEXT | TARGET]`. Target tokens are raster order, no patchify.

- Timestep: sinusoidal 256 dims of `sigma * 1000`, cos half first, max period
  10000, then `time_text_embed.timestep_embedder.linear_1`, SiLU, `linear_2`.
- Text input: RMSNorm with effective weight `txt_in.text_norm.weight + 1`
  (zero-centered, unlike every other norm here), `in_layer`, GELU
  (`approximate="tanh"`), `out_layer`.
- `img_in`: linear from 64.
- `modulation.1`: `linear(silu(temb))` gives `[scale1, gate1, scale2, gate2]`,
  one projection shared by all blocks. Text uses the embedding of t=0, target
  uses t.
- Block, sequential (unlike FLUX.3's parallel form):

  ```
  a = attn(LayerNorm(x) * (1 + scale1));  x += tanh(gate1) * a
  m = mlp(LayerNorm(x) * (1 + scale2));   x += tanh(gate2) * m
  ```

  No shift. `mlp(n) = out(silu(gate_layer(n)) * proj(n))`, three linears.
- Attention: separate `to_q/k/v`, `to_out.0`; per-head RMSNorm on Q and K using
  `attn.norm_q/k.weight` directly; RoPE on Q and K only; scale `1/sqrt(128)`.
  iris's consecutive-pair rotation matches upstream's complex multiply.
- RoPE: text token i sits at `(i, i, i)`. Target tokens sit at
  `(T, h, w)` with T the text length, and `h`, `w` centred:
  `range(-(H - H//2), H//2)`.
- Output: `norm_out.linear(silu(temb))` gives scale only,
  `LayerNorm(x) * (1 + scale)`, then `proj_out` to 64. Target tokens only.

### Design decision: prefill once, then one code path

Upstream runs step 0 in "extract" mode and later steps in "cached" mode. Text
never attends to the target, so its activations do not depend on the target or
the step. Run the text through all 32 blocks once before sampling, with t=0
modulation and causal attention (`iris_dev_attention` with `causal=1`), storing
post-RoPE K and V per block. Every step then computes target queries against
`[prefix KV, target KV]`, including step 0.

The math is the same, but results match upstream only to rounding. Upstream
documents that toggling its KV cache changes samples in reduced precision, so
compare against a cache-enabled reference in float32.

This also means each transformer call has a single modulation vector, so no new
per-token modulation API is needed. Prefix cost is about 0.16 GB for 300 tokens.

Device API notes:
- `iris_dev_attention` already takes `seq_q != seq_k`, but q, k and v share one
  row stride, so prefix and target K/V must be laid out contiguously. Working
  out that layout per block is the main design job here.
- `iris_dev_layer_norm_mod` works with `shift = NULL`.
- `iris_dev_residual` has no tanh. Pass `tanh(gate)` as its gate vector.
- FLUX.3's device path shows weight fusing, but its blocks are parallel. These
  are sequential, so the fusing differs.

## 5. VAE

Not the Flux VAE. It is Wan-derived and handles a single frame, so everything
is 2D: `QwenImage21CausalConv3d` subclasses `nn.Conv2d` and stores 2D weights.
Ignore every `time_conv` weight and all feature caching. Load decoder tensors
only.

Config: z 64, decoder base 144, `dim_mult [1,2,4,8,8]`, 2 res blocks, RGBA out.
Latent denormalization is `latent * std + mean`, with 64 values each in
`vae/config.json`.

Decode: `post_quant_conv` (1x1), `conv_in`, mid block, 5 up blocks,
`norm_out`, SiLU, `conv_out`, clamp to [-1, 1].

- Mid block: res, attention, res. There is attention even though `attn_scales`
  is empty. It is single-head over all pixels at latent resolution: norm,
  `to_qkv` (1x1 to 3c), SDPA with scale `1/sqrt(c)`, `proj` (1x1), plus identity.
- Up block channels: 1152→1152, 1152→1152, 1152→576, 576→288, 288→144. The
  first four upsample. Each has 3 res blocks.
- Res block: `h = conv_shortcut(x)` if the channels differ, else x; then norm1,
  SiLU, conv1, norm2, SiLU, conv2; add h.
- Norm is per-pixel RMS, not GroupNorm: `x / rms(x over channels) * gamma`.
- Up block: keep `x_copy`, run res blocks, then the upsampler (nearest 2x, conv
  at `upsampler.resample.1`), then add the shortcut of `x_copy`.
- Shortcut (`QwenImage21DupUp3D`): `repeat_interleave` channels by
  `out * factor / in`, fold into `[out, t, s, s]`, pixel-shuffle, and with
  `first_chunk=True` keep only temporal slot `factor_t - 1`. This is easy to get
  wrong. `factor_t` comes from the reversed `temperal_downsample`; check that
  against `QwenImage21Decoder3d` and `AutoencoderKLQwenImage21.__init__`.

`iris_dev_conv2d` and `iris_dev_upsample2x` exist on CUDA, ROCm and Metal, so
one device path can serve all GPUs. That differs from `iris_vae.c`, whose device
path is compiled out on MPS. New kernels needed: per-pixel RMS norm with f32
output, and pixel shuffle.

Tensor names for the mid-block attention are inferred from the class. Confirm
them against the safetensors header.

Output has 4 channels, always. `iris_image` and PNG support RGBA; PPM drops
alpha. Check what the reference does for opaque images (it prints the mode).

## 6. Plumbing

- Makefile `SRCS` and dependency lines.
- `download_model.py` and `download_model.sh` entries. Note the licence there.
- `run_test.py`: `is_qimage_model_dir`, a `--qimage-model-dir` flag, and a test.
- Test vector in `test_vectors/`, generated from **iris's** noise for the seed
  (`iris_init_noise`) via `--noise`, as the FLUX.3 vectors were.
- README and CLAUDE.md: variants, file structure, architecture, pitfalls.

## Acceptance

- `emb.bin`, `sigmas.txt`, step velocities and `image.bin` match the reference,
  with the tolerances you used stated in the final report.
- `./iris -d ~/GenAI/models/Qwen-Image-2.1-Turbo -p "a fish" -W 1024 -H 1024 -o
  /tmp/q.png` gives a sensible image. Do the same for base.
- Device and CPU paths agree on one MPS and one BLAS build.
- FLUX.3 tests still match the recorded baseline.
- CUDA/ROCm builds compile if a toolchain is available. If not, say they were
  not built.

## Expected cost

Unmeasured. Block count and width are close to Flux 9B, so 8-step Turbo at 1024
should take about twice 4-step 9B. At 2048 there are 4x the tokens and about
16x the attention cost. Memory at 2K is unknown, including the VAE buffers.

## Phase 2: reference images and editing (separate task)

Much bigger than Phase 1:

- VAE encoder (residual down blocks with `QwenImage21AvgDown3D`).
- Qwen3-VL vision tower: 27 layers, dim 1152, patch 16, merge 2, deepstack at
  8/16/24.
- Qwen3 changes: `<|image_pad|>` substitution and 3-axis MRoPE (sections
  `[24,20,20]`, interleaved) on the CPU, Metal and device paths. Deepstack
  injection into early text layers is assumed from Qwen3-VL's design; confirm it
  in the transformers source.
- Transformer: interleaved text and image blocks. Text after an image needs a
  causal query offset, which `iris_dev_attention` lacks.
- Prefix KV memory: about 2.3 GB per 1 MP reference, about 22 GB for 10.
- Reference preparation: RGBA, resized to about 1 MP on the 32-pixel grid;
  composited over white for the vision encoder only.

Vision tokens are overwritten by latents in the transformer, so the tower only
affects the text that follows the image. It still cannot be skipped: the
checkpoint was trained with it.

## Licence

Qwen Research License: non-commercial only. Commercial use needs a separate
licence from Qwen. "Qwen" may be used descriptively but not as the primary name
of a derivative product.

## Sources

- Model cards: https://huggingface.co/Qwen/Qwen-Image-2.1 and
  https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo
- Project: https://github.com/QwenLM/Qwen-Image-2.1
- Diffusers docs: https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21
- Diffusers source, also installed in the venv above:
  - `src/diffusers/models/transformers/transformer_qwenimage21.py`
  - `src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py`
  - `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`
