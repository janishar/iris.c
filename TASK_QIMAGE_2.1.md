# Task: add Qwen-Image 2.1 and Turbo support

Status: text-to-image implemented and verified on the real weights on MPS,
CUDA and BLAS (see "Measured on the real weights"). ROCm has never been
compiled. Phase 2 (reference images, any number up to memory) is implemented on
MPS and BLAS, on CUDA it is not yet compiled. Delete this file when the work is
merged.

## Implementation status

What exists:

| Milestone | Where |
|---|---|
| M0 reference | `debug/qimage_reference.py` (`--emb`, `--noise`, `--decode`, `--blocks`), `debug/dump_noise.c` |
| M1 detection, scheduler | `iris_load_dir` (`QwenImage21Pipeline`, `sample_sigmas`, `context_in_dim`), `iris_qimage_sched_load` / `iris_schedule_qimage` in iris_sample.c |
| M2 text encoder | `QWEN3_EXTRACT_QIMAGE`, `qwen3_tokenize_chat_system` (system block, drop index, tail-preserving truncation), `processor/tokenizer.json`, single `model.safetensors` |
| M3 transformer CPU | `iris_transformer_qimage.c` (prefill once, image-only steps) |
| M4 transformer device | same file under `IRIS_DEV`; new `iris_dev_copy_2d` in iris_cuda.cu and iris_dev_metal.m |
| M5 VAE decoder | `iris_vae_qimage.c` (CPU, BLAS) |
| M6 plumbing | iris.c routing, iris_cli.c, main.c help, Makefile, download scripts, run_test.py smoke test, README, CLAUDE.md |

## Measured on the real weights

M5 Pro, 64 GB, against `debug/qimage_reference.py` (diffusers float32, CPU) from
the noise iris generates for seed 42, prompt "A fluffy orange cat sitting on a
windowsill". `debug/dump_qimage.c` runs each step on the reference latent, so
velocity errors do not accumulate; "free run" uses C velocities only.

| Check | BLAS | MPS |
|---|---|---|
| Text embeddings, relative L2, 6 prompts of 9 to 555 rows (empty, Chinese, punctuation) | 4.1e-6 to 6.0e-5 | 4.1e-6 to 2.1e-5 |
| Same, max abs (values around 8.7 in mean magnitude) | 0.095 | 0.072 |
| Embedding row count | exact | exact |
| Velocity relative L2, Turbo 256x256, 8 steps | 1.7e-6 to 3.1e-6 | 3.4e-3 to 1.2e-2 |
| Velocity relative L2, Turbo 384x256, 8 steps | 2.0e-6 to 7.4e-6 | 3.1e-3 to 8.9e-3 |
| Velocity relative L2, base 256x256, 40 steps | 1.3e-6 to 4.5e-6 | 2.4e-3 to 1.3e-2 |
| Velocity max abs, all runs | 1.4e-4 | 0.20 |
| Free-run latent relative L2, all runs | 7.8e-6 to 1.6e-5 | 2.0e-2 to 3.3e-2 |
| VAE decode of the reference latent, max per channel | 1/255 | 1/255 |
| `./iris -e --noise` vs reference PNG, mean / max (of 255), Turbo | 0.0003 / 1 | 1.28 / 140 |
| Same, base | 0.0007 / 1 | 0.70 / 114 |

Upstream diffusers in bf16 against its own float32 run has velocity relative L2
6.3e-3 to 1.6e-2 on the same Turbo inputs, so the MPS error is bf16 rounding.

Scheduler: base within 1.24e-7 of diffusers for 4, 20, 40 and 50 steps at
256x256, 1024x1024, 2048x2048 and 2752x1536. Turbo's config has shift and
stretch off, so its grid is unchanged. `--steps 1` on base divides by zero in
upstream's stretch, and iris skips the stretch there.

Full runs, MPS, 1024x1024: Turbo 52 s, base 139 s, about 2.5 s per step, peak
memory footprint 26 GB. VAE decode runs on the CPU and takes 22.8 s.

FLUX.3: output byte-identical to `main` (41fc94e) on MPS; BLAS within 1/255 of
the reference vectors. REPL: the first generation of a prompt is byte-identical
to the CLI. A repeated prompt uses the 4-bit embedding cache, as on the other
distilled models, and differs by a mean of 1.66/255.

The GPU VAE decoder (`iris_dev_pixel_rms_norm`, `iris_dev_dup_up_add`) decodes
the reference latents within a mean of 0.15/255 (max 12) on Metal and CUDA, which
is bf16 rounding; the CPU decoder stays within 1/255.

CUDA, RTX PRO 6000 Blackwell, CUDA 13, no OpenBLAS: velocity relative L2 3.2e-3
to 1.4e-2 on all three runs, free-run latent 1.7e-2 to 2.2e-2, reference test
mean 1.07. Turbo 1024x1024 takes 4.5 s (1.6 s denoising, 0.4 s decode), base
11.1 s, and 2048x2048, 2752x1536 and 1536x2752 take 13 s. With the VAE on the GPU,
Turbo on MPS takes 26.7 s at 1024x1024 (was 52 s), base 117 s (was 139 s).

MPS at 2048x2048 and 2752x1536: 150 s (128 s denoising), peak footprint 44 GB, no
transformer eviction needed.

M5 Pro Neural Accelerators: nothing in iris targets them, and MPSGraph is not
documented to use them. Measured on real iris runs, differencing base and Turbo:
107 J per 1024x1024 step, 3.2 pJ/MAC, matching MLX with its Neural Accelerator
kernels (3.3) and not without them (9.1) or f32 (13.1). bf16 is what keeps this
path: f32 GEMMs and attention run at a quarter of the rate. Benchmarks were on
macOS 27.2, and MPSGraph routing can change with the OS.

Not verified: ROCm (never compiled; downloading its toolchain was blocked), generic
at runtime (it builds).
`make test` needs numpy in `python3`, and its three Flux.2 tests need
`flux-klein-4b`, which is not on this machine.

## Remaining work

Compile `iris_cuda.cu` with hipcc, then Phase 2.

## Goal

Add text-to-image for `Qwen/Qwen-Image-2.1` (40 steps) and
`Qwen/Qwen-Image-2.1-Turbo` (8 steps) to iris, on MPS, CUDA/ROCm, BLAS and
generic builds. Output must match the official diffusers pipeline within the
tolerance the FLUX.3 tests use.

Base and Turbo are one architecture with identical transformer configs. Turbo
adds a fixed `sample_sigmas` grid in `model_index.json`. It is not a separate
model family.

Out of scope (see "Phase 2" at the end): reference images and editing. Also
skip CFG (`true_cfg_scale` defaults to 1.0 upstream), the prompt-rewrite models
(`Qwen-Image-2.1-PE-*`), flex attention and VAE tiling.

## Verification status of this plan

The architecture notes below were checked line by line against diffusers
`main` (2026-10-31):

- `src/diffusers/models/transformers/transformer_qwenimage21.py`
- `src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py`
- `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`
- `src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`

The C-side references were checked against this repository at commit 41fc94e.

The checkpoint configs (`model_index.json`, `scheduler/scheduler_config.json`,
`transformer/config.json`, `vae/config.json`) could not be fetched. Hugging Face
was not reachable from the planning machine. Values below that come from class
defaults are marked **[config]**. Read them from the files at runtime and
confirm them against the local checkpoints first.

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
  passed. Record a test baseline (`make test`, FLUX.3 only) before any change.
- Machine: M5 Pro, 64 GB. A float32 reference run needs the text encoder freed
  before the transformer loads.

## Cost estimate (computed from the config defaults)

- Transformer: 32 blocks of `4*dim^2` (attention) plus `3*dim*3*dim` (SwiGLU)
  weights, which is about 7.0 B parameters, 14 GB in bf16. The text encoder
  (Qwen3-VL 8B text model) is about 16 GB in bf16.
- At 1024x1024 a step has 4096 target tokens: about 57 TFLOP of GEMMs plus
  about 9 TFLOP of attention, so roughly 66 TFLOP per step and 0.5 PFLOP for
  8-step Turbo. The text prefix is computed once per prompt, not per step.
- At 2048x2048 there are 4x the tokens: 4x the GEMMs and about 16x the attention.
- VAE decode at 1024x1024 peaks at a few GB of f32 activations. The last up
  block reads 288 ch at 1024x1024, which is 1.2 GB per buffer. At 2048 that is
  4x, so VAE memory, not the transformer, limits the maximum size.

## Order of work (original plan; the acceptance criteria apply to the real-weight runs)

Work one component at a time, in one checkout. Parity runs load tens of GB and
all builds share the same object files, so parallel builds or model runs will
collide. Each milestone ends with a passing comparison and a commit.

| # | Milestone | Done when |
|---|---|---|
| M0 | Baseline + `debug/qimage_reference.py` | Turbo 256x256 seed 42 dumps exist; FLUX.3 `make test` baseline recorded |
| M1 | Loader detection + scheduler | `sigmas.txt` matches for Turbo and base at 4, 20, 40 steps, at 256 and 1024 px |
| M2 | Text encoder | `tokens.txt`, `drop_idx.txt` exact; `emb.bin` within tolerance |
| M3 | Transformer, CPU | `rope_*.bin`, `block<l>_s0/s1.bin`, `step<i>_v.bin` within tolerance |
| M4 | Transformer, device | device and CPU paths agree; same reference tolerances |
| M5 | VAE decoder, CPU then device | `latent.bin` decodes to `image.bin` within tolerance |
| M6 | Routing, CLI, REPL, downloads, tests, docs | end-to-end images; `make test` passes; FLUX.3 matches its baseline |

## M0. Reference script

Write `debug/qimage_reference.py`, modelled on `debug/flux3_reference.py`. Run
the official `QwenImage21Pipeline` in float32 on the CPU with `use_kv_cache=True`
(the default), encoding text first and freeing the encoder before loading the
transformer. Accept `--noise FILE` to start from C noise, and `--decode FILE` to
only decode a given latent.

Dump as raw little-endian float32:

| File | Contents |
|---|---|
| `tokens.txt`, `drop_idx.txt` | encoder token ids (full template), and `pipe._drop_idx` |
| `emb.bin` | text embeddings after the drop `[L, 4096]` |
| `noise.bin` | initial noise `[64, H/16, W/16]` |
| `sigmas.txt` | `scheduler.sigmas`, `num_steps + 1` values |
| `rope_cos.bin`, `rope_sin.bin` | first-step RoPE tables `[L + N, 64]` (real/imag of `pos_embed` output) |
| `step<i>_v.bin`, `step<i>_x.bin` | velocity and latent tokens per step `[N, 64]` |
| `block<l>_s<i>.bin` | hidden states after block l, steps 0 and 1 |
| `latent.bin` | final latent `[64, H/16, W/16]`, before denormalization |
| `image.bin`, `image.png` | decoded image `[4, H, W]` in [-1, 1] |

`L` is the text length after the drop and `N = (H/16) * (W/16)`. Block dumps
cover `[L + N, 4096]` at step 0 (extract mode) and `[N, 4096]` after it. Capture
them with forward hooks on the transformer blocks and `pos_embed`, and use
`callback_on_step_end` for the latents. Also print the scheduler config the
pipeline loaded. M1 needs it.

## M1. Loader detection and scheduler

### Detection (`iris_load_dir`, iris.c:272)

- New flag `is_qimage`. Detect `"QwenImage21Pipeline"` in `model_index.json`.
  Turbo is marked by a non-null `sample_sigmas` array in that file. Parse and
  store it (`ctx->qimage_sigmas`, `ctx->qimage_num_sigmas`).
- `model_index.json` has no `is_distilled`, so the current logic would mark
  both models as base and send them down the CFG path (iris.c:308). Set
  `is_distilled = 1` for both. Neither uses CFG.
- Text dim is `context_in_dim` (4096) from `transformer/config.json`, not
  `joint_attention_dim`.
- Defaults: 40 steps base, `len(sample_sigmas)` for Turbo, guidance 0,
  `is_non_commercial = 1`.
- The Flux VAE load (iris.c:506) would fail on this checkpoint. Branch there
  to `iris_vae_load_qimage()`.

### Scheduler (`iris_schedule_qimage`, iris_sample.c)

The spec's earlier claim, "Turbo: append 0, no shift", is not what upstream
does. The pipeline passes `sample_sigmas` (Turbo) or `linspace(1, 1/n, n)`
(base) into the same `scheduler.set_timesteps(sigmas=..., mu=mu)`, which then
applies the scheduler's configured processing to either:

```
sigmas = turbo ? sample_sigmas : linspace(1, 1/n, n)
if use_dynamic_shifting:                         # [config]
    mu = base_shift + (N - base_seq) * (max_shift - base_shift) / (max_seq - base_seq)
    sigmas = exp(mu) / (exp(mu) + (1/sigmas - 1))        # time_shift_type "exponential"
             (or mu / (mu + (1/sigmas - 1)) for "linear")
else:
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
if shift_terminal:                               # [config]
    sigmas = 1 - (1 - sigmas) / ((1 - sigmas[-1]) / (1 - shift_terminal))
append 0
```

- Read every value from `scheduler/scheduler_config.json` (`use_dynamic_shifting`,
  `base_image_seq_len`, `max_image_seq_len`, `base_shift`, `max_shift`,
  `shift`, `shift_terminal`, `time_shift_type`). Fall back to the diffusers
  defaults (256, 4096, 0.5, 1.15, 1.0, none, exponential) when a key is absent.
- `N` is the target token count `(H/16) * (W/16)`, not clamped.
- The old notes put base at `mu = 0.5 + (N - 256) * 0.4 / 7936` with terminal
  0.02, i.e. `max_image_seq_len` 8192, `max_shift` 0.9, `shift_terminal` 0.02.
  Treat that as a hypothesis for the base config.
- Turbo's listed grid (`1.0, 0.978453, ..., 0.414568`) only survives unchanged
  if Turbo's scheduler config disables dynamic shifting, has `shift` 1.0 and no
  `shift_terminal`. The stretch would map the last value to 0.02. Verify with
  `sigmas.txt`. Do not special-case Turbo in C: the generic code gives the
  right answer either way.
- `--steps` on Turbo: upstream ignores `num_inference_steps` when
  `sample_sigmas` is set. Ignore it too and print a notice. `--linear` and
  `--power` keep overriding, as for the other families.
- Euler: `x += (sigma_next - sigma) * v`. The model timestep is the sigma
  itself: the pipeline passes `t / 1000` with `t = sigma * 1000`.
- Reuse `generalized_time_snr_shift` (iris_sample.c:160) for the exponential
  shift.

## M2. Text encoder (iris_qwen3.c, iris_qwen3_tokenizer.c)

The encoder (Qwen3-VL 8B text model) has the same shape as the Flux 9B Qwen3:
hidden 4096, 36 layers, 32 query heads, 8 KV heads, `rope_theta` 5e6, all read
from config. Tensors are under `model.language_model.`, which
`qwen3_detect_variant` already handles. That also sets `max_seq_len` to
`QWEN3_VL_MAX_SEQ_LEN` (1280). Since 41fc94e, Metal attention covers all 1280
positions, so the old `seq_len <= 512` concern is gone.

- **Extraction**: add `QWEN3_EXTRACT_QIMAGE` to `qwen3_output_layers`
  (iris_qwen3.c:296) with `layers[0] = num_layers - 1`. Upstream wants the last
  decoder layer's output **before** the final norm. It hooks the norm away
  because transformers 5 would normalize `hidden_states[-1]`. iris loads
  `norm_weight` (iris_qwen3.c:1800) but never applies it, so this works as is.
  The layer loop already stops at the last extracted layer, which is now 35.
- **Template**: a raw string, not `apply_chat_template`. Upstream says the two
  tokenize differently. There is no `<think>` block:

  ```
  <|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
  ```

  `qwen3_tokenize_chat` (iris_qwen3_tokenizer.c:991) has no system message.
  Add `qwen3_tokenize_chat_system(tok, system, prompt, ...)`, or a `system`
  argument, and share the existing piece-by-piece code. Special tokens and
  `\n` are pre-token boundaries under the Qwen2 split regex, so tokenizing the
  pieces separately equals tokenizing the whole string.
- **Drop index**: upstream drops `len(apply_chat_template(system message))`
  tokens, i.e. the tokens of
  `<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n`.
  Compute it in C the same way, by tokenizing that block alone, and check it
  against `drop_idx.txt`. Everything after it is kept, including the trailing
  `<|im_start|>assistant\n` tokens.
- An empty prompt becomes `" "`.
- **Truncation**: upstream never truncates. iris must cap at 17280 tokens.
  Truncate the prompt body, not the tail, so the template suffix survives, and
  warn as the tokenizer already does since d3c1060.
- **Padding**: pad to a multiple of `QWEN3_FLUX3_PAD_MULTIPLE` (80) as FLUX.3
  does, then keep only the real tokens. Pads sit after every real token and
  the encoder is causal, so they cannot change real tokens. Keep
  `pad_at_one` (iris_qwen3.c:280) FLUX3-only: pad RoPE positions do not matter
  here. Generalize `qwen3_encode_text_ex` so its FLUX3 branches become
  "VL modes", and return the post-drop embeddings `[L, 4096]`. Only real tokens
  reach the transformer, as for Z-Image (iris.c:720).
- **Loader gaps**:
  - Tokenizer fallback: `processor/tokenizer.json`, after the two existing
    paths (iris_qwen3.c:2024).
  - Single-file `text_encoder/model.safetensors` (iris_qwen3.c:1764 only knows
    the index file and the two-shard pattern). Hub Turbo downloads need it.
- **Acceptance**: `tokens.txt` exact, `drop_idx` exact, `emb.bin` max abs
  error within the FLUX.3 encoder tolerance, on CPU and on one GPU build.

## M3/M4. Transformer (new `iris_transformer_qimage.c`)

Config **[config]**: dim 4096 (`num_attention_heads * attention_head_dim`),
heads 32, head_dim 128, 32 blocks, `mlp_ratio` 3 (12288), in/out channels 64,
`patch_size` 1, `axes_dims_rope` `[16, 56, 56]`, eps 1e-6,
`causal_condition` true. RoPE theta 10000 is hardcoded upstream, not in the
config. No linear has a bias, including the timestep MLP.

### Forward, as verified upstream

The sequence is `[TEXT (L) | TARGET (N)]`, with target tokens in raster order
and no patchify. With no condition images, `img_mask` is L false slots plus
N/4 true slots, each expanded 4x, so the joint layout is exactly
`[L text, N target]`.

1. **Timestep embedding**: `arg = 1000 * sigma * exp(-ln(10000) * j / 128)`,
   `j < 128`; `emb = [cos(arg) | sin(arg)]` (cos first), then
   `time_text_embed.timestep_embedder.linear_1`, SiLU, `linear_2`. This gives
   `temb_t` for target tokens. `temb_0` (sigma = 0) is used for text.
2. **Modulation**: `mod = modulation.1(silu(temb))` -> `[scale1 | gate1 |
   scale2 | gate2]`, each `dim` wide. One vector is shared by **all 32 blocks**.
   The docstring says "slices", but the code passes the same tensor to each
   block. Text uses `mod(temb_0)`, target uses `mod(temb_t)`.
3. **Text input**: `n = x * rsqrt(mean(x^2) + 1e-6) * (txt_in.text_norm.weight + 1)`
   (zero-centered weight, the only such norm here), then `in_layer`,
   GELU-tanh, `out_layer`.
4. **Target input**: `img_in` (64 -> dim).
5. **Block** (sequential, unlike FLUX.3's parallel block):

   ```
   a = attn(LayerNorm(x) * (1 + scale1));  x += tanh(gate1) * a
   m = mlp(LayerNorm(x) * (1 + scale2));   x += tanh(gate2) * m
   ```

   LayerNorm has no affine part and no shift term.
   `mlp(n) = out(silu(gate_layer(n)) * proj(n))`.
6. **Attention**: `to_q/k/v` -> per-head RMSNorm (`attn.norm_q/k.weight`,
   eps 1e-6) -> RoPE on Q and K -> SDPA, scale `1/sqrt(128)` -> `to_out.0`.
   Text queries are **causal** over text keys. Target queries see every text
   key and every target key (bidirectional within the target).
7. **RoPE**: per axis with dim `d`, pair `j < d/2` has frequency
   `10000^(-2j/d)`. The three axes are concatenated (8 + 28 + 28 = 64 pairs) and
   rotate **consecutive** pairs (`view_as_complex` of `[..., 64, 2]`). That is
   `iris_dev_head_norm_rope(..., split_half = 0)`, which fuses step 6's norm.
   Positions: text token `i` -> `(i, i, i)`. Target token `(r, c)` ->
   `(L, r - (Hl - Hl/2), c - (Wl - Wl/2))`, with `Hl = H/16`, `Wl = W/16`. That
   is `range(-(Hl - Hl//2), Hl//2)`, which can be negative. Upstream's table
   wraps negative indices onto a separately built negative half, and that equals
   computing the angle from the signed position directly.
8. **Output**: `scale = norm_out.linear(silu(temb_t))` (scale only, no shift),
   `LayerNorm(x) * (1 + scale)`, `proj_out` -> 64. Target tokens only.

### Design: prefill once, then one target-only path

Text never attends to the target, and text is modulated from t = 0, so text
activations depend on neither the target nor the step. That is why upstream can
use a KV cache. Do the same, but compute the prefix once **before** sampling
rather than inside step 0:

- `qimage_prefill(tf, txt_emb, L)`: run the text alone through the 32 blocks
  with `mod(temb_0)` and causal attention. Store each block's post-norm,
  post-RoPE K and V, `[L, 2*dim]` bf16 per block: about 0.5 MB per token in
  total, so 50 MB for 100 tokens and 670 MB for 1280. Block 31's text attention
  output and MLP are never used, so the prefill can stop after block 31's K/V
  projection.
- Every step, including step 0: target queries attend to
  `[prefix K/V, target K/V]` with no mask.
- Same math as upstream with the cache on. It matches only to rounding, since
  upstream documents that the cache changes reduced-precision samples. Compare
  against the float32 cache-on reference.
- Each step then has exactly one modulation vector, so no per-token modulation
  API is needed.
- Keep the prefix in the transformer struct, keyed on the embedding contents.
  Repeated generations with the same prompt (REPL, `--seed` sweeps) skip the
  prefill.

### Device memory layout (resolves the earlier "main design job")

`iris_dev_attention` requires q, k and v to share one row stride. Use a fused
QKV scratch buffer `qkv [L + N, 3*dim]` bf16, one per model, reused by every
block:

1. The target QKV GEMM (fused `[to_q; to_k; to_v]` weight, assembled on the
   device at load like FLUX.3's fused weights) writes rows `[L, L + N)`.
2. Copy the block's cached prefix K|V `[L, 2*dim]` into rows `[0, L)`, columns
   `[dim, 3*dim)`.
3. `iris_dev_head_norm_rope` on the Q and K column ranges of the target rows
   (`ld = 3*dim`, RoPE rows for the target only).
4. `iris_dev_attention(out, dim, q = qkv + L*3dim, k = qkv + dim, v = qkv + 2dim,
   ldi = 3*dim, seq_q = N, seq_k = L + N, heads = kv_heads = 32, causal = 0)`.
   This hits the flash kernel (head_dim 128, non-causal).

Step 2 needs a **new device API**: `iris_dev_copy_2d(dst, dst_stride, src,
src_stride, row_bytes, rows)`, a device-to-device copy that stays
stream-ordered. CUDA/HIP: `MemcpyAsync2D` with `DeviceToDevice` on `g_stream`.
Metal: a blit encoder in the current command buffer. A CPU `memcpy` would force
a flush per block (32 per step). The rejected alternative is per-block
`[L + N]` K/V buffers, which need no copy but cost 2.2 GB at 1 MP and 8.6 GB at
4 MP.

The prefill uses the same buffer with `seq_q = seq_k = L`, `causal = 1`.
Causal attention misses the flash kernel and takes the GEMM + softmax fallback,
which is fine at L <= 1280 and done once per prompt.

Other kernels already exist:

- `iris_dev_layer_norm_mod` with `shift = NULL`.
- `iris_dev_residual` with a `tanh(gate)` vector computed on the host (4 x 4096
  values per step).
- `iris_dev_swiglu` on a fused `[gate_layer; proj]` GEMM output (`ldi = 2*mlp`),
  as FLUX.3 does.

The text-input GELU-tanh runs once per prompt, so the CPU is fine for it.

### CPU path

Same structure on f32/bf16 weights with BLAS GEMMs. The per-head attention
takes a causal flag for the prefill. Reuse `iris_kernels.c` helpers (LayerNorm,
RMSNorm, SiLU, GELU) rather than adding copies.

### API

```c
iris_transformer_qimage_t *iris_transformer_load_qimage(const char *model_dir);
void  iris_transformer_free_qimage(iris_transformer_qimage_t *tf);
int   iris_transformer_prefill_qimage(iris_transformer_qimage_t *tf,
                                      const float *txt_emb, int txt_len);
float *iris_transformer_forward_qimage(iris_transformer_qimage_t *tf,
                                       const float *x, int hl, int wl, float sigma);
float *iris_sample_euler_qimage(void *tf, float *z, int hl, int wl,
                                const float *txt_emb, int txt_len,
                                const float *sigmas, int num_steps, ...);
```

### Acceptance

RoPE tables exact to float rounding. Block states and step velocities within
tolerance at steps 0 and 1, CPU first, then device vs CPU on one MPS and one
BLAS build.

## M5. VAE decoder (new `iris_vae_qimage.c`)

It is not the Flux VAE. It is Wan-derived and specialized to one frame:
`QwenImage21CausalConv3d` subclasses `nn.Conv2d`, stores 2D weights
`[out, in, k, k]`, and pads symmetrically (k=3, pad 1). Ignore every
`time_conv` weight: with one frame and `first_chunk = True`, upstream never runs
them. Load decoder tensors and `post_quant_conv` only.

Config **[config]**: `z_dim` 64, `decoder_base_dim` 144, `dim_mult
[1,2,4,8,8]`, `num_res_blocks` 2, `temperal_downsample [F,T,T,T]`,
`is_residual` true, `out_channels` 4, `patch_size` null. `latents_mean` and
`latents_std` have 64 values each. Denormalize with `latent * std + mean`.
If `patch_size` is set and > 1, fail clearly. Upstream would unpatchify.

Decode:

```
x = post_quant_conv(z)                         # 1x1, 64->64, bias
x = conv_in(x)                                 # 3x3, 64->1152
x = mid: res(1152), attn(1152), res(1152)
x = up_blocks[0..4]
x = conv_out(silu(norm_out(x)))                # 144->4
clamp to [-1, 1]
```

- Up block channels (`dims = 144 * [8,8,8,4,2,1]`, `is_residual` keeps in_dim):
  1152->1152, 1152->1152, 1152->576, 576->288, 288->144. The first four
  upsample. Each has 3 res blocks (`num_res_blocks + 1`).
- **Norm** (`QwenImage21RMS_norm`): `x / max(||x||_2, 1e-12) * sqrt(C) * gamma`
  over the channels of each pixel. That is RMS normalization with no eps inside
  the root. Gamma is stored `[C,1,1,1]`. No bias.
- **Res block**: `h = conv_shortcut(x)` (1x1 with bias, only when channels
  differ), then norm1, SiLU, conv1 (3x3), norm2, SiLU, conv2 (3x3), `+ h`. The
  shortcut reads the block input, before norm1.
- **Mid attention**: one head over all latent pixels, head dim = C = 1152.
  `n = norm(x)`; `to_qkv` 1x1 conv with bias -> channels `[q | k | v]`; SDPA with
  scale `1/sqrt(1152)`; `proj` 1x1 with bias; `+ x`. Head dim 1152 rules out
  flash attention. At 2048 px there are 16384 latent pixels and a 1 GiB f32
  score matrix, so chunk the queries on CPU.
- **Up block** (`QwenImage21ResidualUpBlock`): `x_copy = x`; 3 res blocks;
  upsampler = nearest 2x (`nearest-exact` equals nearest at 2x) then the 3x3
  conv at `upsampler.resample.1` (out->out); then `x += dup_up(x_copy)`.
- **DupUp shortcut, in closed form.** Earlier notes said "easy to get wrong".
  Derived from `QwenImage21DupUp3D.forward` with `first_chunk = True`:

  ```
  ft      = temperal_upsample[i] ? 2 : 1      # temperal_upsample = reversed(temperal_downsample)
  repeats = out_ch * ft * 4 / in_ch
  out[o, 2y+i, 2x+j] = in[(((o*ft + ft-1)*2 + i)*2 + j) / repeats, y, x]
  ```

  With the default config (`ft = 2,2,2,1` for up blocks 0..3):
  - blocks 0, 1 (1152->1152, repeats 8): `src = o`, plain nearest 2x.
  - block 2 (1152->576, repeats 4): `src = 2o + 1`, odd input channels only.
  - block 3 (576->288, repeats 2): `src = 2o + i`, which depends on the output
    row parity.

  Implement the general formula: one CPU loop and one device kernel. Test it
  against a tiny PyTorch run of `QwenImage21DupUp3D` for each block shape.
- Conv weights are `[out, in, k, k]`. `iris_dev_conv2d` wants channels-last
  `[out, (ky*k + kx)*in + c]` in bf16, so repack at load, as `iris_vae.c` does.
  The F32 base VAE converts at load. The BF16 Hub VAE uploads directly.
- Tensor names are inferred from the module tree. Confirm them against the
  safetensors header before writing the loader: `decoder.conv_in`,
  `decoder.mid_block.resnets.{0,1}.{norm1.gamma,conv1,norm2.gamma,conv2}`,
  `decoder.mid_block.attentions.0.{norm.gamma,to_qkv,proj}`,
  `decoder.up_blocks.{i}.resnets.{j}.{...,conv_shortcut}`,
  `decoder.up_blocks.{i}.upsampler.resample.1`, `decoder.norm_out.gamma`,
  `decoder.conv_out`, `post_quant_conv`.
- **Device path**: one implementation for CUDA, ROCm and Metal on
  `iris_dev.h`. `iris_dev_conv2d` and `iris_dev_upsample2x` exist on all three.
  This differs from `iris_vae.c`, whose device path is compiled out on MPS.
  New kernels:
  1. `iris_dev_pixel_rms_norm(out, in, gamma, hw, c, silu)`: f32 in and out,
     the norm above, optional fused SiLU. The existing `iris_dev_rms_norm`
     writes bf16 and adds eps.
  2. `iris_dev_dup_up(out, in, h, w, cin, cout, ft)`: the formula above.

  The mid attention can use `iris_dev_linear` for the 1x1 convs plus
  `iris_dev_attention` with head_dim 1152, which takes the GEMM fallback.
- **Output**: always 4 channels (RGBA). `iris_image` and PNG support RGBA, PPM
  drops alpha. For opaque results, consider writing RGB when every alpha value
  rounds to 255. Check what the reference image mode is first.
- **Size cap**: `IRIS_VAE_MAX_DIM` (1792) is below the recommended sizes (up to
  2752x1536). Give this VAE its own limit, `IRIS_QIMAGE_MAX_DIM`, sized from
  measured decode memory.

## M6. Routing and plumbing

- `iris.c`: `iris_is_qimage()`, `iris_generate_qimage()` (encode, release the
  encoder, load the transformer, prefill, sample, decode),
  `iris_load_qimage_transformer_if_needed()`. Add `qimage` cases at every
  `is_zimage` / `is_flux3` site: iris.c 460, 473, 577, 663, 694-720, 1149,
  1283, 1382, 1551, 1761, 1995, 2010. `iris_img2img` and `iris_multiref` must
  refuse with a clear error until Phase 2.
- `iris_encode_text`: select `QWEN3_EXTRACT_QIMAGE` and return the real token
  count, as for Z-Image.
- Sizes: floor to multiples of 32. The pipeline floors to `vae_scale_factor * 2`
  and the Flux paths floor to 16.
- Noise: `iris_init_noise` into `[64, H/16, W/16]`, packed to `[N, 64]` raster
  order (`_pack_latents`).
- Step previews: `iris_step_image_vae` assumes the Flux VAE type. Route it
  through the qimage decoder, or disable previews for qimage at first and
  say so.
- `iris_cli.c` (REPL) and `main.c`: help text defaults (40 / 8 steps, no
  guidance), the scheduler name (iris_cli.c:85), and the `is_zimage` branches
  at iris_cli.c:365-1134. Base and Turbo both use the distilled-style path
  (no CFG). Do not reuse the FLUX.3 base CFG route.
- `iris.h`: declare `iris_is_qimage`.
- Makefile: add `iris_transformer_qimage.c` and `iris_vae_qimage.c` to `SRCS`,
  plus dependency lines like iris_transformer_flux3.o (Makefile:228).
- `download_model.py` / `download_model.sh`: `qimage` and `qimage-turbo`
  entries, with the licence and disk size noted.
- `run_test.py`: `is_qimage_model_dir`, `--qimage-model-dir`, and a 256x256
  Turbo test. Generate the vector from **iris's** noise for the seed via
  `--noise`, as the FLUX.3 vectors were.
- README and CLAUDE.md: variants, file structure, architecture, pitfalls
  (including the new `iris_dev_copy_2d`, `iris_dev_pixel_rms_norm` and
  `iris_dev_dup_up` in the device-backend section).

## Acceptance

- `tokens.txt`, `drop_idx.txt` exact. `emb.bin`, `sigmas.txt`, step velocities
  and `image.bin` match the reference, with the tolerances used stated in the
  final report.
- `./iris -d ~/GenAI/models/Qwen-Image-2.1-Turbo -p "a fish" -W 1024 -H 1024 -o
  /tmp/q.png` gives a sensible image. Do the same for base.
- Device and CPU paths agree on one MPS and one BLAS build.
- FLUX.3 tests still match the recorded baseline.
- CUDA/ROCm builds compile if a toolchain is available. If not, say they were
  not built.

## Risks

| Risk | Mitigation |
|---|---|
| Turbo/base scheduler configs differ from the assumptions above | Generic config-driven scheduler; `sigmas.txt` check in M1 |
| Tensor names in the VAE/transformer headers differ from the module tree | Dump the safetensors headers in M0 before writing loaders |
| Cache-on vs cache-off rounding confuses parity | Reference always runs `use_kv_cache=True` in float32 |
| VAE memory at 2K+ | Separate size cap; tiling stays out of scope |
| Prompts over 17280 tokens | Truncate the body, keep the template tail, warn |
| Metal `iris_dev_copy_2d` forcing syncs | Blit in the current command buffer, never a host memcpy |

## Phase 2: reference images and editing (separate task)

Audited against diffusers 0.42 dev and transformers 5.17 in the venv. Everything
below comes from reading the source; only AvgDown, the RoPE and tokenization were
executed.

### Facts

- **Vision tower** (27 blocks, 1152, 16 heads, head_dim 72, patch 16, merge 2):
  windowless, one image at a time. Patch embed is a Conv3d on two identical frames,
  so a linear `[1152, 768]` with summed weights. Position embedding 48x48,
  bilinear with align_corners. Patches are block-major. RoPE theta 10000, split-half
  pairs (i, i + 36), Q and K only, no head norm. Blocks are pre-norm, LayerNorm
  with affine, GELU-tanh MLP. Mergers use exact GELU; deepstack mergers read blocks
  8, 16, 24 and norm after flattening 2x2.
- **Deepstack**: outputs are added at the image rows right after text layers 0, 1, 2.
- **Vision input**: RGB, `2v/255 - 1`, `<|image_pad|>` count `(H/32)*(W/32)`, 1024
  at 1024x1024.
- **Text**: `<image1>` is literal text; pieces between special tokens tokenize
  separately; the prompt is tokenized on its own. Drop index stays 14. MRoPE is one
  function, `qwen3_set_rope_positions`: text `(p, p, p)`, image `(p, p + r, p + c)`
  in merged units, then `p += max(gh, gw) / 2`. Frequencies with `i < 60`: `i%3==1`
  height, `i%3==2` width, others frame.
- **Transformer**: `[8 text][B1][text]...[BN][vision_end, prompt, tail][target]`.
  Each slot is 4 rows, filled with latents in plain raster order. Position advance
  after a block is `max(h, w)` in latent units (unlike the text encoder). Target
  frame = text rows + sum of `max(hi, wi)`. Text and references use t = 0. `mu` and
  noise depend on the target only. Prefill runs 2N+1 segments: Q `[start, end)`, K/V
  `[0, end)`; text segments are causal.
- **Attention offset**: treat `causal` as an integer, 0 = none, n > 0 = query i sees
  keys `j < i + n`. Existing callers pass 1, the tail run passes `start + 1`. One-line
  changes in iris_cuda.cu (`qi + causal`), iris_dev_metal.m (`j = i + causal`) and
  the CPU attention.
- **VAE encoder**: dims `[96,96,192,384,768,768]`, down conv pads right and bottom
  by 1, then 3x3 stride 2 (`iris_dev_conv2d(k=3, stride=2, pad_before=0, pad_after=1)`).
  `out[o, y, x] = (1/G) sum_g X[oG+g, y, x]`, `G = cin*ft*fs^2/cout`,
  `X[((c*ft+t)*2+i)*2+j] = in[c, 2y+i, 2x+j]` if `t == ft-1`, else 0. Then `norm_out`,
  SiLU, `conv_out`, `quant_conv`, first 64 channels, `(z - mean) / std`.
- **Reference prep**: RGBA, `round(sqrt(1024^2 * ratio) / 32) * 32` (banker's),
  Lanczos with premultiplied alpha, skipped at equal size. The size ignores `-W -H`.
  The default target size follows the **last** image. The vision copy is the resized
  RGBA over white.

### Gaps the plan must cover

- **1280 token limit** (done, M8): sequences past the Metal threadgroup arrays go
  through `qwen3_forward_dev`, `attn_scores` holds one head, work buffers follow
  the real sequence. The limit is now 17280 tokens, 16 references and a prompt.
- `iris_dev_head_norm_rope` cannot serve the vision tower: it normalizes and needs
  a power-of-two thread count.
- `iris_image_convert` mishandles 2-channel input, `iris_image_resize` has no
  premultiply step.
- The REPL's embedding cache (`emb_cache_lookup_ex` in iris_cli.c) is keyed on the
  prompt, so it must not serve image-conditioned prompts.

### Milestones

Each reads Python dumps, so stages are tested alone. Ship one reference first.

| # | Milestone | Done when |
|---|---|---|
| M0 | `qimage_reference.py --image`, per-stage dumps | dumps exist for 1 and 2 references |
| M1 | Reference prep | exact at equal size, RMS <= 2/255 when resized |
| M2 | VAE encoder, CPU then GPU (`iris_dev_avg_down_add`) | per-stage match, then bf16 tolerance |
| M3 | Transformer interleaving, integer `causal` | block states and v match, 1 and 2 refs from dumps |
| M4 | Vision tower, CPU (`iris_qwen3_vision.c`) | blocks, deepstack, embeds match |
| M5 | Vision tower, GPU (`iris_dev_rope`, `iris_dev_gelu_tanh`) | bf16 tolerance |
| M6 | Text encoder with images, up to 1280 tokens | tokens, positions, layers, `emb.bin` non-slot rows |
| M7 | End to end, one reference | final latent and image match, Turbo and base |
| M8 | Long sequences, 2+ references, done | 2-ref `emb.bin` match, 10-ref memory run |

GPU memory at 1 MP, peak on Metal: about 24 GB for one
reference, 51 GB for ten (prefix K/V 2.2 and 21.5 GB). Skipped: CFG with
references, VAE tiling.

## Licence

Qwen Research License: non-commercial only. Commercial use needs a separate
licence from Qwen. "Qwen" may be used descriptively but not as the primary name
of a derivative product.

## Sources

- Model cards: https://huggingface.co/Qwen/Qwen-Image-2.1 and
  https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo
- Project: https://github.com/QwenLM/Qwen-Image-2.1
- Diffusers docs: https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21
- Diffusers source (also installed in the venv above):
  - `src/diffusers/models/transformers/transformer_qwenimage21.py`
  - `src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py`
  - `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`
  - `src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`
