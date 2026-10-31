/*
 * Qwen-Image 2.1 (base and Turbo): internal API shared by iris.c and the
 * Qwen-Image components (scheduler, transformer, VAE decoder).
 */

#ifndef IRIS_QIMAGE_H
#define IRIS_QIMAGE_H

#include "iris.h"

/* Upper bound on Turbo's sample_sigmas in model_index.json. */
#define QIMAGE_MAX_SIGMAS 64

/* Latent channels (VAE z_dim, transformer in/out channels). */
#define IRIS_QIMAGE_LATENT_CHANNELS 64

/* Image size limits. The model recommends up to 2752x1536; the VAE decoder
 * needs several GB of activations at 4 MP, so the pixel count is capped
 * there. */
#define IRIS_QIMAGE_MAX_DIM    2816
#define IRIS_QIMAGE_MAX_PIXELS (2048 * 2048)

/* ========================================================================
 * Scheduler (iris_sample.c)
 * ======================================================================== */

/* FlowMatchEulerDiscreteScheduler settings from scheduler_config.json,
 * with the diffusers defaults for absent keys. */
typedef struct {
    int use_dynamic_shifting;
    float base_seq_len, max_seq_len;    /* base/max_image_seq_len */
    float base_shift, max_shift;
    float shift;                        /* static shift (no dynamic shifting) */
    float shift_terminal;               /* 0 = no stretch */
    int linear_time_shift;              /* time_shift_type "linear" */
} iris_qimage_sched_t;

void iris_qimage_sched_load(iris_qimage_sched_t *s, const char *model_dir);

/* Schedule of num_steps + 1 sigmas (terminal 0), the way
 * FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=...) processes
 * them. sigmas is Turbo's sample_sigmas grid (num_steps values), or NULL
 * for linspace(1, 1/num_steps, num_steps). image_seq_len is the target
 * token count (H/16 * W/16). */
float *iris_schedule_qimage(const iris_qimage_sched_t *s, const float *sigmas,
                            int num_steps, int image_seq_len);

/* ========================================================================
 * Transformer (iris_transformer_qimage.c)
 * ======================================================================== */

typedef struct iris_transformer_qimage iris_transformer_qimage_t;

iris_transformer_qimage_t *iris_transformer_load_qimage(const char *model_dir);
void iris_transformer_free_qimage(iris_transformer_qimage_t *tf);

/* Latent channels the transformer reads and predicts (64). */
int iris_transformer_channels_qimage(const iris_transformer_qimage_t *tf);

/* Run the text tokens txt [txt_seq, context_in_dim] through every block
 * once and keep their keys and values. Text never attends to the image
 * and is modulated from t = 0, so this prefix serves every step (and
 * every generation with the same prompt). Returns 1 on success. */
int iris_transformer_prefill_qimage(iris_transformer_qimage_t *tf,
                                    const float *txt, int txt_seq);

/* Velocity [channels, h * w] of the latent x [channels, h * w] at sigma,
 * attending to the prefix from the last prefill (caller frees). */
float *iris_transformer_forward_qimage(iris_transformer_qimage_t *tf,
                                       const float *x, int h, int w, float sigma);

/* ========================================================================
 * VAE decoder (iris_vae_qimage.c)
 * ======================================================================== */

typedef struct iris_vae_qimage iris_vae_qimage_t;

iris_vae_qimage_t *iris_vae_load_qimage(const char *model_dir);
void iris_vae_free_qimage(iris_vae_qimage_t *vae);

/* Decode a normalized latent [z_dim, h, w] (as the transformer produces
 * it) into an RGBA image of 16h x 16w pixels. */
iris_image *iris_vae_decode_qimage(iris_vae_qimage_t *vae, const float *latent,
                                   int h, int w);

/* ========================================================================
 * Sampler (iris_sample.c)
 * ======================================================================== */

/* Euler sampler: z += (sigma_next - sigma) * v. The text prefix must have
 * been prefilled. preview_vae, if not NULL, decodes step previews for the
 * step image callback. Returns the final latent (caller frees). */
float *iris_sample_euler_qimage(iris_transformer_qimage_t *tf,
                                const float *z, int channels, int h, int w,
                                const float *schedule, int num_steps,
                                iris_vae_qimage_t *preview_vae);

#endif /* IRIS_QIMAGE_H */
