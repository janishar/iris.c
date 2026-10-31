/*
 * Qwen-Image 2.1 VAE Decoder
 *
 * Decoder of AutoencoderKLQwenImage21, a Wan-derived VAE specialized to a
 * single frame: its "causal 3D" convolutions are plain 2D convolutions
 * (symmetric padding), and with one frame the temporal convolutions never
 * run, so the whole decoder is 2D:
 *
 *   x = post_quant_conv(latent * std + mean)        1x1, z_dim -> z_dim
 *   x = conv_in(x)                                  3x3, z_dim -> top width
 *   x = mid: res, attention, res
 *   x = up blocks (residual): res blocks, nearest 2x + 3x3 conv, plus a
 *       DupUp shortcut of the block input
 *   x = conv_out(silu(norm_out(x)))                 -> RGBA
 *   clamp to [-1, 1]
 *
 * Norms are per-pixel RMS norms over the channels: x / ||x|| * sqrt(C) *
 * gamma (F.normalize, no epsilon inside the root). Res block: shortcut
 * (1x1 when the channel count changes) + conv2(silu(norm2(conv1(silu(
 * norm1(x)))))). The mid attention is single-head over all latent pixels.
 *
 * Activations are f32 NCHW [C, H, W], channels-last on the GPU. Weights are
 * converted to f32 at load (the checkpoint may be f32 or bf16).
 */

#include "iris.h"
#include "iris_kernels.h"
#include "iris_safetensors.h"
#include "iris_qimage.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifdef IRIS_DEV
#include "iris_dev.h"
#endif

#define QV_MAX_CH      512   /* latent channels / config array bound */
#define QV_MAX_LEVELS  8     /* entries of dim_mult */
#define QV_MAX_RES     4     /* res blocks per up block */
#define QV_ATTN_QCHUNK 256   /* queries per attention chunk */

/* ========================================================================
 * Data Structures
 * ======================================================================== */

typedef struct {
    float *w, *b;            /* [cout, cin, k, k], [cout] */
    int cin, cout, k;
} qv_conv_t;

typedef struct {
    int cin, cout;
    float *gamma1, *gamma2;  /* [cin], [cout] */
    qv_conv_t conv1, conv2;
    qv_conv_t shortcut;      /* 1x1, only when cin != cout (w == NULL otherwise) */
} qv_res_t;

typedef struct {
    int cin, cout;
    int num_res;
    qv_res_t res[QV_MAX_RES];
    int upsample;            /* 2x upsampling with a DupUp shortcut */
    int factor_t;            /* temporal factor of the DupUp shortcut (1 or 2) */
    qv_conv_t up_conv;       /* conv after the nearest 2x upsampling */
} qv_up_t;

struct iris_vae_qimage {
    int z_dim;
    int out_channels;        /* 4 (RGBA) */
    float mean[QV_MAX_CH], std[QV_MAX_CH];

    qv_conv_t post_quant, conv_in, conv_out;
    qv_res_t mid_res[2];
    float *attn_gamma;       /* mid attention norm [C] */
    qv_conv_t attn_qkv;      /* 1x1, C -> 3C */
    qv_conv_t attn_proj;     /* 1x1, C -> C */
    int num_up;
    qv_up_t up[QV_MAX_LEVELS];
    float *norm_out;         /* [C] */
#ifdef IRIS_DEV
    struct qv_dev *dev;      /* device copy, uploaded at the first decode */
#endif
};

/* ========================================================================
 * Layers
 * ======================================================================== */

/* Convolution with zero padding k/2 and stride 1 (caller frees). */
static float *conv(const qv_conv_t *c, const float *x, int h, int w) {
    float *out = malloc((size_t)c->cout * h * w * sizeof(float));
    if (out) iris_conv2d(out, x, c->w, c->b, 1, c->cin, c->cout, h, w, c->k, c->k, 1, c->k / 2);
    return out;
}

/* In place: x = x / max(||x||, 1e-12) * sqrt(C) * gamma per pixel, then
 * SiLU if silu is set. */
static void pixel_rms_norm(float *x, const float *gamma, int c, int hw, int silu) {
    float *inv = calloc(hw, sizeof(float));
    for (int ch = 0; ch < c; ch++) {
        const float *xc = x + (size_t)ch * hw;
        for (int p = 0; p < hw; p++) inv[p] += xc[p] * xc[p];
    }
    float root_c = sqrtf((float)c);
    for (int p = 0; p < hw; p++) {
        float norm = sqrtf(inv[p]);
        inv[p] = root_c / (norm > 1e-12f ? norm : 1e-12f);
    }
    for (int ch = 0; ch < c; ch++) {
        float *xc = x + (size_t)ch * hw;
        float g = gamma[ch];
        for (int p = 0; p < hw; p++) {
            float v = xc[p] * inv[p] * g;
            xc[p] = silu ? v / (1.0f + expf(-v)) : v;
        }
    }
    free(inv);
}

/* Residual block on x [cin, h, w]. Returns the output [cout, h, w] and
 * frees x when consume is set (x is then used as scratch). */
static float *res_forward(const qv_res_t *r, float *x, int h, int w, int consume) {
    size_t hw = (size_t)h * w;
    float *skip = r->shortcut.w ? conv(&r->shortcut, x, h, w) : NULL;
    float *t = x;
    if (!consume) {
        t = malloc((size_t)r->cin * hw * sizeof(float));
        memcpy(t, x, (size_t)r->cin * hw * sizeof(float));
    } else if (!skip) {
        skip = malloc((size_t)r->cout * hw * sizeof(float));
        memcpy(skip, x, (size_t)r->cout * hw * sizeof(float));
    }
    pixel_rms_norm(t, r->gamma1, r->cin, (int)hw, 1);
    float *h1 = conv(&r->conv1, t, h, w);
    free(t);
    pixel_rms_norm(h1, r->gamma2, r->cout, (int)hw, 1);
    float *out = conv(&r->conv2, h1, h, w);
    free(h1);
    const float *s = skip ? skip : x;   /* identity shortcut reads x */
    for (size_t i = 0; i < (size_t)r->cout * hw; i++) out[i] += s[i];
    free(skip);
    return out;
}

/* Single-head self-attention over the pixels of x [c, h, w], in place:
 * x += proj(softmax(q^T k / sqrt(c)) v), with q, k, v from a 1x1 conv of
 * the normalized x. */
static void mid_attention(const iris_vae_qimage_t *vae, float *x, int c, int h, int w) {
    int n = h * w;
    size_t cn = (size_t)c * n;
    float *t = malloc(cn * sizeof(float));
    memcpy(t, x, cn * sizeof(float));
    pixel_rms_norm(t, vae->attn_gamma, c, n, 0);
    float *qkv = conv(&vae->attn_qkv, t, h, w);   /* [3c, n]: q, k, v */
    const float *q = qkv, *k = qkv + cn, *v = qkv + 2 * cn;
    float *scores = malloc((size_t)QV_ATTN_QCHUNK * n * sizeof(float));
    float scale = 1.0f / sqrtf((float)c);

    for (int q0 = 0; q0 < n; q0 += QV_ATTN_QCHUNK) {
        int nq = n - q0 < QV_ATTN_QCHUNK ? n - q0 : QV_ATTN_QCHUNK;
        /* scores[i, j] = scale * sum_c q[c, q0 + i] * k[c, j] */
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nq, n, c,
                    scale, q + q0, n, k, n, 0.0f, scores, n);
#else
        memset(scores, 0, (size_t)nq * n * sizeof(float));
        for (int ch = 0; ch < c; ch++) {
            const float *qc = q + (size_t)ch * n + q0, *kc = k + (size_t)ch * n;
            for (int i = 0; i < nq; i++) {
                float qi = qc[i] * scale;
                float *si = scores + (size_t)i * n;
                for (int j = 0; j < n; j++) si[j] += qi * kc[j];
            }
        }
#endif
        iris_softmax(scores, nq, n);
        /* t[c, q0 + i] = sum_j v[c, j] * scores[i, j] */
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, c, nq, n,
                    1.0f, v, n, scores, n, 0.0f, t + q0, n);
#else
        for (int ch = 0; ch < c; ch++) {
            const float *vc = v + (size_t)ch * n;
            for (int i = 0; i < nq; i++) {
                const float *si = scores + (size_t)i * n;
                float acc = 0;
                for (int j = 0; j < n; j++) acc += vc[j] * si[j];
                t[(size_t)ch * n + q0 + i] = acc;
            }
        }
#endif
    }
    free(scores);
    free(qkv);
    float *o = conv(&vae->attn_proj, t, h, w);
    for (size_t i = 0; i < cn; i++) x[i] += o[i];
    free(o);
    free(t);
}

/* out [cout, 2h, 2w] += DupUp(in [cin, h, w]) with first_chunk set
 * (QwenImage21DupUp3D): channels repeated, folded into
 * (cout, factor_t, 2, 2), pixel-shuffled, temporal slot factor_t - 1 kept:
 *   out[o, 2y+i, 2x+j] += in[(((o*ft + ft-1)*2 + i)*2 + j) / repeats, y, x] */
static void dup_up_add(float *out, const float *in, int cin, int cout, int ft, int h, int w) {
    int repeats = cout * ft * 4 / cin;
    int ow = 2 * w;
    for (int o = 0; o < cout; o++) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                int src = (((o * ft + ft - 1) * 2 + i) * 2 + j) / repeats;
                const float *s = in + (size_t)src * h * w;
                float *d = out + (size_t)o * 4 * h * w;
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        d[(size_t)(2 * y + i) * ow + 2 * x + j] += s[(size_t)y * w + x];
            }
        }
    }
}

/* Up block on x [cin, h, w] (consumed). Returns [cout, h', w'] and updates
 * h, w when the block upsamples. */
static float *up_forward(const qv_up_t *u, float *x, int *h, int *w) {
    float *cur = x;
    for (int r = 0; r < u->num_res; r++)
        cur = res_forward(&u->res[r], cur, *h, *w, r > 0 || !u->upsample);
    if (!u->upsample) return cur;

    float *big = malloc((size_t)u->cout * 4 * *h * *w * sizeof(float));
    iris_upsample_nearest(big, cur, 1, u->cout, *h, *w, 2, 2);
    free(cur);
    float *out = conv(&u->up_conv, big, 2 * *h, 2 * *w);
    free(big);
    dup_up_add(out, x, u->cin, u->cout, u->factor_t, *h, *w);
    free(x);
    *h *= 2;
    *w *= 2;
    return out;
}

/* [channels, h, w] floats in [-1, 1] to an image: clamp, then round(v * 255)
 * like diffusers, dropping the alpha channel of a fully opaque image. */
static iris_image *rgba_to_image(const float *rgba, int channels, int h, int w) {
    size_t npix = (size_t)h * w;
    iris_image *img = iris_image_create(w, h, channels);
    if (!img) return NULL;
    int opaque = channels == 4;
    for (size_t p = 0; p < npix; p++) {
        for (int ch = 0; ch < channels; ch++) {
            float v = rgba[ch * npix + p];
            v = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
            img->data[p * channels + ch] = (uint8_t)((v + 1.0f) * 127.5f + 0.5f);
        }
        if (opaque && img->data[p * channels + 3] != 255) opaque = 0;
    }
    if (opaque) {
        for (size_t p = 0; p < npix; p++)
            memmove(img->data + p * 3, img->data + p * 4, 3);
        img->channels = 3;
    }
    return img;
}

#ifdef IRIS_DEV
/* ========================================================================
 * GPU Path (CUDA / ROCm / Metal)
 * ======================================================================== */

typedef struct {
    uint16_t *w;             /* bf16 [cout, k * k * cin] */
    float *b;                /* [cout] */
} qd_conv_t;

typedef struct {
    float *gamma1, *gamma2;
    qd_conv_t conv1, conv2, shortcut;
} qd_res_t;

struct qv_dev {
    qd_conv_t post_quant, conv_in, conv_out, attn_qkv, attn_proj;
    qd_res_t mid_res[2];
    qd_res_t res[QV_MAX_LEVELS][QV_MAX_RES];
    qd_conv_t up_conv[QV_MAX_LEVELS];
    float *attn_gamma, *norm_out;

    /* Activations, sized by dv_ensure() */
    float *buf[4];           /* x, y, t1, t2 */
    float *skip;             /* input of an upsampling block */
    uint16_t *attn;          /* q, k, v, out */
    size_t buf_elems, skip_elems, attn_elems;
};

static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

static float *dv_f32(const float *src, size_t n) {
    float *d = iris_dev_alloc(n * sizeof(float));
    if (d) iris_dev_upload(d, src, n * sizeof(float));
    return d;
}

/* Upload a convolution: weight [cout, cin, k, k] to the channels-last
 * layout of iris_dev_conv2d. A missing convolution stays NULL. */
static int dv_conv(qd_conv_t *d, const qv_conv_t *c) {
    if (!c->w) return 1;
    int kk = c->k * c->k;
    size_t row = (size_t)kk * c->cin, n = row * c->cout;
    uint16_t *tmp = malloc(n * sizeof(uint16_t));
    if (!tmp) return 0;
    for (int o = 0; o < c->cout; o++)
        for (int ci = 0; ci < c->cin; ci++)
            for (int t = 0; t < kk; t++)
                tmp[o * row + (size_t)t * c->cin + ci] =
                    f32_to_bf16(c->w[((size_t)o * c->cin + ci) * kk + t]);
    d->w = iris_dev_alloc(n * sizeof(uint16_t));
    if (d->w) iris_dev_upload(d->w, tmp, n * sizeof(uint16_t));
    free(tmp);
    d->b = dv_f32(c->b, c->cout);
    return d->w && d->b;
}

static int dv_res(qd_res_t *d, const qv_res_t *r) {
    d->gamma1 = dv_f32(r->gamma1, r->cin);
    d->gamma2 = dv_f32(r->gamma2, r->cout);
    return d->gamma1 && d->gamma2 && dv_conv(&d->conv1, &r->conv1) &&
           dv_conv(&d->conv2, &r->conv2) && dv_conv(&d->shortcut, &r->shortcut);
}

static void dv_free_conv(qd_conv_t *d) {
    iris_dev_free(d->w);
    iris_dev_free(d->b);
}

static void dv_free_res(qd_res_t *d) {
    iris_dev_free(d->gamma1);
    iris_dev_free(d->gamma2);
    dv_free_conv(&d->conv1);
    dv_free_conv(&d->conv2);
    dv_free_conv(&d->shortcut);
}

static void dv_free_buffers(struct qv_dev *d) {
    for (int i = 0; i < 4; i++) {
        iris_dev_free(d->buf[i]);
        d->buf[i] = NULL;
    }
    iris_dev_free(d->skip);
    iris_dev_free(d->attn);
    d->skip = NULL;
    d->attn = NULL;
    d->buf_elems = d->skip_elems = d->attn_elems = 0;
}

static void dv_free(iris_vae_qimage_t *vae) {
    struct qv_dev *d = vae->dev;
    if (!d) return;
    dv_free_conv(&d->post_quant);
    dv_free_conv(&d->conv_in);
    dv_free_conv(&d->conv_out);
    dv_free_conv(&d->attn_qkv);
    dv_free_conv(&d->attn_proj);
    dv_free_res(&d->mid_res[0]);
    dv_free_res(&d->mid_res[1]);
    for (int i = 0; i < vae->num_up; i++) {
        for (int r = 0; r < vae->up[i].num_res; r++) dv_free_res(&d->res[i][r]);
        dv_free_conv(&d->up_conv[i]);
    }
    iris_dev_free(d->attn_gamma);
    iris_dev_free(d->norm_out);
    dv_free_buffers(d);
    free(d);
    vae->dev = NULL;
}

static int dv_load(iris_vae_qimage_t *vae) {
    struct qv_dev *d = calloc(1, sizeof(*d));
    if (!d) return 0;
    vae->dev = d;
    int top = vae->mid_res[0].cout, last = vae->up[vae->num_up - 1].cout;
    int ok = dv_conv(&d->post_quant, &vae->post_quant) && dv_conv(&d->conv_in, &vae->conv_in) &&
             dv_conv(&d->conv_out, &vae->conv_out) && dv_conv(&d->attn_qkv, &vae->attn_qkv) &&
             dv_conv(&d->attn_proj, &vae->attn_proj) && dv_res(&d->mid_res[0], &vae->mid_res[0]) &&
             dv_res(&d->mid_res[1], &vae->mid_res[1]);
    for (int i = 0; ok && i < vae->num_up; i++) {
        for (int r = 0; ok && r < vae->up[i].num_res; r++)
            ok = dv_res(&d->res[i][r], &vae->up[i].res[r]);
        ok = ok && dv_conv(&d->up_conv[i], &vae->up[i].up_conv);
    }
    d->attn_gamma = ok ? dv_f32(vae->attn_gamma, top) : NULL;
    d->norm_out = ok ? dv_f32(vae->norm_out, last) : NULL;
    if (!d->attn_gamma || !d->norm_out) {
        dv_free(vae);
        return 0;
    }
    iris_dev_sync();
    return 1;
}

/* Size the activations for a latent of h x w pixels. */
static int dv_ensure(iris_vae_qimage_t *vae, int h, int w) {
    struct qv_dev *d = vae->dev;
    size_t hw = (size_t)h * w, top = vae->mid_res[0].cout;
    size_t need = hw * 3 * top, skip = 0, attn = hw * 4 * top;
    for (int i = 0; i < vae->num_up; i++) {
        const qv_up_t *u = &vae->up[i];
        size_t c = u->cin > u->cout ? u->cin : u->cout;
        if (hw * c > need) need = hw * c;
        if (u->upsample) {
            if (hw * u->cin > skip) skip = hw * u->cin;
            hw *= 4;
            if (hw * u->cout > need) need = hw * u->cout;
        }
    }
    if (need <= d->buf_elems && skip <= d->skip_elems && attn <= d->attn_elems) return 1;
    dv_free_buffers(d);
    int ok = 1;
    for (int i = 0; i < 4; i++) {
        d->buf[i] = iris_dev_alloc(need * sizeof(float));
        ok = ok && d->buf[i];
    }
    d->skip = iris_dev_alloc(skip * sizeof(float));
    d->attn = iris_dev_alloc(attn * sizeof(uint16_t));
    if (!ok || !d->skip || !d->attn) {
        dv_free_buffers(d);
        return 0;
    }
    d->buf_elems = need;
    d->skip_elems = skip;
    d->attn_elems = attn;
    return 1;
}

static void dv_conv2d(float *out, const float *in, const qd_conv_t *d, const qv_conv_t *c,
                      int h, int w) {
    iris_dev_conv2d(out, in, d->w, d->b, h, w, c->cin, c->cout, c->k, 1, c->k / 2, c->k / 2);
}

/* out = shortcut(x) + conv2(silu(norm2(conv1(silu(norm1(x)))))) */
static void dv_res_forward(const qd_res_t *d, const qv_res_t *r, float *out, const float *x,
                           float *t1, float *t2, int h, int w) {
    int hw = h * w;
    iris_dev_pixel_rms_norm(t1, x, d->gamma1, hw, r->cin, 1);
    dv_conv2d(t2, t1, &d->conv1, &r->conv1, h, w);
    iris_dev_pixel_rms_norm(t2, t2, d->gamma2, hw, r->cout, 1);
    dv_conv2d(out, t2, &d->conv2, &r->conv2, h, w);
    if (d->shortcut.w) {
        dv_conv2d(t1, x, &d->shortcut, &r->shortcut, h, w);
        iris_dev_add(out, t1, (size_t)hw * r->cout);
    } else {
        iris_dev_add(out, x, (size_t)hw * r->cout);
    }
}

/* Single-head self-attention over the n pixels, plus the residual. */
static void dv_attention(iris_vae_qimage_t *vae, float *out, const float *x, float *t1,
                         float *t2, int n, int c) {
    struct qv_dev *d = vae->dev;
    uint16_t *qkv = d->attn, *o = qkv + (size_t)n * 3 * c;
    iris_dev_pixel_rms_norm(t1, x, d->attn_gamma, n, c, 0);
    dv_conv2d(t2, t1, &d->attn_qkv, &vae->attn_qkv, n, 1);
    iris_dev_f32_to_bf16(qkv, t2, (size_t)n * 3 * c);
    iris_dev_attention(o, c, qkv, qkv + c, qkv + 2 * c, 3 * c, n, n, 1, 1, c,
                       1.0f / sqrtf((float)c), 0, NULL);
    iris_dev_linear_f32(out, o, d->attn_proj.w, n, c, c);
    iris_dev_add_bias(out, d->attn_proj.b, n, c);
    iris_dev_add(out, x, (size_t)n * c);
}

int iris_vae_reserve_qimage(iris_vae_qimage_t *vae, int h, int w) {
    if (!iris_dev_available()) return 1;
    if (!vae->dev && !dv_load(vae)) return 0;
    return dv_ensure(vae, h, w);
}

void iris_vae_release_qimage(iris_vae_qimage_t *vae) {
    if (vae->dev) dv_free_buffers(vae->dev);
}

static iris_image *decode_dev(iris_vae_qimage_t *vae, const float *latent, int h, int w) {
    if (!iris_vae_reserve_qimage(vae, h, w)) return NULL;
    struct qv_dev *d = vae->dev;
    float *x = d->buf[0], *y = d->buf[1], *t1 = d->buf[2], *t2 = d->buf[3], *tmp;
#define DV_SWAP() (tmp = x, x = y, y = tmp)

    int z = vae->z_dim, top = vae->mid_res[0].cout;
    size_t hw = (size_t)h * w;
    float *host = malloc((size_t)z * hw * sizeof(float));
    if (!host) return NULL;
    for (int c = 0; c < z; c++)
        for (size_t p = 0; p < hw; p++)
            host[p * z + c] = latent[c * hw + p] * vae->std[c] + vae->mean[c];
    iris_dev_upload(x, host, (size_t)z * hw * sizeof(float));
    free(host);

    dv_conv2d(y, x, &d->post_quant, &vae->post_quant, h, w);
    dv_conv2d(x, y, &d->conv_in, &vae->conv_in, h, w);
    dv_res_forward(&d->mid_res[0], &vae->mid_res[0], y, x, t1, t2, h, w);
    dv_attention(vae, x, y, t1, t2, (int)hw, top);
    dv_res_forward(&d->mid_res[1], &vae->mid_res[1], y, x, t1, t2, h, w);
    DV_SWAP();

    for (int i = 0; i < vae->num_up; i++) {
        const qv_up_t *u = &vae->up[i];
        /* A byte copy of the block input for the DupUp shortcut. */
        if (u->upsample)
            iris_dev_copy_2d((uint16_t *)d->skip, 2 * u->cin, (const uint16_t *)x, 2 * u->cin,
                             h * w, 2 * u->cin);
        for (int r = 0; r < u->num_res; r++) {
            dv_res_forward(&d->res[i][r], &u->res[r], y, x, t1, t2, h, w);
            DV_SWAP();
        }
        if (u->upsample) {
            iris_dev_upsample2x(y, x, h, w, u->cout);
            dv_conv2d(x, y, &d->up_conv[i], &u->up_conv, 2 * h, 2 * w);
            iris_dev_dup_up_add(x, d->skip, h, w, u->cin, u->cout, u->factor_t);
            h *= 2;
            w *= 2;
        }
        if (iris_vae_progress_callback) iris_vae_progress_callback(i + 1, vae->num_up);
    }
#undef DV_SWAP

    int c = vae->up[vae->num_up - 1].cout, channels = vae->out_channels;
    iris_dev_pixel_rms_norm(y, x, d->norm_out, h * w, c, 1);
    dv_conv2d(t1, y, &d->conv_out, &vae->conv_out, h, w);

    hw = (size_t)h * w;
    float *hwc = malloc((size_t)channels * hw * sizeof(float));
    float *rgba = malloc((size_t)channels * hw * sizeof(float));
    iris_image *img = NULL;
    if (hwc && rgba) {
        iris_dev_download(hwc, t1, (size_t)channels * hw * sizeof(float));
        for (int ch = 0; ch < channels; ch++)
            for (size_t p = 0; p < hw; p++) rgba[ch * hw + p] = hwc[p * channels + ch];
        img = rgba_to_image(rgba, channels, h, w);
    }
    free(hwc);
    free(rgba);
    return img;
}
#else
int iris_vae_reserve_qimage(iris_vae_qimage_t *vae, int h, int w) {
    (void)vae, (void)h, (void)w;
    return 1;
}

void iris_vae_release_qimage(iris_vae_qimage_t *vae) {
    (void)vae;
}
#endif /* IRIS_DEV */

/* ========================================================================
 * Decode
 * ======================================================================== */

iris_image *iris_vae_decode_qimage(iris_vae_qimage_t *vae, const float *latent,
                                   int h, int w) {
#ifdef IRIS_DEV
    if (iris_dev_available()) {
        iris_image *img = decode_dev(vae, latent, h, w);
        if (img) return img;
        fprintf(stderr, "Qwen-Image VAE: GPU decode failed, using the CPU\n");
    }
#endif
    size_t hw = (size_t)h * w;
    float *z = malloc((size_t)vae->z_dim * hw * sizeof(float));
    for (int c = 0; c < vae->z_dim; c++)
        for (size_t p = 0; p < hw; p++)
            z[c * hw + p] = latent[c * hw + p] * vae->std[c] + vae->mean[c];

    float *x = conv(&vae->post_quant, z, h, w);
    free(z);
    z = conv(&vae->conv_in, x, h, w);
    free(x);
    x = res_forward(&vae->mid_res[0], z, h, w, 1);
    mid_attention(vae, x, vae->mid_res[0].cout, h, w);
    x = res_forward(&vae->mid_res[1], x, h, w, 1);

    int oh = h, ow = w;
    for (int i = 0; i < vae->num_up; i++) {
        x = up_forward(&vae->up[i], x, &oh, &ow);
        if (iris_vae_progress_callback) iris_vae_progress_callback(i + 1, vae->num_up);
    }

    int c = vae->up[vae->num_up - 1].cout;
    pixel_rms_norm(x, vae->norm_out, c, oh * ow, 1);
    float *rgba = conv(&vae->conv_out, x, oh, ow);
    free(x);

    iris_image *img = rgba_to_image(rgba, vae->out_channels, oh, ow);
    free(rgba);
    return img;
}

/* ========================================================================
 * Loading
 * ======================================================================== */

/* Read a JSON number array "key": [..] into out (at most max values).
 * Returns the count, 0 when absent. true/false read as 1/0. */
static int json_array(const char *json, const char *key, float *out, int max) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p || !(p = strchr(p + strlen(pat), ':'))) return 0;
    p += 1 + strspn(p + 1, " \t\r\n");
    if (*p != '[') return 0;
    p++;
    int n = 0;
    while (n < max) {
        p += strspn(p, " \t\r\n");
        if (*p == ']' || !*p) break;
        if (!strncmp(p, "true", 4)) { out[n++] = 1; p += 4; }
        else if (!strncmp(p, "false", 5)) { out[n++] = 0; p += 5; }
        else {
            char *end;
            out[n] = strtof(p, &end);
            if (end == p) break;
            n++;
            p = end;
        }
        p += strspn(p, " \t\r\n");
        if (*p == ',') p++;
    }
    return n;
}

static double json_num(const char *json, const char *key, double def) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p || !(p = strchr(p + strlen(pat), ':'))) return def;
    p += 1 + strspn(p + 1, " \t\r\n");
    if (!strncmp(p, "null", 4)) return def;
    if (!strncmp(p, "true", 4)) return 1;
    if (!strncmp(p, "false", 5)) return 0;
    return atof(p);
}

/* Tensor as f32 (caller frees), checking its element count. */
static float *load_f32(safetensors_file_t *sf, const char *name, int64_t numel) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "Qwen-Image VAE: missing tensor %s\n", name);
        return NULL;
    }
    if (safetensor_numel(t) != numel) {
        fprintf(stderr, "Qwen-Image VAE: %s has %lld elements, expected %lld\n", name,
                (long long)safetensor_numel(t), (long long)numel);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static int load_conv(safetensors_file_t *sf, qv_conv_t *c, const char *prefix,
                     int cin, int cout, int k) {
    char name[256];
    c->cin = cin;
    c->cout = cout;
    c->k = k;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    c->w = load_f32(sf, name, (int64_t)cout * cin * k * k);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    c->b = load_f32(sf, name, cout);
    return c->w && c->b;
}

static int load_res(safetensors_file_t *sf, qv_res_t *r, const char *prefix, int cin, int cout) {
    char name[256];
    r->cin = cin;
    r->cout = cout;
    snprintf(name, sizeof(name), "%s.norm1.gamma", prefix);
    r->gamma1 = load_f32(sf, name, cin);
    snprintf(name, sizeof(name), "%s.norm2.gamma", prefix);
    r->gamma2 = load_f32(sf, name, cout);
    snprintf(name, sizeof(name), "%s.conv1", prefix);
    int ok = r->gamma1 && r->gamma2 && load_conv(sf, &r->conv1, name, cin, cout, 3);
    snprintf(name, sizeof(name), "%s.conv2", prefix);
    ok = ok && load_conv(sf, &r->conv2, name, cout, cout, 3);
    if (ok && cin != cout) {
        snprintf(name, sizeof(name), "%s.conv_shortcut", prefix);
        ok = load_conv(sf, &r->shortcut, name, cin, cout, 1);
    }
    return ok;
}

static void free_conv(qv_conv_t *c) {
    free(c->w);
    free(c->b);
}

static void free_res(qv_res_t *r) {
    free(r->gamma1);
    free(r->gamma2);
    free_conv(&r->conv1);
    free_conv(&r->conv2);
    free_conv(&r->shortcut);
}

/* Load the decoder from <model_dir>/vae (config.json and the weights). */
iris_vae_qimage_t *iris_vae_load_qimage(const char *model_dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/vae/config.json", model_dir);
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Qwen-Image VAE: missing %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *cfg = malloc(size + 1);
    size_t got = fread(cfg, 1, size, f);
    cfg[got] = '\0';
    fclose(f);

    iris_vae_qimage_t *vae = calloc(1, sizeof(*vae));
    float dim_mult[QV_MAX_LEVELS], tdown[QV_MAX_LEVELS];
    int levels = json_array(cfg, "dim_mult", dim_mult, QV_MAX_LEVELS);
    int ntdown = json_array(cfg, "temperal_downsample", tdown, QV_MAX_LEVELS);
    int base = (int)json_num(cfg, "decoder_base_dim", json_num(cfg, "base_dim", 96));
    int num_res = (int)json_num(cfg, "num_res_blocks", 2) + 1;
    int residual = (int)json_num(cfg, "is_residual", 0);
    int patch = (int)json_num(cfg, "patch_size", 1);
    vae->z_dim = (int)json_num(cfg, "z_dim", 64);
    vae->out_channels = (int)json_num(cfg, "out_channels", 4);
    int nmean = json_array(cfg, "latents_mean", vae->mean, QV_MAX_CH);
    int nstd = json_array(cfg, "latents_std", vae->std, QV_MAX_CH);
    free(cfg);

    if (levels < 2 || ntdown != levels - 1 || !residual || patch > 1 ||
        num_res > QV_MAX_RES || vae->z_dim > QV_MAX_CH || nmean != vae->z_dim ||
        nstd != vae->z_dim || (vae->out_channels != 3 && vae->out_channels != 4)) {
        fprintf(stderr, "Qwen-Image VAE: unsupported config in %s\n", path);
        free(vae);
        return NULL;
    }

    snprintf(path, sizeof(path), "%s/vae/diffusion_pytorch_model.safetensors", model_dir);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) {
        fprintf(stderr, "Qwen-Image VAE: cannot open %s\n", path);
        free(vae);
        return NULL;
    }

    /* Decoder widths: the top width, then dim_mult reversed. */
    int dims[QV_MAX_LEVELS + 1];
    dims[0] = base * (int)dim_mult[levels - 1];
    for (int i = 0; i < levels; i++) dims[i + 1] = base * (int)dim_mult[levels - 1 - i];
    int top = dims[0], z = vae->z_dim;

    int ok = load_conv(sf, &vae->post_quant, "post_quant_conv", z, z, 1) &&
             load_conv(sf, &vae->conv_in, "decoder.conv_in", z, top, 3) &&
             load_res(sf, &vae->mid_res[0], "decoder.mid_block.resnets.0", top, top) &&
             load_res(sf, &vae->mid_res[1], "decoder.mid_block.resnets.1", top, top) &&
             load_conv(sf, &vae->attn_qkv, "decoder.mid_block.attentions.0.to_qkv",
                       top, 3 * top, 1) &&
             load_conv(sf, &vae->attn_proj, "decoder.mid_block.attentions.0.proj",
                       top, top, 1);
    if (ok) {
        vae->attn_gamma = load_f32(sf, "decoder.mid_block.attentions.0.norm.gamma", top);
        ok = vae->attn_gamma != NULL;
    }

    /* Up blocks; temperal_upsample is temperal_downsample reversed. */
    vae->num_up = levels;
    char name[256];
    for (int i = 0; ok && i < levels; i++) {
        qv_up_t *u = &vae->up[i];
        u->cin = dims[i];
        u->cout = dims[i + 1];
        u->num_res = num_res;
        u->upsample = i != levels - 1;
        u->factor_t = (u->upsample && tdown[levels - 2 - i] != 0) ? 2 : 1;
        for (int r = 0; ok && r < num_res; r++) {
            snprintf(name, sizeof(name), "decoder.up_blocks.%d.resnets.%d", i, r);
            ok = load_res(sf, &u->res[r], name, r ? u->cout : u->cin, u->cout);
        }
        if (ok && u->upsample) {
            if ((u->cout * u->factor_t * 4) % u->cin) {
                fprintf(stderr, "Qwen-Image VAE: up block %d widths %d -> %d\n",
                        i, u->cin, u->cout);
                ok = 0;
                break;
            }
            snprintf(name, sizeof(name), "decoder.up_blocks.%d.upsampler.resample.1", i);
            ok = load_conv(sf, &u->up_conv, name, u->cout, u->cout, 3);
        }
    }
    if (ok) {
        int last = dims[levels];
        vae->norm_out = load_f32(sf, "decoder.norm_out.gamma", last);
        ok = vae->norm_out &&
             load_conv(sf, &vae->conv_out, "decoder.conv_out", last, vae->out_channels, 3);
    }
    safetensors_close(sf);

    if (!ok) {
        iris_vae_free_qimage(vae);
        return NULL;
    }
    if (iris_verbose)
        fprintf(stderr, "  Qwen-Image VAE: z=%d base=%d levels=%d out=%d\n",
                vae->z_dim, base, levels, vae->out_channels);
    return vae;
}

void iris_vae_free_qimage(iris_vae_qimage_t *vae) {
    if (!vae) return;
#ifdef IRIS_DEV
    dv_free(vae);
#endif
    free_conv(&vae->post_quant);
    free_conv(&vae->conv_in);
    free_conv(&vae->conv_out);
    free_res(&vae->mid_res[0]);
    free_res(&vae->mid_res[1]);
    free(vae->attn_gamma);
    free_conv(&vae->attn_qkv);
    free_conv(&vae->attn_proj);
    for (int i = 0; i < vae->num_up; i++) {
        for (int r = 0; r < vae->up[i].num_res; r++) free_res(&vae->up[i].res[r]);
        free_conv(&vae->up[i].up_conv);
    }
    free(vae->norm_out);
    free(vae);
}
