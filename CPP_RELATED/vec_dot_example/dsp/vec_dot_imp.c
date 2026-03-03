/*
 * vec_dot_imp.c
 *
 * DSP-side implementation of the vec_dot FastRPC interface.
 * Runs on the Hexagon DSP.  Pure C with no external SDK dependencies,
 * making it an ideal first test for FastRPC on a Snapdragon board.
 *
 * Build output: libvec_dot_skel.so
 *   Push to: /vendor/lib/rfsa/dsp/  (or /usr/lib/rfsa/adsp/)
 */

#include "vec_dot.h"
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Internal state                                                     */
/* ------------------------------------------------------------------ */

static int g_initialised = 0;
static int g_vec_len     = VEC_DOT_DEFAULT_LEN;
static int g_use_hvx     = 0;

#define IMP_VERSION  1

/* ------------------------------------------------------------------ */
/*  init                                                               */
/* ------------------------------------------------------------------ */
AEEResult vec_dot_init(void)
{
    g_initialised = 1;
    g_vec_len     = VEC_DOT_DEFAULT_LEN;
    g_use_hvx     = 0;
    return AEE_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  deinit                                                             */
/* ------------------------------------------------------------------ */
AEEResult vec_dot_deinit(void)
{
    g_initialised = 0;
    return AEE_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  set_param                                                          */
/* ------------------------------------------------------------------ */
AEEResult vec_dot_set_param(int param_id, int value)
{
    switch (param_id) {
    case VEC_DOT_PARAM_VEC_LEN:
        if (value <= 0) return AEE_EBADPARM;
        g_vec_len = value;
        return AEE_SUCCESS;
    case VEC_DOT_PARAM_USE_HVX:
        g_use_hvx = (value != 0) ? 1 : 0;
        return AEE_SUCCESS;
    default:
        return AEE_EBADPARM;
    }
}

/* ------------------------------------------------------------------ */
/*  get_param                                                          */
/* ------------------------------------------------------------------ */
AEEResult vec_dot_get_param(int param_id, int *value)
{
    if (!value)
        return AEE_EBADPARM;

    switch (param_id) {
    case VEC_DOT_PARAM_VEC_LEN:  *value = g_vec_len;   break;
    case VEC_DOT_PARAM_USE_HVX:  *value = g_use_hvx;   break;
    case VEC_DOT_PARAM_VERSION:  *value = IMP_VERSION;  break;
    default:
        return AEE_EBADPARM;
    }
    return AEE_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  process  –  dot product of two float32 vectors                     */
/* ------------------------------------------------------------------ */
AEEResult vec_dot_process(const uint8_t *vec_a,   int vec_a_len,
                          const uint8_t *vec_b,   int vec_b_len,
                          uint8_t       *result,  int result_len,
                          int           *result_lenout)
{
    if (!g_initialised)
        return AEE_ENOTINITIALIZED;
    if (!vec_a || !vec_b || !result || !result_lenout)
        return AEE_EBADPARM;

    int expected_bytes = g_vec_len * (int)sizeof(float);

    if (vec_a_len < expected_bytes || vec_b_len < expected_bytes)
        return AEE_EBADPARM;
    if (result_len < (int)sizeof(float))
        return AEE_EBADPARM;

    const float *a = (const float *)vec_a;
    const float *b = (const float *)vec_b;
    int n = g_vec_len;

    /*
     * Scalar dot product.
     *
     * For a production HVX version you would replace this loop with
     * Hexagon Vector eXtension intrinsics, e.g.:
     *
     *   #include <hexagon_types.h>
     *   #include <hvx_hexagon_protos.h>
     *   HVX_Vector va = *(HVX_Vector *)&a[i];
     *   ...
     *
     * Toggle via set_param(VEC_DOT_PARAM_USE_HVX, 1) once implemented.
     */

    float dot = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        dot += a[i] * b[i];
    }

    memcpy(result, &dot, sizeof(float));
    *result_lenout = (int)sizeof(float);

    return AEE_SUCCESS;
}
