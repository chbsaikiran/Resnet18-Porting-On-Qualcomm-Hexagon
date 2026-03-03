/*
 * main.cpp
 *
 * ARM-side host application for the vector dot-product FastRPC example.
 * Runs on the Snapdragon applications processor (ARM).
 *
 * What it does:
 *   1. Allocates ION buffers for zero-copy FastRPC transport
 *   2. Fills two vectors with known values on ARM
 *   3. Sends them to the Hexagon DSP which computes the dot product
 *   4. Reads back and displays the result on ARM
 *
 * Usage:
 *   ./vec_dot_arm [vector_length]
 *   Default vector_length = 1024
 */

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <chrono>

#include "vec_dot.h"
#include "rpcmem.h"

#define ION_HEAP_ID_SYSTEM  25

static uint8_t *alloc_rpcmem(int size)
{
    return (uint8_t *)rpcmem_alloc(ION_HEAP_ID_SYSTEM,
                                   RPCMEM_DEFAULT_FLAGS,
                                   size);
}

static void free_rpcmem(void *p)
{
    if (p) rpcmem_free(p);
}

/* Reference dot product computed on ARM for verification */
static float dot_product_arm(const float *a, const float *b, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

int main(int argc, char **argv)
{
    int vec_len = VEC_DOT_DEFAULT_LEN;
    if (argc >= 2)
        vec_len = atoi(argv[1]);

    if (vec_len <= 0) {
        std::cerr << "Invalid vector length\n";
        return 1;
    }

    int vec_bytes    = vec_len * (int)sizeof(float);
    int result_bytes = (int)sizeof(float);

    std::cout << "=== Vector Dot Product – FastRPC DSP Example ===\n"
              << "Vector length : " << vec_len << " floats ("
              << vec_bytes << " bytes)\n\n";

    /* ---- Init FastRPC memory ---------------------------------------- */
    rpcmem_init();

    uint8_t *buf_a   = alloc_rpcmem(vec_bytes);
    uint8_t *buf_b   = alloc_rpcmem(vec_bytes);
    uint8_t *buf_res = alloc_rpcmem(result_bytes);

    if (!buf_a || !buf_b || !buf_res) {
        std::cerr << "rpcmem_alloc failed\n";
        free_rpcmem(buf_a);
        free_rpcmem(buf_b);
        free_rpcmem(buf_res);
        rpcmem_deinit();
        return 1;
    }

    /* ---- Fill vectors with test data -------------------------------- */
    float *vec_a = reinterpret_cast<float *>(buf_a);
    float *vec_b = reinterpret_cast<float *>(buf_b);

    for (int i = 0; i < vec_len; i++) {
        vec_a[i] = 1.0f;                         /* a = [1, 1, 1, ...] */
        vec_b[i] = static_cast<float>(i + 1);    /* b = [1, 2, 3, ...] */
    }
    /* Expected: dot(a, b) = 1+2+3+...+N = N*(N+1)/2 */
    float expected = static_cast<float>(vec_len) *
                     static_cast<float>(vec_len + 1) / 2.0f;

    /* ---- Init the DSP side ------------------------------------------ */
    AEEResult rc = vec_dot_init();
    if (rc != AEE_SUCCESS) {
        std::cerr << "vec_dot_init failed: " << rc << "\n";
        goto cleanup;
    }

    /* Tell DSP the vector length */
    rc = vec_dot_set_param(VEC_DOT_PARAM_VEC_LEN, vec_len);
    if (rc != AEE_SUCCESS) {
        std::cerr << "vec_dot_set_param(VEC_LEN) failed: " << rc << "\n";
        goto cleanup;
    }

    /* Read back to verify round-trip */
    {
        int readback = 0;
        vec_dot_get_param(VEC_DOT_PARAM_VEC_LEN, &readback);
        std::cout << "DSP reports vec_len = " << readback << "\n";

        int version = 0;
        vec_dot_get_param(VEC_DOT_PARAM_VERSION, &version);
        std::cout << "DSP implementation version = " << version << "\n\n";
    }

    /* ---- Run dot product on DSP ------------------------------------- */
    {
        int result_lenout = 0;

        auto t0 = std::chrono::high_resolution_clock::now();

        rc = vec_dot_process(buf_a,   vec_bytes,
                             buf_b,   vec_bytes,
                             buf_res, result_bytes,
                             &result_lenout);

        auto t1 = std::chrono::high_resolution_clock::now();

        if (rc != AEE_SUCCESS) {
            std::cerr << "vec_dot_process failed: " << rc << "\n";
            goto cleanup;
        }

        double elapsed_us =
            std::chrono::duration<double, std::micro>(t1 - t0).count();

        float dsp_result;
        std::memcpy(&dsp_result, buf_res, sizeof(float));

        /* ARM-side reference for comparison */
        float arm_result = dot_product_arm(vec_a, vec_b, vec_len);

        std::cout << "====== Results ======\n"
                  << "DSP result    : " << dsp_result  << "\n"
                  << "ARM result    : " << arm_result  << "\n"
                  << "Expected      : " << expected    << "\n"
                  << "Match         : "
                  << (std::fabs(dsp_result - expected) < 1.0f ? "YES" : "NO")
                  << "\n"
                  << "DSP time (us) : " << elapsed_us  << "\n"
                  << "=====================\n";
    }

    /* ---- Run a few more iterations for timing ----------------------- */
    std::cout << "\nRunning 100 iterations for timing ...\n";
    {
        double total_us = 0.0;
        const int iterations = 100;

        for (int iter = 0; iter < iterations; iter++) {
            int result_lenout = 0;

            auto t0 = std::chrono::high_resolution_clock::now();

            vec_dot_process(buf_a,   vec_bytes,
                            buf_b,   vec_bytes,
                            buf_res, result_bytes,
                            &result_lenout);

            auto t1 = std::chrono::high_resolution_clock::now();

            total_us +=
                std::chrono::duration<double, std::micro>(t1 - t0).count();
        }

        std::printf("Avg round-trip (FastRPC + DSP compute): %.1f us\n",
                    total_us / iterations);
    }

cleanup:
    vec_dot_deinit();
    free_rpcmem(buf_a);
    free_rpcmem(buf_b);
    free_rpcmem(buf_res);
    rpcmem_deinit();

    return (rc == AEE_SUCCESS) ? 0 : 1;
}
