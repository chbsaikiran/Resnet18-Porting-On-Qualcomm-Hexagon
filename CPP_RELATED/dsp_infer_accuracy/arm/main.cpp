/*
 * main.cpp
 *
 * ARM-side host application for CIFAR-10 accuracy calculation
 * with inference offloaded to the Hexagon DSP via FastRPC.
 *
 * What it does:
 *   1. Sends the model path to the DSP (cifar10_infer_init)
 *   2. Loads CIFAR-10 test images on ARM and preprocesses them
 *   3. Sends each preprocessed tensor to the DSP for inference
 *   4. Collects predictions, computes overall & per-class accuracy
 *
 * Usage:
 *   ./cifar10_dsp_accuracy <model_path> <data_path>
 *
 *   model_path  – path on the device, e.g. /data/local/tmp/resnet18_cifar10.dlc
 *   data_path   – CIFAR-10 binary file (test_batch.bin) on the device
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <chrono>

#include "cifar10_infer.h"
#include "rpcmem.h"

#define ION_HEAP_ID_SYSTEM  25

static const float MEAN[3] = {0.4914f, 0.4822f, 0.4465f};
static const float STD[3]  = {0.2470f, 0.2435f, 0.2616f};

static const char *CLASS_NAMES[CIFAR10_NUM_CLASSES] = {
    "airplane", "automobile", "bird",  "cat",   "deer",
    "dog",      "frog",       "horse", "ship",  "truck",
};

/* ------------------------------------------------------------------ */
/*  ION buffer helpers                                                 */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/*  Load CIFAR-10 binary test batch and normalise on ARM               */
/*  Format: 10000 records, each = 1 byte label + 3072 bytes image      */
/*          (1024 R + 1024 G + 1024 B, row-major)                      */
/* ------------------------------------------------------------------ */

struct ValSample {
    int label;
};

static int load_cifar10_bin(const std::string &bin_path,
                            std::vector<ValSample> &samples,
                            std::vector<std::vector<float>> &pixels)
{
    std::ifstream ifs(bin_path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Cannot open " << bin_path << "\n";
        return -1;
    }

    const int spatial     = CIFAR10_IMG_SIZE * CIFAR10_IMG_SIZE;
    const int record_size = 1 + 3 * spatial;
    std::vector<uint8_t> record(record_size);

    while (ifs.read(reinterpret_cast<char *>(record.data()), record_size)) {
        ValSample s;
        s.label = static_cast<int>(record[0]);
        samples.push_back(s);

        std::vector<float> px(CIFAR10_INPUT_FLOATS);
        const uint8_t *raw = record.data() + 1;
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < spatial; i++) {
                float v = static_cast<float>(raw[c * spatial + i]) / 255.0f;
                px[c * spatial + i] = (v - MEAN[c]) / STD[c];
            }
        }
        pixels.push_back(std::move(px));
    }

    return static_cast<int>(samples.size());
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_path> <test_batch.bin>\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string data_path  = argv[2];

    /* ---- Init FastRPC memory ---------------------------------------- */

    rpcmem_init();

    uint8_t *buf_input  = alloc_rpcmem(CIFAR10_INPUT_BYTES);
    uint8_t *buf_output = alloc_rpcmem(CIFAR10_OUTPUT_BYTES);
    uint8_t *buf_model  = alloc_rpcmem(static_cast<int>(model_path.size() + 1));

    if (!buf_input || !buf_output || !buf_model) {
        std::cerr << "rpcmem_alloc failed\n";
        free_rpcmem(buf_input);
        free_rpcmem(buf_output);
        free_rpcmem(buf_model);
        rpcmem_deinit();
        return 1;
    }

    /* ---- Init the DSP side ------------------------------------------ */

    std::memcpy(buf_model, model_path.c_str(), model_path.size() + 1);

    AEEResult rc = cifar10_infer_init(buf_model,
                                      static_cast<int>(model_path.size() + 1));
    if (rc != AEE_SUCCESS) {
        std::cerr << "cifar10_infer_init failed: " << rc << "\n";
        goto cleanup;
    }

    {
        int version = 0;
        cifar10_infer_get_param(CIFAR10_PARAM_VERSION, &version);
        int num_cls = 0;
        cifar10_infer_get_param(CIFAR10_PARAM_NUM_CLASSES, &num_cls);
        std::cout << "DSP backend version : " << version << "\n"
                  << "DSP num_classes     : " << num_cls  << "\n"
                  << "Model path          : " << model_path << "\n\n";
    }

    /* ---- Load test data on ARM -------------------------------------- */

    {
        std::vector<ValSample> samples;
        std::vector<std::vector<float>> pixels;

        int total_images = load_cifar10_bin(data_path, samples, pixels);
        if (total_images <= 0) {
            std::cerr << "No images found at " << data_path << "\n";
            rc = AEE_EFAILED;
            goto cleanup;
        }

        std::cout << "Test images loaded: " << total_images << "\n\n";

        /* ---- Inference loop (each image → DSP via FastRPC) ---------- */

        int correct = 0;
        int total   = 0;
        int per_class_correct[CIFAR10_NUM_CLASSES] = {};
        int per_class_total[CIFAR10_NUM_CLASSES]   = {};
        double total_inference_ms = 0.0;

        for (int idx = 0; idx < total_images; idx++) {
            std::memcpy(buf_input, pixels[idx].data(), CIFAR10_INPUT_BYTES);

            int output_lenout = 0;

            auto t0 = std::chrono::high_resolution_clock::now();

            rc = cifar10_infer_process(buf_input,  CIFAR10_INPUT_BYTES,
                                       buf_output, CIFAR10_OUTPUT_BYTES,
                                       &output_lenout);

            auto t1 = std::chrono::high_resolution_clock::now();

            if (rc != AEE_SUCCESS) {
                std::cerr << "cifar10_infer_process failed at image "
                          << idx << ": " << rc << "\n";
                goto cleanup;
            }

            total_inference_ms +=
                std::chrono::duration<double, std::milli>(t1 - t0).count();

            float logits[CIFAR10_NUM_CLASSES];
            std::memcpy(logits, buf_output, CIFAR10_OUTPUT_BYTES);

            int pred = static_cast<int>(
                std::max_element(logits, logits + CIFAR10_NUM_CLASSES) - logits);

            int gt = samples[idx].label;
            per_class_total[gt]++;
            total++;

            if (pred == gt) {
                correct++;
                per_class_correct[gt]++;
            }

            if ((idx + 1) % 1000 == 0 || idx == total_images - 1) {
                double running_acc = 100.0 * correct / total;
                std::printf("[%5d / %5d]  running accuracy: %.2f%%\n",
                            idx + 1, total_images, running_acc);
            }
        }

        /* ---- Results ------------------------------------------------ */

        double accuracy = 100.0 * correct / total;

        std::cout << "\n====== CIFAR-10 DSP Validation Results ======\n";
        std::cout << "Total images : " << total   << "\n";
        std::cout << "Correct      : " << correct << "\n";
        std::printf("Top-1 Accuracy: %.2f%%\n", accuracy);
        std::printf("Avg DSP infer : %.3f ms  (includes FastRPC round-trip)\n",
                    total_inference_ms / total);
        std::cout << "=============================================\n";

        std::cout << "\nPer-class accuracy:\n";
        for (int c = 0; c < CIFAR10_NUM_CLASSES; c++) {
            if (per_class_total[c] > 0) {
                double cls_acc = 100.0 * per_class_correct[c]
                                       / per_class_total[c];
                std::printf("  [%d] %-12s  %4d / %4d  (%.2f%%)\n",
                            c, CLASS_NAMES[c],
                            per_class_correct[c], per_class_total[c], cls_acc);
            }
        }
    }

cleanup:
    cifar10_infer_deinit();
    free_rpcmem(buf_input);
    free_rpcmem(buf_output);
    free_rpcmem(buf_model);
    rpcmem_deinit();

    return (rc == AEE_SUCCESS) ? 0 : 1;
}
