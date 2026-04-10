/*
 * main_snpe.cpp — Demo C++ app: load ResNet18 CIFAR-10 DLC via SNPE C++ API.
 *
 * Input: float32 raw file, NCHW layout [1,3,32,32] = 12288 bytes (same as
 *        prepare_cifar10_inputs_snpe.py in dsp_infer_accuracy).
 *
 * Usage (on device):
 *   ./cifar10_snpe_demo <input_nchw.raw> [runtime] [dlc_path]
 *
 *   runtime   — dsp (default) | gpu | cpu
 *   dlc_path  — defaults to /data/local/tmp/demo_snpe_app/resnet18_cifar10_quantized.dlc
 *
 * Build: ./build_android_snpe.sh
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <strings.h>

#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/PlatformConfig.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/TensorShape.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"

static const char *const kDefaultDlc =
    "/data/local/tmp/demo_snpe_app/resnet18_cifar10_quantized.dlc";

static const char *CLASSES[] = {
    "airplane", "automobile", "bird", "cat",  "deer",
    "dog",      "frog",       "horse", "ship", "truck"
};

static zdl::DlSystem::Runtime_t parse_runtime(const char *s) {
    if (!s || s[0] == '\0')
        return zdl::DlSystem::Runtime_t::DSP;
    if (strcasecmp(s, "dsp") == 0)
        return zdl::DlSystem::Runtime_t::DSP;
    if (strcasecmp(s, "gpu") == 0)
        return zdl::DlSystem::Runtime_t::GPU;
    if (strcasecmp(s, "cpu") == 0)
        return zdl::DlSystem::Runtime_t::CPU;
    fprintf(stderr, "ERROR: unknown runtime '%s'. Use: dsp | gpu | cpu\n", s);
    std::exit(1);
}

static const char *runtime_name(zdl::DlSystem::Runtime_t r) {
    using R = zdl::DlSystem::Runtime_t;
    if (r == R::DSP || r == R::DSP_FIXED8_TF)  return "DSP";
    if (r == R::GPU || r == R::GPU_FLOAT32_16_HYBRID || r == R::GPU_FLOAT16)
        return "GPU";
    if (r == R::CPU || r == R::CPU_FLOAT32) return "CPU";
    return "UNKNOWN";
}

static float *read_raw_file(const char *path, size_t expected_bytes) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        std::exit(1);
    }
    float *buf = static_cast<float *>(malloc(expected_bytes));
    if (!buf) {
        fclose(fp);
        fprintf(stderr, "ERROR: malloc failed\n");
        std::exit(1);
    }
    size_t n = fread(buf, 1, expected_bytes, fp);
    fclose(fp);
    if (n != expected_bytes) {
        fprintf(stderr, "ERROR: %s is %zu bytes, expected %zu\n",
                path, n, expected_bytes);
        free(buf);
        std::exit(1);
    }
    return buf;
}

int main(int argc, char **argv) {
    const char *input_path = nullptr;
    const char *runtime_str = "dsp";
    const char *dlc_path = kDefaultDlc;

    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <input_nchw.raw> [dsp|gpu|cpu] [dlc_path]\n\n"
                "  input_nchw.raw — float32 NCHW 1x3x32x32 = 12288 bytes\n"
                "  runtime        — dsp (default), gpu, or cpu\n"
                "  dlc_path       — optional (default: %s)\n",
                argv[0], kDefaultDlc);
        return 1;
    }

    input_path = argv[1];
    if (argc >= 3)
        runtime_str = argv[2];
    if (argc >= 4)
        dlc_path = argv[3];

    zdl::DlSystem::Runtime_t runtime = parse_runtime(runtime_str);

    const size_t kInputBytes =
        static_cast<size_t>(1) * 3 * 32 * 32 * sizeof(float);
    const size_t kOutputFloats = 10;

    printf("============================================================\n");
    printf("  CIFAR-10 SNPE Inference Demo (C++)\n");
    printf("============================================================\n");
    printf("  DLC      : %s\n", dlc_path);
    printf("  Input    : %s\n", input_path);
    printf("  Runtime  : %s\n\n", runtime_name(runtime));

    auto container = zdl::DlContainer::IDlContainer::open(std::string(dlc_path));
    if (!container) {
        fprintf(stderr, "ERROR: IDlContainer::open(%s) failed\n", dlc_path);
        return 1;
    }

    zdl::SNPE::SNPEBuilder builder(container.get());
    builder.setRuntimeProcessor(runtime);
    builder.setUseUserSuppliedBuffers(false);
    // Match snpe-net-run --platform_options="unsignedPD:OFF" for DSP loads.
    if (runtime == zdl::DlSystem::Runtime_t::DSP ||
        runtime == zdl::DlSystem::Runtime_t::DSP_FIXED8_TF) {
        zdl::DlSystem::PlatformConfig platformConfig;
        if (platformConfig.setPlatformOptions(std::string("unsignedPD:OFF"))) {
            builder.setPlatformConfig(platformConfig);
        }
    }

    std::unique_ptr<zdl::SNPE::SNPE> snpe(builder.build());
    if (!snpe) {
        fprintf(stderr, "ERROR: SNPEBuilder::build() failed\n");
        return 1;
    }

    float *input_host = read_raw_file(input_path, kInputBytes);
    std::unique_ptr<float, decltype(&free)> input_guard(input_host, &free);

    zdl::DlSystem::TensorShape inShape{1, 3, 32, 32};
    auto inputTensor =
        zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inShape);
    if (!inputTensor) {
        fprintf(stderr, "ERROR: createTensor failed\n");
        return 1;
    }

    float *tensor_ptr =
        reinterpret_cast<float *>(inputTensor->begin().dataPointer());
    if (!tensor_ptr) {
        fprintf(stderr, "ERROR: input tensor has no writable pointer\n");
        return 1;
    }
    memcpy(tensor_ptr, input_host, kInputBytes);

    zdl::DlSystem::TensorMap outputMap;
    if (!snpe->execute(inputTensor.get(), outputMap)) {
        fprintf(stderr, "ERROR: SNPE::execute failed\n");
        return 1;
    }

    if (outputMap.size() == 0) {
        fprintf(stderr, "ERROR: empty output map\n");
        return 1;
    }

    zdl::DlSystem::StringList outNames = outputMap.getTensorNames();
    if (outNames.size() == 0) {
        fprintf(stderr, "ERROR: no output tensor names\n");
        return 1;
    }
    zdl::DlSystem::ITensor *outTensor = outputMap.getTensor(outNames.at(0));
    if (!outTensor) {
        fprintf(stderr, "ERROR: output tensor '%s' not found\n", outNames.at(0));
        return 1;
    }
    const float *logits = reinterpret_cast<const float *>(
        outTensor->begin().dataPointer());
    if (!logits) {
        fprintf(stderr, "ERROR: output tensor has no data pointer\n");
        return 1;
    }

    // SNPE ITensor::getSize() is the element count (not bytes).
    size_t out_count = outTensor->getSize();
    if (out_count < kOutputFloats) {
        fprintf(stderr, "WARNING: expected at least %zu output elements, got %zu\n",
                kOutputFloats, out_count);
    }

    int best = 0;
    size_t ncls = out_count < kOutputFloats ? out_count : kOutputFloats;
    printf("  Class logits:\n");
    for (size_t i = 0; i < ncls; i++) {
        printf("    [%zu] %-12s  %+.4f", i, CLASSES[i], logits[i]);
        if (logits[i] > logits[best])
            best = static_cast<int>(i);
        if (static_cast<int>(i) == best)
            printf("  <--");
        printf("\n");
    }
    printf("\n  *** Prediction: %s (class %d) ***\n\n",
           CLASSES[best], best);

    return 0;
}
