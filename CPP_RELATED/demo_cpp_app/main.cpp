/*
 * main.cpp — Demo C++ application that loads the QNN .so files and runs
 *            ResNet18 CIFAR-10 inference.
 *
 * THIS IS EXACTLY HOW qnn-net-run WORKS INTERNALLY.
 *
 * The two .so files involved:
 *   1. libQnnCpu.so        — the QNN BACKEND (inference engine)
 *   2. libresnet18_cifar10.so — the MODEL (graph + weights)
 *
 * Both are loaded at runtime via dlopen(). No link-time dependency.
 *
 * Build:  ./build.sh
 * Run:    ./build/cifar10_qnn_app <input.raw>
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>     // dlopen, dlsym, dlclose
#include <algorithm>

#include "QnnInterface.h"
#include "QnnTypes.h"
#include "QnnCommon.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"

// ---- Types exported by the model .so (libresnet18_cifar10.so) ----
// These are defined in QnnWrapperUtils.hpp from the QAIRT SDK.

typedef enum { MODEL_NO_ERROR = 0 } ModelError_t;

typedef struct {
    Qnn_GraphHandle_t graph;
    char             *graphName;
    Qnn_Tensor_t     *inputTensors;
    uint32_t          numInputTensors;
    Qnn_Tensor_t     *outputTensors;
    uint32_t          numOutputTensors;
} GraphInfo_t;
typedef GraphInfo_t *GraphInfoPtr_t;

typedef struct {
    char                     *graphName;
    const QnnGraph_Config_t **graphConfigs;
} GraphConfigInfo_t;

// Function pointer types for the model .so exports
typedef ModelError_t (*ComposeGraphsFn_t)(
    Qnn_BackendHandle_t       backend,
    QNN_INTERFACE_VER_TYPE    qnnInterface,
    Qnn_ContextHandle_t       context,
    const GraphConfigInfo_t **graphsConfigInfo,
    const uint32_t            numGraphsConfigInfo,
    GraphInfoPtr_t          **graphsInfo,
    uint32_t                 *numGraphsInfo,
    bool                      debug,
    QnnLog_Callback_t         logCallback,
    QnnLog_Level_t            maxLogLevel);

typedef ModelError_t (*FreeGraphsInfoFn_t)(
    GraphInfoPtr_t **graphsInfo,
    uint32_t         numGraphsInfo);

// Function pointer type for backend .so export
typedef Qnn_ErrorHandle_t (*GetProvidersFn_t)(
    const QnnInterface_t ***providerList,
    uint32_t               *numProviders);

// ---- CIFAR-10 class names ----
static const char *CLASSES[] = {
    "airplane", "automobile", "bird", "cat",  "deer",
    "dog",      "frog",       "horse", "ship", "truck"
};

// ---- Helper: load a .so and return its dlopen handle ----
static void *load_library(const char *path) {
    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "ERROR: dlopen(%s) failed: %s\n", path, dlerror());
        exit(1);
    }
    printf("  Loaded: %s\n", path);
    return handle;
}

// ---- Helper: look up a symbol in a .so ----
static void *get_symbol(void *handle, const char *name) {
    void *sym = dlsym(handle, name);
    if (!sym) {
        fprintf(stderr, "ERROR: dlsym(%s) failed: %s\n", name, dlerror());
        exit(1);
    }
    return sym;
}

// ---- Helper: read a binary file into a malloc'd buffer ----
static float *read_raw_file(const char *path, size_t expected_bytes) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        exit(1);
    }
    float *buf = (float *)malloc(expected_bytes);
    size_t n = fread(buf, 1, expected_bytes, fp);
    fclose(fp);
    if (n != expected_bytes) {
        fprintf(stderr, "ERROR: %s is %zu bytes, expected %zu\n", path, n, expected_bytes);
        exit(1);
    }
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <input.raw> [backend.so] [model.so]\n\n", argv[0]);
        printf("  input.raw  — float32 NHWC tensor [1,32,32,3] = 12288 bytes\n");
        printf("  backend.so — path to libQnnCpu.so (default: from QAIRT SDK)\n");
        printf("  model.so   — path to libresnet18_cifar10.so (default: from lib/)\n");
        return 1;
    }

    const char *input_path   = argv[1];
    const char *backend_path = argc > 2 ? argv[2]
        : "/opt/qcom/aistack/qairt/2.31.0.250130/lib/x86_64-linux-clang/libQnnCpu.so";
    const char *model_path   = argc > 3 ? argv[3]
        : "../dsp_infer_accuracy/lib/x86_64-linux-clang/libresnet18_cifar10.so";

    printf("============================================================\n");
    printf("  CIFAR-10 QNN Inference Demo (C++)\n");
    printf("  This app loads .so files via dlopen and uses the QNN C API\n");
    printf("============================================================\n\n");

    // ==================================================================
    // STEP 1: dlopen() the BACKEND .so  (e.g. libQnnCpu.so)
    //
    // This is the inference engine. On a phone you'd swap in
    // libQnnDsp.so for Hexagon DSP or libQnnGpu.so for Adreno GPU.
    // The .so exports one key function: QnnInterface_getProviders
    // ==================================================================
    printf("[Step 1] Loading backend .so ...\n");
    void *backend_lib = load_library(backend_path);

    auto getProviders = (GetProvidersFn_t)get_symbol(
        backend_lib, "QnnInterface_getProviders");

    // Call getProviders to obtain the QNN function table.
    // This single struct contains pointers to ALL QNN API functions:
    // backendCreate, contextCreate, graphCreate, graphExecute, etc.
    const QnnInterface_t **providers = nullptr;
    uint32_t numProviders = 0;

    if (getProviders(&providers, &numProviders) != QNN_SUCCESS
        || numProviders == 0) {
        fprintf(stderr, "ERROR: QnnInterface_getProviders failed\n");
        return 1;
    }

    // The function table — this is the "API" we use for everything
    QNN_INTERFACE_VER_TYPE qnn = providers[0]->QNN_INTERFACE_VER_NAME;
    printf("  QNN provider: %s\n\n", providers[0]->providerName);

    // ==================================================================
    // STEP 2: Create a QNN Backend handle
    //
    // The backend handle represents an instance of the inference engine.
    // ==================================================================
    printf("[Step 2] Creating QNN backend ...\n");
    Qnn_BackendHandle_t backend = nullptr;
    if (qnn.backendCreate(/*logger=*/nullptr, /*config=*/nullptr, &backend)
        != QNN_SUCCESS) {
        fprintf(stderr, "ERROR: backendCreate failed\n");
        return 1;
    }
    printf("  Backend created.\n\n");

    // ==================================================================
    // STEP 3: Create a QNN Context
    //
    // A context is a container for one or more graphs. It manages
    // memory and resources.
    // ==================================================================
    printf("[Step 3] Creating QNN context ...\n");
    Qnn_ContextHandle_t context = nullptr;
    if (qnn.contextCreate(backend, /*device=*/nullptr,
                          /*config=*/nullptr, &context)
        != QNN_SUCCESS) {
        fprintf(stderr, "ERROR: contextCreate failed\n");
        return 1;
    }
    printf("  Context created.\n\n");

    // ==================================================================
    // STEP 4: dlopen() the MODEL .so  (libresnet18_cifar10.so)
    //
    // This .so contains the graph definition (all the Conv, ReLU,
    // BatchNorm, etc. operations) and the trained weights (11M params).
    //
    // It exports two functions:
    //   QnnModel_composeGraphs  — builds the graph inside a context
    //   QnnModel_freeGraphsInfo — cleanup
    // ==================================================================
    printf("[Step 4] Loading model .so ...\n");
    void *model_lib = load_library(model_path);

    auto composeGraphs = (ComposeGraphsFn_t)get_symbol(
        model_lib, "QnnModel_composeGraphs");
    auto freeGraphsInfo = (FreeGraphsInfoFn_t)get_symbol(
        model_lib, "QnnModel_freeGraphsInfo");
    printf("\n");

    // ==================================================================
    // STEP 5: Compose the graph
    //
    // This calls into the model .so, which uses the QNN API (via the
    // function table we got in step 1) to add all tensors and nodes
    // to the graph. After this call, the graph is fully defined but
    // not yet optimised for execution.
    //
    // graphsInfo will contain the graph handle, plus metadata about
    // input and output tensors (names, shapes, types).
    // ==================================================================
    printf("[Step 5] Composing graph (adding all ops + weights) ...\n");
    GraphInfoPtr_t *graphsInfo = nullptr;
    uint32_t numGraphs = 0;

    ModelError_t modelErr = composeGraphs(
        backend, qnn, context,
        /*graphsConfigInfo=*/nullptr, /*numGraphsConfigInfo=*/0,
        &graphsInfo, &numGraphs,
        /*debug=*/false,
        /*logCallback=*/nullptr, QNN_LOG_LEVEL_ERROR);

    if (modelErr != MODEL_NO_ERROR || numGraphs == 0) {
        fprintf(stderr, "ERROR: QnnModel_composeGraphs failed (err=%d)\n",
                modelErr);
        return 1;
    }

    GraphInfo_t *gi = graphsInfo[0];
    printf("  Graph name: %s\n", gi->graphName);
    printf("  Inputs:  %u tensor(s)\n", gi->numInputTensors);
    printf("  Outputs: %u tensor(s)\n\n", gi->numOutputTensors);

    // ==================================================================
    // STEP 6: Finalize the graph
    //
    // The backend optimises the graph for the target hardware:
    //   - Operator fusion
    //   - Memory planning
    //   - Tiling (DSP), kernel selection (GPU), SIMD scheduling (CPU)
    //
    // After this, the graph is ready to execute.
    // ==================================================================
    printf("[Step 6] Finalizing graph (backend optimisation) ...\n");
    if (qnn.graphFinalize(gi->graph, /*profile=*/nullptr, /*signal=*/nullptr)
        != QNN_SUCCESS) {
        fprintf(stderr, "ERROR: graphFinalize failed\n");
        return 1;
    }
    printf("  Graph finalised and ready.\n\n");

    // ==================================================================
    // STEP 7: Prepare input and output tensors
    //
    // Read the raw float32 input file (NHWC: 1x32x32x3 = 12288 bytes)
    // and set up QNN tensor descriptors pointing to our data buffers.
    //
    // The graph told us what input/output tensors it expects (step 5).
    // We copy those descriptors and fill in our own data buffers.
    // ==================================================================
    printf("[Step 7] Preparing input/output tensors ...\n");

    const size_t INPUT_BYTES  = 1 * 32 * 32 * 3 * sizeof(float);  // 12288
    const size_t OUTPUT_BYTES = 1 * 10 * sizeof(float);            // 40

    float *input_data  = read_raw_file(input_path, INPUT_BYTES);
    float *output_data = (float *)calloc(10, sizeof(float));

    // Copy the tensor descriptors from the graph and attach our buffers.
    // The graph's tensors already have the correct names, shapes, types;
    // we just need to set the memory type and client buffer.
    Qnn_Tensor_t inputTensor  = gi->inputTensors[0];
    Qnn_Tensor_t outputTensor = gi->outputTensors[0];

    Qnn_ClientBuffer_t inBuf  = { .data = input_data,  .dataSize = (uint32_t)INPUT_BYTES };
    Qnn_ClientBuffer_t outBuf = { .data = output_data, .dataSize = (uint32_t)OUTPUT_BYTES };

    // Access the v2 struct to set client buffer fields
    inputTensor.v2.memType   = QNN_TENSORMEMTYPE_RAW;
    inputTensor.v2.clientBuf = inBuf;
    outputTensor.v2.memType   = QNN_TENSORMEMTYPE_RAW;
    outputTensor.v2.clientBuf = outBuf;

    printf("  Input:  %zu bytes (NHWC 1x32x32x3 float32)\n", INPUT_BYTES);
    printf("  Output: %zu bytes (10 class logits float32)\n\n", OUTPUT_BYTES);

    // ==================================================================
    // STEP 8: Execute inference!
    //
    // This is the actual forward pass through the neural network.
    // The backend processes the input through all 18 layers of ResNet
    // and writes the 10 output logits to our output buffer.
    // ==================================================================
    printf("[Step 8] Executing inference ...\n");
    Qnn_ErrorHandle_t execErr = qnn.graphExecute(
        gi->graph,
        &inputTensor,  1,    // 1 input tensor
        &outputTensor, 1,    // 1 output tensor
        /*profile=*/nullptr,
        /*signal=*/nullptr);

    if (execErr != QNN_SUCCESS) {
        fprintf(stderr, "ERROR: graphExecute failed (code=%lu)\n",
                (unsigned long)execErr);
        return 1;
    }
    printf("  Inference complete!\n\n");

    // ==================================================================
    // STEP 9: Read results and take action
    //
    // The output buffer now contains 10 float32 logits, one per class.
    // argmax gives us the predicted class.
    //
    // THIS IS WHERE YOUR APPLICATION LOGIC GOES.
    // ==================================================================
    printf("[Step 9] Results:\n\n");
    printf("  Class logits:\n");
    int best = 0;
    for (int i = 0; i < 10; i++) {
        printf("    [%d] %-12s  %+.4f", i, CLASSES[i], output_data[i]);
        if (output_data[i] > output_data[best]) best = i;
        if (i == best) printf("  <--");
        printf("\n");
    }

    printf("\n  *** Prediction: %s (class %d) ***\n\n", CLASSES[best], best);

    // ==================================================================
    // STEP 10: Cleanup — free all QNN resources
    //
    // Order matters: graph info → context → backend → dlclose
    // ==================================================================
    printf("[Step 10] Cleaning up ...\n");
    freeGraphsInfo(&graphsInfo, numGraphs);
    qnn.contextFree(context, /*profile=*/nullptr);
    qnn.backendFree(backend);

    free(input_data);
    free(output_data);

    dlclose(model_lib);
    dlclose(backend_lib);

    printf("  Done.\n");
    return 0;
}
