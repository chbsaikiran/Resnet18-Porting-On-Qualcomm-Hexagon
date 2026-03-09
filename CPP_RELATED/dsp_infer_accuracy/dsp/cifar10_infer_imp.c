/*
 * cifar10_infer_imp.c
 *
 * DSP-side implementation of the cifar10_infer FastRPC interface.
 * Runs on the Hexagon DSP.
 *
 * Build output: libcifar10_infer_skel.so
 *   Push to: /vendor/lib/rfsa/dsp/  (or /usr/lib/rfsa/adsp/)
 *
 * ---- Runtime integration ----
 *
 * This scaffold supports two backends selected at compile time:
 *
 *   1. SNPE (Snapdragon Neural Processing Engine)
 *      - Model format: .dlc  (converted from ONNX via snpe-onnx-to-dlc)
 *      - Compile with: -DUSE_SNPE  -I${SNPE_ROOT}/include/zdl
 *      - Link with:    -lSNPE
 *
 *   2. QNN (Qualcomm AI Engine Direct)
 *      - Model format: .bin context binary (compiled via qnn-context-binary-generator)
 *      - Compile with: -DUSE_QNN  -I${QNN_SDK_ROOT}/include/QNN
 *      - Link with:    -lQnnHtp  (or -lQnnDsp)
 *
 * Without either flag, a mock implementation is compiled that returns
 * deterministic dummy logits. This is useful for testing the FastRPC
 * pipeline end-to-end before integrating a real NN runtime.
 */

#include "cifar10_infer.h"
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Internal state                                                     */
/* ------------------------------------------------------------------ */

static int  g_initialised = 0;
static int  g_use_hvx     = 0;
static char g_model_path[512];

#define IMP_VERSION  1

/* ------------------------------------------------------------------ */
/*  Backend: SNPE                                                      */
/* ------------------------------------------------------------------ */
#if defined(USE_SNPE)

#include "DlSystem/DlError.h"
#include "DlSystem/RuntimeList.h"
#include "DlContainer/IDlContainer.h"
#include "SNPE/SNPE.h"
#include "SNPE/SNPEFactory.h"
#include "SNPE/SNPEBuilder.h"
#include "DlSystem/ITensorFactory.h"

static zdl::SNPE::SNPE *g_snpe = NULL;

static AEEResult backend_init(const char *path)
{
    zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::DSP;
    auto container = zdl::DlContainer::IDlContainer::open(path);
    if (!container) return AEE_EFAILED;

    zdl::SNPE::SNPEBuilder builder(container.get());
    builder.setRuntimeProcessor(runtime);
    builder.setUseUserSuppliedBuffers(false);
    g_snpe = builder.build().release();
    return g_snpe ? AEE_SUCCESS : AEE_EFAILED;
}

static AEEResult backend_infer(const float *input, float *output)
{
    auto inputTensor = zdl::SNPE::SNPEFactory::getTensorFactory()
        .createTensor({1, CIFAR10_NUM_CHANNELS, CIFAR10_IMG_SIZE, CIFAR10_IMG_SIZE});
    memcpy(inputTensor->begin().dataPointer(), input,
           CIFAR10_INPUT_FLOATS * sizeof(float));

    zdl::DlSystem::TensorMap outputMap;
    if (!g_snpe->execute(inputTensor.get(), outputMap))
        return AEE_EFAILED;

    auto it = outputMap.begin();
    auto outTensor = (*it).second;
    memcpy(output, outTensor->begin().dataPointer(),
           CIFAR10_NUM_CLASSES * sizeof(float));
    return AEE_SUCCESS;
}

static void backend_deinit(void)
{
    if (g_snpe) { delete g_snpe; g_snpe = NULL; }
}

/* ------------------------------------------------------------------ */
/*  Backend: QNN                                                       */
/* ------------------------------------------------------------------ */
#elif defined(USE_QNN)

#include "QnnInterface.h"
#include "HTP/QnnHtpDevice.h"

/*
 * QNN integration is device/SOC-specific.  The typical flow is:
 *
 *   1. QnnBackend_create()       → backend handle
 *   2. QnnDevice_create()        → device (HTP) handle
 *   3. QnnContext_createFromBinary() → load .bin context
 *   4. QnnGraph_retrieve()       → get the inference graph
 *   5. QnnGraph_execute()        → run inference
 *
 * Fill in below once your QNN SDK version and target SOC are known.
 */

static AEEResult backend_init(const char *path)
{
    /* TODO: Implement QNN context loading from path */
    (void)path;
    return AEE_EFAILED;
}

static AEEResult backend_infer(const float *input, float *output)
{
    /* TODO: Implement QnnGraph_execute() */
    (void)input; (void)output;
    return AEE_EFAILED;
}

static void backend_deinit(void)
{
    /* TODO: Implement QNN teardown */
}

/* ------------------------------------------------------------------ */
/*  Backend: Mock (default – for testing the FastRPC pipeline)         */
/* ------------------------------------------------------------------ */
#else

static AEEResult backend_init(const char *path)
{
    (void)path;
    return AEE_SUCCESS;
}

/*
 * Mock inference: compute a simple hash of the input pixels to
 * produce repeatable but non-trivial logits.  This lets you verify
 * the entire ARM ↔ DSP data path is working before plugging in
 * a real NN runtime.
 */
static AEEResult backend_infer(const float *input, float *output)
{
    int i, c;
    float sum = 0.0f;
    for (i = 0; i < CIFAR10_INPUT_FLOATS; i++)
        sum += input[i];

    for (c = 0; c < CIFAR10_NUM_CLASSES; c++)
        output[c] = sum * (float)(c + 1) * 0.01f;

    return AEE_SUCCESS;
}

static void backend_deinit(void) { }

#endif /* backend selection */


/* ================================================================== */
/*  FastRPC interface implementation                                   */
/* ================================================================== */

AEEResult cifar10_infer_init(const uint8_t *model_path,
                             int            model_path_len)
{
    if (!model_path || model_path_len <= 0)
        return AEE_EBADPARM;
    if (model_path_len > (int)sizeof(g_model_path) - 1)
        return AEE_EBADPARM;

    memcpy(g_model_path, model_path, model_path_len);
    g_model_path[model_path_len] = '\0';

    AEEResult rc = backend_init(g_model_path);
    if (rc == AEE_SUCCESS)
        g_initialised = 1;
    return rc;
}

AEEResult cifar10_infer_deinit(void)
{
    backend_deinit();
    g_initialised = 0;
    g_model_path[0] = '\0';
    return AEE_SUCCESS;
}

AEEResult cifar10_infer_set_param(int param_id, int value)
{
    switch (param_id) {
    case CIFAR10_PARAM_USE_HVX:
        g_use_hvx = (value != 0) ? 1 : 0;
        return AEE_SUCCESS;
    default:
        return AEE_EBADPARM;
    }
}

AEEResult cifar10_infer_get_param(int param_id, int *value)
{
    if (!value)
        return AEE_EBADPARM;

    switch (param_id) {
    case CIFAR10_PARAM_IMG_SIZE:    *value = CIFAR10_IMG_SIZE;    break;
    case CIFAR10_PARAM_NUM_CLASSES: *value = CIFAR10_NUM_CLASSES; break;
    case CIFAR10_PARAM_USE_HVX:    *value = g_use_hvx;           break;
    case CIFAR10_PARAM_VERSION:    *value = IMP_VERSION;          break;
    default:
        return AEE_EBADPARM;
    }
    return AEE_SUCCESS;
}

AEEResult cifar10_infer_process(const uint8_t *input_tensor,
                                int            input_tensor_len,
                                uint8_t       *output_logits,
                                int            output_logits_len,
                                int           *output_logits_lenout)
{
    if (!g_initialised)
        return AEE_ENOTINITIALIZED;
    if (!input_tensor || !output_logits || !output_logits_lenout)
        return AEE_EBADPARM;
    if (input_tensor_len < CIFAR10_INPUT_BYTES)
        return AEE_EBADPARM;
    if (output_logits_len < CIFAR10_OUTPUT_BYTES)
        return AEE_EBADPARM;

    const float *in_f  = (const float *)input_tensor;
    float        out_f[CIFAR10_NUM_CLASSES];

    AEEResult rc = backend_infer(in_f, out_f);
    if (rc != AEE_SUCCESS)
        return rc;

    memcpy(output_logits, out_f, CIFAR10_OUTPUT_BYTES);
    *output_logits_lenout = CIFAR10_OUTPUT_BYTES;

    return AEE_SUCCESS;
}
