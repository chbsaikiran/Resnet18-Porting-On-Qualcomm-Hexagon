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
#include "QnnTypes.h"
#include "QnnCommon.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "HTP/QnnHtpDevice.h"

#include <stdlib.h>
#include <stdio.h>

/*
 * These names must match the QNN graph produced by qnn-onnx-converter.
 * Override at compile time if your model uses different names, e.g.
 *   -DQNN_GRAPH_NAME=\"my_graph\"
 */
#ifndef QNN_GRAPH_NAME
#define QNN_GRAPH_NAME  "resnet18_cifar10"
#endif
#ifndef QNN_INPUT_NAME
#define QNN_INPUT_NAME  "input"
#endif
#ifndef QNN_OUTPUT_NAME
#define QNN_OUTPUT_NAME "output"
#endif

static QNN_INTERFACE_VER_TYPE  g_qnn;
static Qnn_BackendHandle_t    g_backend = NULL;
static Qnn_DeviceHandle_t     g_device  = NULL;
static Qnn_ContextHandle_t    g_context = NULL;
static Qnn_GraphHandle_t      g_graph   = NULL;
static uint8_t               *g_ctx_buf = NULL;

static AEEResult backend_init(const char *path)
{
    const QnnInterface_t **providers = NULL;
    uint32_t num_providers = 0;

    if (QnnInterface_getProviders(&providers, &num_providers) != QNN_SUCCESS
        || num_providers == 0)
        return AEE_EFAILED;

    g_qnn = providers[0]->QNN_INTERFACE_VER_NAME;

    if (g_qnn.backendCreate(NULL, NULL, &g_backend) != QNN_SUCCESS)
        return AEE_EFAILED;

    /* Read the .bin context binary from the device filesystem */
    FILE *fp = fopen(path, "rb");
    if (!fp) return AEE_EFAILED;

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (fsize <= 0) { fclose(fp); return AEE_EFAILED; }

    g_ctx_buf = (uint8_t *)malloc((size_t)fsize);
    if (!g_ctx_buf) { fclose(fp); return AEE_ENOMEMORY; }

    if ((long)fread(g_ctx_buf, 1, (size_t)fsize, fp) != fsize) {
        fclose(fp);
        free(g_ctx_buf);
        g_ctx_buf = NULL;
        return AEE_EFAILED;
    }
    fclose(fp);

    if (g_qnn.contextCreateFromBinary(g_backend, g_device, NULL,
                                       g_ctx_buf, (uint64_t)fsize,
                                       &g_context, NULL) != QNN_SUCCESS)
        return AEE_EFAILED;

    if (g_qnn.graphRetrieve(g_context, QNN_GRAPH_NAME,
                             &g_graph) != QNN_SUCCESS)
        return AEE_EFAILED;

    return AEE_SUCCESS;
}

static AEEResult backend_infer(const float *input, float *output)
{
    /*
     * The ARM host sends NCHW [1,3,32,32] but qnn-onnx-converter with
     * perform_axes_to_spatial_first_order=True produces an NHWC graph,
     * so the input tensor expects [1,32,32,3].  Transpose here.
     */
    float nhwc[CIFAR10_INPUT_FLOATS];
    int c, h, w;
    for (h = 0; h < CIFAR10_IMG_SIZE; h++)
        for (w = 0; w < CIFAR10_IMG_SIZE; w++)
            for (c = 0; c < CIFAR10_NUM_CHANNELS; c++)
                nhwc[(h * CIFAR10_IMG_SIZE + w) * CIFAR10_NUM_CHANNELS + c] =
                    input[c * CIFAR10_IMG_SIZE * CIFAR10_IMG_SIZE
                          + h * CIFAR10_IMG_SIZE + w];

    /* ---- input tensor ------------------------------------------------ */
    uint32_t in_dims[4] = {1, CIFAR10_IMG_SIZE, CIFAR10_IMG_SIZE,
                           CIFAR10_NUM_CHANNELS};
    Qnn_ClientBuffer_t in_buf  = {(void *)nhwc,
                                  CIFAR10_INPUT_FLOATS * sizeof(float)};
    Qnn_QuantizeParams_t no_qp = QNN_QUANTIZE_PARAMS_INIT;

    Qnn_Tensor_t in_tensor = QNN_TENSOR_INIT;
    QNN_TENSOR_SET_NAME(in_tensor, QNN_INPUT_NAME);
    QNN_TENSOR_SET_TYPE(in_tensor, QNN_TENSOR_TYPE_APP_WRITE);
    QNN_TENSOR_SET_DATA_FORMAT(in_tensor, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
    QNN_TENSOR_SET_DATA_TYPE(in_tensor, QNN_DATATYPE_FLOAT_32);
    QNN_TENSOR_SET_QUANT_PARAMS(in_tensor, no_qp);
    QNN_TENSOR_SET_RANK(in_tensor, 4);
    QNN_TENSOR_SET_DIMENSIONS(in_tensor, in_dims);
    QNN_TENSOR_SET_MEM_TYPE(in_tensor, QNN_TENSORMEMTYPE_RAW);
    QNN_TENSOR_SET_CLIENT_BUF(in_tensor, in_buf);

    /* ---- output tensor ----------------------------------------------- */
    uint32_t out_dims[2] = {1, CIFAR10_NUM_CLASSES};
    float    out_data[CIFAR10_NUM_CLASSES];
    Qnn_ClientBuffer_t out_buf = {(void *)out_data,
                                  CIFAR10_NUM_CLASSES * sizeof(float)};

    Qnn_Tensor_t out_tensor = QNN_TENSOR_INIT;
    QNN_TENSOR_SET_NAME(out_tensor, QNN_OUTPUT_NAME);
    QNN_TENSOR_SET_TYPE(out_tensor, QNN_TENSOR_TYPE_APP_READ);
    QNN_TENSOR_SET_DATA_FORMAT(out_tensor, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
    QNN_TENSOR_SET_DATA_TYPE(out_tensor, QNN_DATATYPE_FLOAT_32);
    QNN_TENSOR_SET_QUANT_PARAMS(out_tensor, no_qp);
    QNN_TENSOR_SET_RANK(out_tensor, 2);
    QNN_TENSOR_SET_DIMENSIONS(out_tensor, out_dims);
    QNN_TENSOR_SET_MEM_TYPE(out_tensor, QNN_TENSORMEMTYPE_RAW);
    QNN_TENSOR_SET_CLIENT_BUF(out_tensor, out_buf);

    /* ---- execute ----------------------------------------------------- */
    if (g_qnn.graphExecute(g_graph,
                            &in_tensor,  1,
                            &out_tensor, 1,
                            NULL, NULL) != QNN_SUCCESS)
        return AEE_EFAILED;

    memcpy(output, out_data, CIFAR10_NUM_CLASSES * sizeof(float));
    return AEE_SUCCESS;
}

static void backend_deinit(void)
{
    if (g_context) { g_qnn.contextFree(g_context, NULL); g_context = NULL; }
    if (g_device)  { g_qnn.deviceFree(g_device);         g_device  = NULL; }
    if (g_backend) { g_qnn.backendFree(g_backend);        g_backend = NULL; }
    if (g_ctx_buf) { free(g_ctx_buf);                     g_ctx_buf = NULL; }
    g_graph = NULL;
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
