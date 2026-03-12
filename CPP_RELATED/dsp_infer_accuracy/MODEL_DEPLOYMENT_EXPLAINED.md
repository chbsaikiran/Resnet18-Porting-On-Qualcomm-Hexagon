# Model Deployment Pipeline — From PyTorch to Qualcomm Hexagon DSP

## What Is This?

This document explains the **model deployment pipeline** used in this
project: taking a ResNet18 model trained in PyTorch on a PC and running
it on a Qualcomm Snapdragon 710 phone's CPU, GPU, or Hexagon DSP.

In the industry, this process is referred to as:

- **Model deployment** / **model porting** — the full chain from trained
  model to running on a target device
- **Model compilation** — translating a neural network from one
  representation to another, analogous to a source code compiler
- **Edge AI inference** / **on-device inference** — running ML models
  directly on edge/embedded hardware rather than in the cloud
- **Hardware-accelerated inference** — leveraging specialized processors
  (DSP, NPU, GPU) instead of a general-purpose CPU

---

## The Problem

You trained a ResNet18 model in **PyTorch** on your development machine.
PyTorch is a training framework designed for x86 CPUs and NVIDIA GPUs.
But the target device is a **Qualcomm Snapdragon 710 phone** with:

| Processor | Type | Use case |
|-----------|------|----------|
| ARM Cortex-A75/A55 | CPU | General purpose |
| Adreno 616 | GPU | Graphics / parallel compute |
| Hexagon 685 (V65) | DSP | Sustained, power-efficient compute |

PyTorch cannot run on these processors directly. So the question is:
**how do you take a model trained on one platform and run it on
completely different hardware?**

The answer is a multi-stage compilation pipeline.

---

## The Pipeline

```
Training World                    Deployment World
(PC, Python, NVIDIA GPU)          (phone, C/C++, Qualcomm HW)

┌──────────┐    export    ┌──────┐   convert    ┌──────────┐   compile   ┌─────────────┐
│ PyTorch  │ ──────────→  │ ONNX │ ──────────→  │ QNN C++  │ ────────→  │ model.so    │
│ model    │              │ .onnx│              │ .cpp+.bin│            │ (ARM64 ELF) │
└──────────┘              └──────┘              └──────────┘            └──────┬──────┘
                                                                              │
                                                                    adb push  │
                                                                              ▼
                                                                    ┌─────────────────┐
                                                                    │ qnn-net-run      │
                                                                    │ + backend .so    │
                                                                    │ (CPU/GPU/DSP)    │
                                                                    │                  │
                                                                    │ input.raw → model│
                                                                    │ → output logits  │
                                                                    └─────────────────┘
```

Each stage is explained below.

---

## Stage 1: PyTorch → ONNX (Model Export)

**Command:**
```bash
python onnx_dump.py
```

**Produces:** `resnet18_cifar10.onnx`

**Technical term:** **Model export** or **model serialization**

ONNX (Open Neural Network Exchange) is a **vendor-neutral intermediate
representation** for neural networks. Think of it like LLVM IR for
compiled programming languages. It captures two things:

1. **Graph structure** — the sequence of operations (Conv2D, BatchNorm,
   ReLU, AvgPool, Linear, etc.) and how they connect
2. **Trained weights** — the learned parameters (convolution kernels,
   biases, batch norm statistics, etc.)

All of this is stored in a single portable `.onnx` file that any
framework or vendor tool can read.

**Why this step is needed:** PyTorch's internal representation is
Python-specific and dynamic (it uses eager execution). ONNX gives us a
static, portable computation graph that Qualcomm's tools can consume.
Without ONNX, there would be no bridge between the training world
(PyTorch/Python) and the deployment world (QNN/C++).

---

## Stage 2: ONNX → QNN C++ Source (Graph Compilation — Front End)

**Command:**
```bash
qnn-onnx-converter \
    --input_network resnet18_cifar10.onnx \
    --output_path resnet18_cifar10
```

**Produces:** `resnet18_cifar10.cpp` + `resnet18_cifar10.bin`

**Technical term:** **Graph compilation** (front-end stage), or
**model conversion**

The `qnn-onnx-converter` reads the ONNX graph and translates every
operation into QNN's own operator representation:

| ONNX operation | QNN equivalent |
|---------------|----------------|
| `Conv` | `QNN_OP_CONV_2D` |
| `BatchNormalization` | (fused into conv) |
| `Relu` | `QNN_OP_RELU` |
| `AveragePool` | `QNN_OP_POOL_AVG_2D` |
| `Gemm` (fully connected) | `QNN_OP_FULLY_CONNECTED` |
| `Add` (residual) | `QNN_OP_ELEMENT_WISE_ADD` |

The output `.cpp` file contains the **graph topology** — which ops
connect to which, their parameters (kernel sizes, strides, padding),
and references to weight data. The `.bin` file contains the **serialized
weights** (all the learned floating-point numbers).

**Graph optimizations** also happen at this stage:
- **Operator fusion** — e.g., Conv + BatchNorm + ReLU fused into a
  single optimized kernel
- **Layout transformation** — NCHW (PyTorch's default: batch, channels,
  height, width) converted to NHWC (batch, height, width, channels),
  which is more efficient for mobile hardware
- **Constant folding** — pre-computing operations on constant tensors

**Why this step is needed:** QNN doesn't understand ONNX. It has its own
runtime API and operator set. This step is the "translation" from the
portable intermediate format to the vendor-specific format. It's
analogous to a compiler's front end translating C source into an
internal AST.

---

## Stage 3: QNN C++ → Model Shared Library (Compilation — Back End)

**Command:**
```bash
qnn-model-lib-generator \
    -c resnet18_cifar10.cpp \
    -b resnet18_cifar10.bin \
    -t aarch64-android
```

**Produces:** `lib/aarch64-android/libresnet18_cifar10.so`

**Technical term:** **Model library compilation** (back-end stage),
or **cross-compilation**

This takes the QNN graph definition (C++ source code) and compiles it
into a native **shared library** (`.so`) using the Android NDK's
cross-compiler. The result is a standard ELF shared library targeting
the `aarch64` (ARM 64-bit) architecture.

When `qnn-net-run` runs on the phone, it calls `dlopen()` on this `.so`
file — the same mechanism used for loading any shared library on
Linux/Android. The library exposes functions that return the complete
graph definition and weights to the QNN runtime.

**Why this step is needed:** The `.cpp` file is source code — it can't
be loaded and executed directly at runtime. Compiling it into a `.so`
produces a deployable binary artifact, like a plugin that the inference
runner can dynamically load. The cross-compilation (x86 host → ARM
target) is necessary because the phone runs ARM, not x86.

---

## Stage 4: On-Device Execution (Inference)

**Command (on device via adb):**
```bash
qnn-net-run \
    --model libresnet18_cifar10.so \
    --backend libQnnCpu.so \      # or libQnnDsp.so / libQnnGpu.so
    --input_list input_list.txt \
    --output_dir outputs/
```

**Technical term:** **On-device inference** or **edge inference**

`qnn-net-run` is Qualcomm's reference inference runner. When it
executes, it goes through four phases:

### Phase 1: Load Model
The model `.so` is loaded via `dlopen()`. The QNN runtime calls into the
library to retrieve the graph definition — all the operators, their
connections, and weights.

### Phase 2: Compose Graph
```
Composing Graphs
```
The runtime instantiates all operators for the chosen **backend**. Each
backend (CPU, GPU, DSP) has its own implementations of the operators.
For example, a `Conv2D` on CPU uses NEON SIMD instructions, on GPU uses
OpenCL compute shaders, and on DSP uses HVX vector instructions.

### Phase 3: Finalize Graph
```
Finalizing Graphs
```
The backend performs **hardware-specific optimizations**:
- **Memory planning** — allocating and reusing intermediate buffers
  efficiently
- **Tiling** (DSP) — breaking large tensors into tiles that fit in the
  DSP's local memory (TCM)
- **Kernel selection** (GPU) — choosing optimal compute shader
  configurations

### Phase 4: Execute
```
Executing Graphs
Finished Executing Graphs
```
For each input image:
1. Read the `.raw` file (float32 tensor, 32x32x3 for CIFAR-10)
2. Feed it through the compiled graph
3. Write the output logits (10 floats, one per CIFAR-10 class) to a
   `.raw` file

The output logits are then compared against ground-truth labels by
`compute_accuracy.py` to measure top-1 accuracy.

---

## Why Do All This?

### Latency
Running inference on-device eliminates the network round-trip to a
cloud server. On-device inference can run in milliseconds vs. hundreds
of milliseconds for a cloud API call.

### Privacy
User data (images, audio, text) never leaves the phone. The model runs
locally. This is critical for applications like face recognition,
health monitoring, and voice assistants.

### Offline Capability
The model works without an internet connection. Once deployed, it's
fully self-contained on the device.

### Power Efficiency
The Hexagon DSP is specifically designed for sustained, power-efficient
compute. It can run inference at a fraction of the power cost of the
main CPU or GPU, which is crucial for always-on workloads like:
- Wake-word detection ("Hey Google", "Hey Siri")
- Camera scene detection
- Sensor fusion
- Background audio processing

### Cost
No cloud compute costs. Once the model is deployed, inference is free.

---

## Analogy to Traditional Software Compilation

The model deployment pipeline is strikingly similar to a traditional
compiler toolchain:

| Compiler concept | Model deployment equivalent |
|-----------------|---------------------------|
| Source code (C/C++) | Trained PyTorch model |
| Intermediate representation (LLVM IR) | ONNX (.onnx) |
| Target-specific code generation | qnn-onnx-converter |
| Object file (.o) | QNN C++ source (.cpp + .bin) |
| Shared library (.so) | Model library (libresnet18_cifar10.so) |
| Cross-compiler (x86 → ARM) | NDK + qnn-model-lib-generator |
| Runtime / dynamic loader | qnn-net-run + backend .so |
| Execution | Inference (input → model → output) |

Just as a C compiler transforms human-readable source code into machine
instructions optimised for a specific CPU architecture, the model
deployment pipeline transforms a neural network graph into an optimised
binary that runs efficiently on specific hardware accelerators.

---

## Tools Used in This Project

| Tool | Source | Role |
|------|--------|------|
| PyTorch | Meta (Facebook) | Model training |
| ONNX | Linux Foundation (joint project) | Portable model format |
| qnn-onnx-converter | Qualcomm (QAIRT SDK) | ONNX → QNN translation |
| qnn-model-lib-generator | Qualcomm (QAIRT SDK) | QNN C++ → .so compilation |
| qnn-net-run | Qualcomm (QAIRT SDK) | On-device inference runner |
| libQnnCpu.so | Qualcomm (QAIRT SDK) | CPU backend |
| libQnnGpu.so | Qualcomm (QAIRT SDK) | GPU backend (Adreno OpenCL) |
| libQnnDsp.so | Qualcomm (QAIRT SDK) | DSP backend (Hexagon) |
| Android NDK | Google | ARM cross-compiler |
| Hexagon SDK | Qualcomm | DSP tools + test signatures |
