# SNPE C++ Demo — Deep Explanation

> This document explains every decision made in `main_snpe.cpp`, `build_android_snpe.sh`,
> and `deploy_demo_snpe.sh`.  It is written to be useful when answering interview
> questions about running neural networks on Qualcomm hardware.

---

## Table of Contents

1. [What is SNPE and why use it?](#1-what-is-snpe-and-why-use-it)
2. [What is a DLC file?](#2-what-is-a-dlc-file)
3. [Architecture overview](#3-architecture-overview)
4. [The C++ code — walkthrough of main_snpe.cpp](#4-the-c-code--walkthrough-of-main_snpecpp)
5. [Cross-compilation — build_android_snpe.sh](#5-cross-compilation--build_android_snpesh)
6. [Deploying to device — what libraries and why](#6-deploying-to-device--what-libraries-and-why)
7. [Running on device — environment variables](#7-running-on-device--environment-variables)
8. [Obstacles encountered and how they were fixed](#8-obstacles-encountered-and-how-they-were-fixed)
9. [Key concepts for interviews](#9-key-concepts-for-interviews)

---

## 1. What is SNPE and why use it?

**SNPE (Snapdragon Neural Processing Engine)** is Qualcomm's SDK for running neural
networks on Snapdragon SoCs.  It can target three hardware backends on the same device:

| Runtime | Hardware | Best for |
|---------|----------|----------|
| CPU | Kryo ARM cores | Debugging / fallback |
| GPU | Adreno GPU | Float models, medium latency |
| DSP / HTP | Hexagon DSP | Quantized models, lowest power + latency |

The key advantage: **one model file (DLC), three runtimes, one API**.  You swap the
runtime string at build time or even at runtime — the same C++ code runs on all three.

SNPE exposes a **C++ API** (`zdl::` namespace) and a command-line tool
(`snpe-net-run`).  This demo uses the C++ API directly so you can embed inference
inside any Android application or daemon.

---

## 2. What is a DLC file?

A **DLC (Deep Learning Container)** is Qualcomm's model format, analogous to ONNX or
TFLite.  It stores:

- The graph structure (all layers, their types, connectivity).
- The trained weights (parameters).
- Optional quantization parameters (scale/offset per tensor) when the model is quantized.

**How the DLC in this project was created:**

```
PyTorch model  →  ONNX export  →  snpe-onnx-to-dlc  →  snpe-dlc-quantize  →  .dlc
```

The final file used here (`resnet18_cifar10_quantized.dlc`) is a **quantized** DLC:
the weights and activations are represented as INT8 instead of FP32.  This is the
standard format for efficient DSP execution.

---

## 3. Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│  Your App (cifar10_snpe_demo)                               │
│                                                             │
│  main_snpe.cpp                                              │
│    ├── IDlContainer::open()   ← loads the .dlc file        │
│    ├── SNPEBuilder::build()   ← picks the runtime          │
│    ├── createTensor()         ← allocates input buffer     │
│    └── snpe->execute()        ← runs the neural network    │
│                                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │  C++ API (libSNPE.so)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  SNPE Runtime Layer (libSNPE.so)                            │
│                                                             │
│  In QAIRT 2.31, libSNPE.so is a thin wrapper over QNN.      │
│  It internally calls dlopen() to load the correct backend:  │
│                                                             │
│    CPU  →  libQnnCpu.so                                     │
│    GPU  →  libQnnGpu.so                                     │
│    DSP  →  libQnnDsp.so + libSnpeDspV66Stub.so              │
│    HTP  →  libQnnHtp.so + libSnpeHtpPrepare.so + skel.so   │
│                                                             │
│  DLC parsing uses:  libQnnModelDlc.so                       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
              Hexagon DSP / Adreno GPU / ARM CPU
```

**Important**: The QNN backend libraries are NOT linked at compile time.  They are
loaded at runtime via `dlopen()` inside `libSNPE.so`.  This is why they must be
present in `LD_LIBRARY_PATH` on the device — the linker cannot find them for you.

---

## 4. The C++ code — walkthrough of main_snpe.cpp

### 4.1 Headers

```cpp
#include "DlContainer/IDlContainer.hpp"   // open() the .dlc file
#include "DlSystem/DlEnums.hpp"           // Runtime_t::CPU / GPU / DSP enum
#include "DlSystem/ITensorFactory.hpp"    // createTensor()
#include "DlSystem/PlatformConfig.hpp"    // DSP platform options
#include "DlSystem/StringList.hpp"        // list of tensor names
#include "DlSystem/TensorMap.hpp"         // map of name → ITensor*
#include "DlSystem/TensorShape.hpp"       // tensor dimensions
#include "SNPE/SNPE.hpp"                  // the SNPE inference object
#include "SNPE/SNPEBuilder.hpp"           // builder pattern for SNPE
#include "SNPE/SNPEFactory.hpp"           // factory: isRuntimeAvailable, getTensorFactory
```

All headers live under `${SNPE_ROOT}/include/SNPE/`.  The `zdl::` namespace prefix
comes from old SNPE (pre-2.x).  QAIRT 2.31 still ships these headers with the same
namespace for backwards compatibility, but the implementation underneath is now QNN.

### 4.2 Runtime selection

```cpp
static zdl::DlSystem::Runtime_t parse_runtime(const char *s) {
    if (strcasecmp(s, "cpu") == 0) return zdl::DlSystem::Runtime_t::CPU;
    if (strcasecmp(s, "gpu") == 0) return zdl::DlSystem::Runtime_t::GPU;
    if (strcasecmp(s, "dsp") == 0) return zdl::DlSystem::Runtime_t::DSP;
    ...
}
```

`Runtime_t` is an enum inside SNPE.  The three values we care about are CPU, GPU, and
DSP.  (HTP is a separate enum value for newer Hexagon Tensor Processors; DSP targets
the older v65/v66 Hexagon DSP.)

### 4.3 Initialize logging

```cpp
zdl::SNPE::SNPEFactory::initializeLogging(zdl::DlSystem::LogLevel_t::LOG_WARN);
```

**Why**: SNPE's internal errors are logged to logcat on Android and to stdout/stderr
on the terminal.  Without this call, silent failures like "backend not found" give you
nothing to debug with.  `LOG_WARN` is a good balance — it shows real problems without
the flood of `LOG_VERBOSE`.

**Must be called before any other SNPE API** — the SNPE docs say this explicitly.

### 4.4 Check if the runtime is available

```cpp
if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
    fprintf(stderr, "WARNING: runtime %s not available ...\n", ...);
}
```

**Why**: Not every device has every backend enabled.  For example:
- DSP requires a signed testsig or a root-enabled device.
- GPU requires OpenCL/OpenGL ES support.
- CPU always works.

If the runtime is unavailable, `SNPEBuilder::build()` may silently fall back to CPU
rather than crashing.  Checking here makes the behaviour explicit.

### 4.5 Open the DLC container

```cpp
auto container = zdl::DlContainer::IDlContainer::open(std::string(dlc_path));
```

`IDlContainer::open()` reads the `.dlc` file from disk.  In QAIRT 2.31 this call
internally uses `libQnnModelDlc.so` (loaded via `dlopen`) to parse the DLC format.
If that library is missing from `LD_LIBRARY_PATH` at runtime, this call returns null.

### 4.6 Build the SNPE instance

```cpp
zdl::SNPE::SNPEBuilder builder(container.get());

zdl::DlSystem::RuntimeList runtimeList;
runtimeList.add(runtime);
builder.setRuntimeProcessorOrder(runtimeList);    // <-- QAIRT 2.31 API
builder.setUseUserSuppliedBuffers(false);          // use ITensor (managed buffers)
```

**Why `RuntimeList` instead of a single runtime?**

In older SNPE (2.10 and below), the API was:
```cpp
builder.setRuntimeProcessor(runtime);   // single Runtime_t — OLD API, removed in 2.31
```

In QAIRT 2.31, this was replaced with:
```cpp
builder.setRuntimeProcessorOrder(runtimeList);   // priority list — NEW API
```

The reason for the change: the newer API allows you to specify a **fallback chain**.
For example `[DSP, CPU]` means "try DSP first, fall back to CPU if DSP is unavailable."
Even if you only have one runtime, you must pass a `RuntimeList`.

**Why `setUseUserSuppliedBuffers(false)`?**

SNPE has two buffer modes:
- `false` (managed): SNPE allocates `ITensor` objects.  You write your data into them
  via the iterator API.  Easier to use, slightly more copies.
- `true` (user-supplied): You allocate `UserBuffer` objects yourself, sharing memory
  with SNPE.  More efficient (zero-copy), but more complex API.

We use managed buffers (`false`) here for simplicity.

**DSP platform option:**

```cpp
if (runtime == DSP) {
    zdl::DlSystem::PlatformConfig cfg;
    cfg.setPlatformOptions("unsignedPD:OFF");
    builder.setPlatformConfig(cfg);
}
```

The Hexagon DSP has a security feature called **Signed Protection Domain (signedPD /
unsignedPD)**.  By default, newer Android kernels require DSP code to be signed.
`unsignedPD:OFF` means "allow unsigned protection domains", which is required when you
are running unsigned skeleton libraries (like `libSnpeDspV66Skel.so`).  Without this
option, the DSP runtime refuses to load on locked-down devices.

### 4.7 Build and introspect the network

```cpp
std::unique_ptr<zdl::SNPE::SNPE> snpe(builder.build());

zdl::DlSystem::StringList inputNames  = snpe->getInputTensorNames();
zdl::DlSystem::StringList outputNames = snpe->getOutputTensorNames();
```

`builder.build()` compiles the network for the target runtime.  After this:
- The graph is optimised for the hardware (operator fusion, memory layout, etc.).
- `getInputTensorNames()` returns the names of input tensors as they appear in the
  original model — in this case `"input"` (from the ONNX export of ResNet18).
- `getOutputTensorNames()` returns output names — `"output"` here.

**Why get the names dynamically?** Because the execute API requires you to pass inputs
by name (in a `TensorMap`).  Hardcoding `"input"` would break if the model is swapped.

### 4.8 Get the actual input shape from the network

```cpp
auto maybeShape = snpe->getInputDimensions(inputNames.at(0));
zdl::DlSystem::TensorShape inShape = *maybeShape;
```

**Why not hardcode `{1,3,32,32}`?**

When we ran the demo, the shape printed was `[1,32,32,3]` — **NHWC**, not NCHW.
Even though the ONNX model is NCHW, SNPE can reorder the axes internally during DLC
conversion.  If you create a tensor with the wrong shape, `execute()` will either fail
silently or produce garbage output.

`getInputDimensions()` returns the exact shape the compiled network expects.  Always
use this rather than assumptions from the original model.

### 4.9 Create the input tensor and copy data

```cpp
auto inputTensor =
    zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inShape);

std::copy(input_host, input_host + kInputBytes / sizeof(float),
          inputTensor->begin());
```

`createTensor()` allocates a managed float32 buffer of the correct shape.
`inputTensor->begin()` returns an iterator to the start of the float data.
`std::copy` fills it from the raw file buffer.

**Why `std::copy` instead of `memcpy`?**

`ITensor::begin()` returns an iterator object (`ITensorItr`), not a raw pointer.
Using `std::copy` works with iterators correctly.  Using `memcpy` on `begin()` would
require extracting the raw pointer (fragile and depends on iterator internals).

### 4.10 Execute inference

```cpp
zdl::DlSystem::TensorMap inputMap;
inputMap.add(inputNames.at(0), inputTensor.get());   // key = "input"

zdl::DlSystem::TensorMap outputMap;   // empty — SNPE fills this

if (!snpe->execute(inputMap, outputMap)) {
    fprintf(stderr, "ERROR: SNPE::execute failed\n");
    return 1;
}
```

**Why `TensorMap` instead of `execute(ITensor*, TensorMap&)`?**

SNPE has two `execute()` overloads:

```cpp
// Overload A — single unnamed input tensor
bool execute(const ITensor* input, TensorMap& output);

// Overload B — named input via TensorMap
bool execute(const TensorMap& input, TensorMap& output);
```

Overload A is a convenience shortcut: it assumes the tensor is the first input of the
network, identified by position.  During testing, overload A always returned `false`
(failure) while overload B succeeded.  The root cause is likely that in QAIRT 2.31,
the internal implementation of overload A calls `getHandle(*input)` on the `ITensor`
wrapper, and the handle resolution has a subtle issue with externally-created tensors.

Using overload B (TensorMap) with the explicit tensor name `"input"` is more robust and
is the recommended pattern in the SNPE 2.x documentation.

### 4.11 Read outputs

```cpp
zdl::DlSystem::StringList outNames = outputMap.getTensorNames();
zdl::DlSystem::ITensor *outTensor = outputMap.getTensor(outNames.at(0));
const float *logits = &(*outTensor->cbegin());
size_t out_count = outTensor->getSize();  // element count, NOT bytes
```

After `execute()`, SNPE fills `outputMap` with the result tensors.  We get the tensor
named `"output"` and read 10 float32 logits — one per CIFAR-10 class.

`getSize()` returns the **number of elements** (10), not bytes (40).

### 4.12 Argmax and display

```cpp
int best = 0;
for (size_t i = 0; i < ncls; i++) {
    if (logits[i] > logits[best]) best = i;
}
printf("*** Prediction: %s (class %d) ***\n", CLASSES[best], best);
```

The class with the highest logit is the network's prediction.  No softmax is needed
for argmax — softmax is a monotone transform and does not change which class wins.

---

## 5. Cross-compilation — build_android_snpe.sh

The phone runs **aarch64 Android** (64-bit ARM).  Your development machine is
**x86_64 Linux**.  You cannot run the same binary on both — you need a
**cross-compiler**.

### 5.1 The cross-compiler

```bash
CLANGXX="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang++"
```

This is the clang++ from the **Android NDK** (Native Development Kit).  The filename
encodes the target:
- `aarch64` — 64-bit ARM architecture
- `linux-android24` — Android API level 24 (Android 7.0+)

API level 24 was chosen because it is the lowest level that ships a complete enough
C++ standard library for SNPE.

### 5.2 Compiler flags explained

```bash
-std=c++17       # SNPE 2.x headers use C++17 features (optional<>, etc.)
-O2              # Optimise for speed — not debug, not maximum
-fPIC            # Position-independent code — required for shared libraries
-pie             # Position-independent executable — Android security requirement
-frtti           # Enable runtime type information — SNPE uses dynamic_cast internally
-DANDROID        # Preprocessor define that some SNPE headers check
-I"${SNPE_INC}"  # SNPE C++ headers path
-L"${SNPE_LIB}"  # Tell linker where libSNPE.so lives at link time
-lSNPE           # Link against libSNPE.so
-llog            # Android logging (liblog.so) — used by libSNPE internally
-landroid        # Android platform library
-ldl             # dlopen/dlsym — used by libSNPE to load QNN backends at runtime
-lc++_shared     # C++ standard library (libc++_shared.so) — shared, not static
```

**Why `-frtti`?** SNPE's RTTI usage (`dynamic_cast`, `typeid`) requires this flag.
Without it you get linker errors about `__dynamic_cast`.

**Why `-lc++_shared` not `-lc++_static`?** Using the shared C++ STL avoids duplicate
STL state (type_info, locale, etc.) when multiple `.so` files are loaded in the same
process.  Using the static STL in both your binary and `libSNPE.so` would cause
crashes or undefined behaviour.

### 5.3 The SNPE_ROOT issue

The build script originally had:
```bash
SNPE_ROOT="${SNPE_ROOT:-/opt/qcom/aistack/snpe/2.10.40.4}"  # OLD — wrong path
```

But the installed SDK is QAIRT 2.31, located at:
```bash
SNPE_ROOT="${SNPE_ROOT:-/opt/qcom/aistack/qairt/2.31.0.250130}"  # CORRECT
```

**QAIRT vs SNPE**: Starting from version 2.x, Qualcomm rebranded the package from
"SNPE SDK" to "QAIRT (Qualcomm AI Runtime) SDK".  The C++ SNPE API still exists inside
QAIRT — it is a compatibility layer over the newer QNN API.  The directory structure
changed from `/opt/qcom/aistack/snpe/` to `/opt/qcom/aistack/qairt/`.

---

## 6. Deploying to device — what libraries and why

Android devices do not have the SNPE SDK installed.  You must push everything your
binary needs to a directory on the device (here `/data/local/tmp/demo_snpe_app/`)
and set `LD_LIBRARY_PATH` to that directory before running.

Here is every file pushed and the reason it is needed:

### 6.1 The binary and the model

| File | Why needed |
|------|------------|
| `cifar10_snpe_demo` | Your compiled application |
| `resnet18_cifar10_quantized.dlc` | The neural network model (quantized ResNet18 CIFAR-10) |

### 6.2 C++ standard library

| File | Why needed |
|------|------------|
| `libc++_shared.so` | C++ STL — the binary was compiled with `-lc++_shared`. Android's system `libc++.so` can be a different version; providing your own ensures ABI compatibility. |

**Problem encountered**: QAIRT 2.31's `lib/aarch64-android/` directory does not ship
`libc++_shared.so`.  The fix was to pull it from the NDK itself:
```
${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so
```
This is guaranteed to match the version the binary was compiled against, since it is
the same NDK used to build the binary.

### 6.3 The SNPE runtime library

| File | Why needed |
|------|------------|
| `libSNPE.so` | The SNPE C++ runtime — all `zdl::` API calls resolve here |

### 6.4 QNN backend libraries (dlopen'd by libSNPE.so at runtime)

In QAIRT 2.31, `libSNPE.so` is a thin wrapper over **QNN (Qualcomm Neural Network
API)**.  When `SNPEBuilder::build()` is called, `libSNPE.so` internally calls
`dlopen()` to load the appropriate QNN backend library for the requested runtime.

This is the key architectural difference from older SNPE:

```
Old SNPE 1.x / 2.10:   libSNPE.so  contains  the CPU/GPU/DSP backend code
New QAIRT 2.31:         libSNPE.so  dlopen()s  libQnnCpu.so / libQnnGpu.so / ...
```

Because `dlopen()` searches `LD_LIBRARY_PATH`, all these libraries must be present on
the device.

| File | Runtime | Why needed |
|------|---------|------------|
| `libQnnModelDlc.so` | All | Parses the `.dlc` file format inside `IDlContainer::open()` |
| `libQnnSystem.so` | All | Core QNN system utilities (memory, threading) |
| `libQnnCpu.so` | CPU | CPU inference backend — all ARM NEON-optimised ops |
| `libQnnGpu.so` | GPU | Adreno GPU backend — OpenCL kernels |
| `libQnnDsp.so` | DSP | Older Hexagon DSP (v65/v66) backend |
| `libQnnHtp.so` | HTP | Newer Hexagon Tensor Processor backend (v68+) |

**How we discovered this**: The first deployment only pushed `libSNPE.so`.  Running the
binary gave `SNPE::execute failed` with no other error message.  Adding
`SNPEFactory::initializeLogging(LOG_WARN)` didn't produce extra output either.  After
checking `libSNPE.so`'s direct dependencies with `llvm-readelf -d`, we saw it only
linked against system libs (`libc`, `libdl`, `libEGL`, etc.) — meaning the QNN backend
libs must be dynamically loaded.  Adding them to the device fixed the execute failure.

### 6.5 DSP-specific skeleton libraries

| File | Why needed |
|------|------------|
| `libSnpeDspV66Stub.so` | ARM-side stub that communicates with the DSP via FastRPC |
| `libSnpeDspV66Skel.so` | DSP-side skeleton that runs inside the Hexagon core |
| `libSnpeHtpPrepare.so` | Prepares the HTP graph; needed even when targeting v66 DSP in some QAIRT versions |
| `libcalculator.so` | Auxillary math ops library for SNPE on DSP |

**Stub vs Skeleton architecture (FastRPC)**:

Qualcomm's Hexagon DSP runs in a separate memory domain from the main CPU.  Code does
not run directly on the DSP — instead:

```
ARM Application Processor                    Hexagon DSP
─────────────────────────                    ──────────────
libSnpeDspV66Stub.so  ←──── FastRPC ────→  libSnpeDspV66Skel.so
  (stub: marshals args,                      (skeleton: actual DSP
   makes RPC call)                            implementation)
```

`FastRPC` (Fast Remote Procedure Call) is a Qualcomm mechanism for calling code
running on the DSP from the ARM CPU.  The stub serialises your input data and the
skeleton deserialises and executes it.

The skeleton (`.so`) must be placed in the **RFSA (Remote File System Access)** path
that the DSP kernel driver reads from:
- `/vendor/lib/rfsa/adsp/` — for older Hexagon ADSP
- `/vendor/lib/rfsa/cdsp/` — for newer Hexagon CDSP

This requires root access to write to `/vendor`.  The deploy script attempts this
automatically with `adb root && adb remount`, and also keeps a copy in the app dir
as a fallback.

### 6.6 DSP test signature

```bash
python3 signer.py --serial <device_serial> --output snpe_testsig/
adb push testsig-*.so /data/local/tmp/demo_snpe_app/
```

Qualcomm's DSP security requires that unsigned code be authorised with a
**test signature** tied to the device's serial number.  The Hexagon SDK provides
`signer.py` which generates `testsig-XXXXXXXX.so` for a specific device.

Without this signature, any attempt to run DSP code on a non-rooted device returns
`CDSP-IPC-ERROR` or a similar DSP access error.

This is a development-only mechanism.  Production apps use proper signed protection
domains.

---

## 7. Running on device — environment variables

```bash
export LD_LIBRARY_PATH=/data/local/tmp/demo_snpe_app
export ADSP_LIBRARY_PATH="/data/local/tmp/demo_snpe_app;/system/lib/rfsa/adsp;/vendor/lib/rfsa/adsp"
export CDSP_LIBRARY_PATH="/data/local/tmp/demo_snpe_app;/system/lib/rfsa/adsp;/vendor/lib/rfsa/adsp"

./cifar10_snpe_demo /data/local/tmp/demo_snpe_app/inputs/img_00000.raw cpu
```

| Variable | Purpose |
|----------|---------|
| `LD_LIBRARY_PATH` | Where the Linux dynamic linker looks for `.so` files at `dlopen()` time.  Must contain all the QNN/SNPE libraries. |
| `ADSP_LIBRARY_PATH` | Where the **Hexagon ADSP** (`adsp` domain) looks for skeleton `.so` files.  Searched in order, semicolon-separated. |
| `CDSP_LIBRARY_PATH` | Same but for the **Hexagon CDSP** (`cdsp` domain, newer devices). |

If `ADSP_LIBRARY_PATH` or `CDSP_LIBRARY_PATH` is not set, the DSP driver falls back
to `/vendor/lib/rfsa/adsp` only.  Since that directory is read-only without root, you
need these variables to point the DSP to your writable device directory.

---

## 8. Obstacles encountered and how they were fixed

### Problem 1 — Build error: `setRuntimeProcessor` not found

```
error: no member named 'setRuntimeProcessor' in 'SNPE::SNPEBuilder';
       did you mean 'setRuntimeProcessorOrder'?
```

**Root cause**: The QAIRT 2.31 SDK removed `setRuntimeProcessor(Runtime_t)` and
replaced it with `setRuntimeProcessorOrder(RuntimeList)`.  The API change was not
documented prominently.

**Fix**: Replace the one-liner with:
```cpp
zdl::DlSystem::RuntimeList runtimeList;
runtimeList.add(runtime);
builder.setRuntimeProcessorOrder(runtimeList);
```

**Interview answer**: "The SNPE API changed between SDK versions.  I checked the
installed SDK headers directly (`grep -n "setRuntime" SNPEBuilder.hpp`), found the
new method, and updated the call.  The new API is actually better — it supports
runtime fallback chains like `[DSP, CPU]`."

---

### Problem 2 — Runtime crash: `libc++_shared.so not found`

```
CANNOT LINK EXECUTABLE: library "libc++_shared.so" not found
```

**Root cause**: The binary was compiled with `-lc++_shared` (shared C++ STL).
QAIRT 2.31 does not ship `libc++_shared.so` in its `lib/aarch64-android/` directory.
The deploy script had no fallback.

**Fix**: Check the NDK sysroot as a fallback source:
```bash
NDK_CXX_SHARED="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
```
This is guaranteed to match the version used during compilation (same NDK).

**Interview answer**: "The QAIRT 2.31 SDK dropped `libc++_shared.so` from its
distribution directory.  I found it in the NDK's sysroot — the same NDK used to build
the binary — ensuring version compatibility.  I updated the deploy script to check
both locations with a fallback."

---

### Problem 3 — Inference failure: `SNPE::execute failed` (silent)

This was the hardest problem.  Everything up to `build()` succeeded.  Execute failed
silently every time.

**Debugging steps taken:**

1. Added `SNPEFactory::initializeLogging(LOG_WARN)` — no extra output appeared.
2. Added `isRuntimeAvailable()` check — confirmed CPU was available.
3. Added `getInputTensorNames()` / `getOutputTensorNames()` print — confirmed network
   loaded correctly (input: `"input"`, output: `"output"`).
4. Checked `libSNPE.so` dependencies with `llvm-readelf -d` — only system libs.
   This revealed the QNN backends must be dlopen'd at runtime.
5. Pushed all QNN backend libs (`libQnnCpu.so`, `libQnnModelDlc.so`, etc.) — execute
   still failed.
6. Switched from `execute(ITensor*, TensorMap&)` to `execute(TensorMap&, TensorMap&)`
   — execute succeeded.

**Root cause**: In QAIRT 2.31, `execute(const ITensor* input, TensorMap& output)`
internally dereferences the `ITensor` C++ wrapper to get a raw C handle, but
externally-created tensors (via `createTensor`) may not be in a state the overload
expects.  The `TensorMap` overload identifies the tensor by name, which is unambiguous.

There were actually two combined causes:
- Missing QNN backend `.so` files (would have been fatal even with the right overload).
- Wrong `execute()` overload (fail-safe behaviour of the handle resolution).

**Fix**:
```cpp
// Instead of: snpe->execute(inputTensor.get(), outputMap)
zdl::DlSystem::TensorMap inputMap;
inputMap.add(inputNames.at(0), inputTensor.get());
snpe->execute(inputMap, outputMap);
```

**Interview answer**: "I debugged systematically: added logging, checked runtime
availability, printed tensor names, checked linker dependencies with readelf, and
finally isolated the execute call.  The combined fix was pushing the missing QNN
backend libraries and switching to the TensorMap execute overload, which is the
recommended API in SNPE 2.x documentation."

---

### Problem 4 — `getInputDimensions` revealed NHWC, not NCHW

After getting execute to work, we printed the actual input shape:

```
Input tensor shape: [1,32,32,3]   ← NHWC (Height, Width, Channels last)
```

The file comment said `NCHW [1,3,32,32]` but the DLC actually expected NHWC.  During
ONNX-to-DLC conversion, `snpe-onnx-to-dlc` can transpose the input layout.

The `kInputBytes` (12288) is the same regardless of layout, so inference still
produced correct results with the NHWC input — the pixels are the same bytes, just
the shape metadata differs.  But if you were doing pre-processing in C++ (normalising
per-channel), you would need the correct layout.

**Lesson**: Always call `getInputDimensions()` and print the shape instead of
assuming the model's original layout carries through to the DLC.

---

## 9. Key concepts for interviews

### What is the difference between SNPE and QNN?

SNPE is the older high-level SDK (C++ API, DLC format).  QNN (Qualcomm Neural Network
API) is the newer lower-level API (C API, flatbuffer format).  Starting with QAIRT
2.x, `libSNPE.so` became a wrapper over QNN internally.  Both are still supported;
SNPE is easier to use, QNN gives more control and better performance.

### Why cross-compile? Why not compile on the phone?

Android devices typically do not have a C++ compiler or SDK installed.  The NDK
provides a hosted cross-compiler that runs on your development machine and produces
binaries for Android/ARM.  The phone only needs the compiled binary and the runtime
`.so` files — not the compiler, headers, or build tools.

### What is `adb`?

Android Debug Bridge — a command-line tool that lets you:
- `adb push` — copy files from PC to device
- `adb shell` — run commands on device
- `adb logcat` — read device logs

It connects over USB (or WiFi in newer versions).  Everything in the deploy and run
scripts uses adb.

### What is LD_LIBRARY_PATH?

The Linux dynamic linker searches for `.so` files in a colon-separated list of
directories.  `LD_LIBRARY_PATH` is an environment variable that prepends directories
to that search path.  On Android, `adb shell` sessions do NOT inherit your host
environment, so you must set it explicitly before running the binary.

### Why does the DSP need a test signature?

The Hexagon DSP has a hardware security model: only signed code from Qualcomm or a
registered OEM can run.  During development, Qualcomm provides a signing tool
(`signer.py`) that generates a device-specific test certificate (`testsig-*.so`).
The DSP driver checks for this file at startup.  Without it, any DSP execution attempt
is rejected with an IPC error.  Production apps use proper code-signing certificates.

### What is FastRPC?

A Qualcomm mechanism for Remote Procedure Calls between the ARM CPU and the Hexagon
DSP, which run in separate memory protection domains.  The calling convention uses a
stub (ARM side) and a skeleton (DSP side).  The DSP kernel driver handles serialisation
and memory sharing.  This is why you need both a stub `.so` (ARM) and a skeleton `.so`
(DSP) for DSP inference.

### What is quantization and why does it matter for DSP?

Quantization represents model weights and activations as INT8 instead of FP32.  This:
- Reduces model size by 4x (from 4 bytes to 1 byte per value)
- Speeds up inference significantly on hardware with INT8 accelerators
- The Hexagon DSP is optimised for INT8 operations — FP32 inference on DSP is either
  unsupported or very slow

SNPE handles the FP32 → INT8 dequantization transparently when you pass FP32 input
to a quantized DLC.  The conversion uses the scale/offset parameters stored in the DLC.
