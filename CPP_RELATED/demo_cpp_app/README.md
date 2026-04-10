# Demo C++ App — QNN (host) and SNPE (on-device)

This folder contains two demos:

| Program | Stack | Where it runs |
|---------|--------|----------------|
| `cifar10_qnn_app` | QNN C API + `dlopen` | Linux x86_64 (host) — `./build.sh` |
| `cifar10_snpe_demo` | SNPE C++ API | Android aarch64 — `./build_android_snpe.sh` |

---

## Part A — QNN host demo (`main.cpp`)

Demonstrates **how the QNN `.so` files are wired**: load backend + model with `dlopen()`, then `QnnModel_composeGraphs` → `graphFinalize` → `graphExecute`.

### The two `.so` files

```
┌───────────────────────────────────────────────────┐
│  main.cpp                                         │
│  dlopen("libQnnCpu.so")      → backend            │
│  dlopen("libresnet18_cifar10.so") → model         │
│  graphExecute → argmax → class name               │
└───────────────────────────────────────────────────┘
```

### Build and run (host)

```bash
./build.sh
./build/cifar10_qnn_app <input_nhwc.raw> [backend.so] [model.so]
```

Input for this path is **float32 NHWC** `1×32×32×3` (12288 bytes).

DSP/GPU on **device** use the QNN scripts under `../dsp_infer_accuracy/` (`qnn-net-run`), not this host binary.

---

## Part B — SNPE on-device demo (`main_snpe.cpp`)

Loads **`resnet18_cifar10_quantized.dlc`** with the SNPE C++ API (`IDlContainer`, `SNPEBuilder`, `execute`). You can choose **DSP**, **GPU**, or **CPU** at the command line (same runtime idea as `snpe-net-run --use_dsp` / `--use_gpu`).

### Input layout

Matches `prepare_cifar10_inputs_snpe.py` in `../dsp_infer_accuracy/`: **float32 NCHW** `1×3×32×32` = **12288 bytes** per image.

### 1. Build (cross-compile)

Requires **Android NDK** and **SNPE** with `lib/aarch64-android/libSNPE.so` and C++ headers under `${SNPE_ROOT}/include/SNPE` (SNPE 2.10 layout).

```bash
export NDK_ROOT=/path/to/android-ndk-r27d    # optional; script has a default
export SNPE_ROOT=/opt/qcom/aistack/snpe/2.10.40.4
./build_android_snpe.sh
```

Output: `build_android/cifar10_snpe_demo`

### 2. Deploy to phone

Pushes the binary, `libSNPE.so`, DSP stubs/skel (and optional RFSA + test signature, same idea as `dsp_infer_accuracy/deploy_snpe.sh`).  
**DLC** is taken read-only from:

`../dsp_infer_accuracy/resnet18_cifar10_quantized.dlc`

```bash
./deploy_demo_snpe.sh
```

Device directory: `/data/local/tmp/demo_snpe_app/`

### 3. Run on device

Prepares **one** CIFAR-10 image using the **existing** script in `dsp_infer_accuracy` (this repo does not modify that folder), pushes it, then runs the demo:

```bash
./run_demo_snpe.sh          # default runtime: dsp
./run_demo_snpe.sh cpu
./run_demo_snpe.sh gpu
./run_demo_snpe.sh dsp
```

Manual invocation on device:

```text
cifar10_snpe_demo <input_nchw.raw> [dsp|gpu|cpu] [path_to.dlc]
```

Default DLC path inside the binary: `/data/local/tmp/demo_snpe_app/resnet18_cifar10_quantized.dlc`

### Environment variables (deploy)

| Variable | Purpose |
|----------|---------|
| `SNPE_ROOT` | SNPE SDK root |
| `HEXAGON_SDK_ROOT` | For `signer.py` (DSP test signature) |
| `RFSA_BASE` / `RFSA_DOMAIN` / `RFSA_TARGET_DIR` | Optional RFSA layout overrides (same pattern as `deploy_snpe.sh`) |

---

## QNN step reference (host app)

| Step | QNN API | Role |
|------|---------|------|
| 1 | `dlopen` + `QnnInterface_getProviders` | Load backend |
| 2–3 | `backendCreate`, `contextCreate` | Instances |
| 4–5 | Model `.so` + `QnnModel_composeGraphs` | Graph |
| 6 | `graphFinalize` | Optimize |
| 8 | `graphExecute` | Inference |

---

## SNPE runtime mapping

| `run_demo_snpe.sh` / argv | SNPE |
|---------------------------|------|
| `dsp` | `Runtime_t::DSP` (Hexagon) |
| `gpu` | `Runtime_t::GPU` (Adreno) |
| `cpu` | `Runtime_t::CPU` |

DSP requires the same style of **skel libraries + library path + test signature** as your working `dsp_infer_accuracy` SNPE flow; `deploy_demo_snpe.sh` mirrors that layout under `/data/local/tmp/demo_snpe_app`.
