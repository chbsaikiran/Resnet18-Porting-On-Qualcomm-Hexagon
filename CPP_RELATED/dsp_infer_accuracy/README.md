# dsp_infer_accuracy — ResNet18 CIFAR-10 Inference on Qualcomm Hexagon DSP

End-to-end pipeline for running a PyTorch-trained ResNet18 model on a
Qualcomm Snapdragon device using QNN (Qualcomm AI Engine Direct).
Supports **CPU**, **GPU**, and **DSP** backends via `qnn-net-run`.

## Prerequisites

| Tool | Version used | Purpose |
|------|-------------|---------|
| QAIRT SDK | 2.31.0 | QNN converter, model-lib-generator, qnn-net-run |
| Android NDK | r27d | Cross-compile ARM host (only for FastRPC path) |
| Hexagon SDK | 6.5.0.0 | DSP tools, test signature generation |
| Python (conda) | 3.10 (`mlpy310env`) | ONNX export, input preparation, accuracy |
| ADB | — | Device communication |

Python packages (install into `mlpy310env`):
```bash
conda run -n mlpy310env pip install numpy==1.26.4 pandas pytz onnx onnxruntime onnxsim
```

> **numpy must be < 2.0** — the qnn-onnx-converter has a compatibility
> issue with numpy 2.x that causes shape inference failures.

## Target Device

- **SoC:** Snapdragon 710 (SDM710)
- **DSP:** Hexagon 685 (V65)
- **QAIRT DSP backend:** V66 (closest available; works via backward compat)

---

## Quick Start (CPU backend — simplest)

If you just want to verify the model works on the device:

```bash
# 1. Convert ONNX → QNN model library  (one time)
#    (see "Step-by-Step: Model Conversion" below)

# 2. Deploy to device
./deploy_qnn.sh

# 3. Run inference (CPU, 100 images)
./run_qnn_infer.sh 100 libQnnCpu.so
```

---

## Step-by-Step: Model Conversion (ONNX → QNN model .so)

This is done once on the host machine. The output is a shared library
that `qnn-net-run` loads on the device.

### 1. Export PyTorch → ONNX (if not already done)

```bash
cd ../../Training
conda run -n mlpy310env python onnx_dump.py
# produces: resnet18_cifar10.onnx
```

### 2. Convert ONNX → QNN C++ source

```bash
export QAIRT_ROOT=/opt/qcom/aistack/qairt/2.31.0.250130
export PATH="${QAIRT_ROOT}/bin/x86_64-linux-clang:$PATH"

conda run -n mlpy310env qnn-onnx-converter \
    --input_network ../../Training/resnet18_cifar10.onnx \
    --output_path resnet18_cifar10
```

This produces `resnet18_cifar10.cpp`, `resnet18_cifar10.bin`, and a
`resnet18_cifar10_net.json` graph description.

### 3. Compile QNN source → model shared library

```bash
export PATH="/home/saikiran/NDK/android-ndk-r27d:$PATH"   # for ndk-build

qnn-model-lib-generator \
    -c resnet18_cifar10.cpp \
    -b resnet18_cifar10.bin \
    -t aarch64-android
```

Output: `lib/aarch64-android/libresnet18_cifar10.so`

---

## Running on CPU

### 1. Deploy

```bash
./deploy_qnn.sh
```

Pushes to `/data/local/tmp/qnn_resnet18/` on the device:
- `qnn-net-run` (inference runner)
- `libresnet18_cifar10.so` (model)
- `libQnnCpu.so`, `libQnnGpu.so`, `libQnnDsp.so` (backends)
- DSP skel libraries (`libQnnDspV66Skel.so`, `libQnnDspV66.so`)
- `libc++_shared.so` (NDK runtime)
- Test signature for DSP (auto-generated)

### 2. Run inference

```bash
./run_qnn_infer.sh 100 libQnnCpu.so
```

Arguments:
- `100` — number of CIFAR-10 test images to evaluate
- `libQnnCpu.so` — use the CPU backend

### What the script does

1. **Prepares inputs** — reads CIFAR-10 test data (pickle or binary
   format), normalises with CIFAR-10 mean/std, writes float32 HWC
   `.raw` files
2. **Pushes inputs** to device
3. **Runs `qnn-net-run`** on device with the CPU backend
4. **Pulls output** `.raw` files (10 float32 logits per image)
5. **Computes accuracy** — argmax vs ground truth labels

### Expected output

```
=== Step 3: Running QNN inference ===
  Backend: libQnnCpu.so
  Model:   libresnet18_cifar10.so
  Images:  100

Composing Graphs
Finalizing Graphs
Executing Graphs
Finished Executing Graphs
WARNING: qnn-net-run exited with code 139 (segfault during teardown is expected)

=== Step 5: Computing accuracy ===

====== CIFAR-10 QNN DSP Inference Results ======
Total images : 100
Correct      : XX
Top-1 Accuracy: XX.XX%
================================================
```

> **Note:** The segfault at exit is a known QNN runtime teardown bug.
> Inference completes successfully — output files are written before
> the crash.

---

## Running on DSP (Hexagon)

### 1. Deploy (same as CPU — includes DSP libraries)

```bash
./deploy_qnn.sh
```

The deploy script automatically:
- Pushes `libQnnDsp.so` + `libQnnDspV66Stub.so` (ARM-side)
- Pushes `libQnnDspV66Skel.so` + `libQnnDspV66.so` (DSP-side)
- Generates and pushes a device-specific test signature using the
  Hexagon SDK's `signer.py` (required for unsigned DSP code)

### 2. Run inference on DSP

```bash
./run_qnn_infer.sh 100 libQnnDsp.so
```

Or simply (DSP is the default backend):

```bash
./run_qnn_infer.sh 100
```

The script sets `ADSP_LIBRARY_PATH` and `CDSP_LIBRARY_PATH` so the
FastRPC loader on the device can find the DSP skel libraries.

### DSP-specific notes

- **V65 vs V66:** The SDM710 has Hexagon 685 (V65). QAIRT 2.31.0 ships
  V66 DSP skel libraries. These may work via backward compatibility.
  If DSP loading fails, fall back to CPU or GPU.
- **Test signature:** Required for loading unsigned DSP code on
  production devices. `deploy_qnn.sh` auto-generates this.
- **Library paths:** The DSP finds its skel libraries via
  `ADSP_LIBRARY_PATH` / `CDSP_LIBRARY_PATH`, both set to
  `/data/local/tmp/qnn_resnet18`.

---

## Running on GPU

```bash
./run_qnn_infer.sh 100 libQnnGpu.so
```

No additional setup beyond `deploy_qnn.sh`.

---

## All Backends at a Glance

| Command | Backend | Runs on |
|---------|---------|---------|
| `./run_qnn_infer.sh 100 libQnnCpu.so` | CPU | ARM Cortex-A75/A55 |
| `./run_qnn_infer.sh 100 libQnnGpu.so` | GPU | Adreno 616 |
| `./run_qnn_infer.sh 100 libQnnDsp.so` | DSP | Hexagon 685 (V65) |
| `./run_qnn_infer.sh 100` | DSP (default) | Hexagon 685 (V65) |

---

## Folder Structure

```
dsp_infer_accuracy/
├── idl/
│   └── cifar10_infer.idl              # FastRPC interface definition
├── inc/
│   └── cifar10_infer.h                # Shared C header
├── arm/
│   └── main.cpp                       # ARM host (FastRPC path)
├── dsp/
│   └── cifar10_infer_imp.c            # DSP skel (FastRPC path)
├── build_dsp.sh                       # Build DSP skel library
├── build_arm.sh                       # Build ARM host binary
├── deploy_qnn.sh                      # Push model + runtime to device
├── run_qnn_infer.sh                   # Run inference + compute accuracy
├── prepare_cifar10_inputs.py          # CIFAR-10 → float32 .raw files
├── compute_accuracy.py                # Compare QNN outputs vs labels
├── lib/aarch64-android/
│   └── libresnet18_cifar10.so         # Compiled QNN model (generated)
├── qnn_inputs/                        # Prepared input .raw files (generated)
├── qnn_outputs/                       # Pulled output .raw files (generated)
└── README.md
```

## FastRPC Path (alternative — for custom DSP code)

The `arm/main.cpp` + `dsp/cifar10_infer_imp.c` files implement a
**custom FastRPC service** for running inference directly on the DSP.
This is a more advanced path that gives full control over the DSP
execution but requires building both the ARM host and DSP skel.

### Build (FastRPC path)

```bash
# 1. Build DSP skeleton
export HEXAGON_SDK_ROOT=/home/saikiran/Qualcomm/Hexagon_SDK/6.5.0.0
./build_dsp.sh

# 2. Build ARM host
export NDK_ROOT=/home/saikiran/NDK/android-ndk-r27d
./build_arm.sh
```

### DSP Backend Selection (FastRPC path)

The DSP implementation (`dsp/cifar10_infer_imp.c`) supports three
compile-time backends:

| Flag | Runtime | Model format | Notes |
|------|---------|-------------|-------|
| (none) | Mock | N/A | Dummy logits; for testing FastRPC pipeline |
| `-DUSE_SNPE` | SNPE | `.dlc` | Snapdragon Neural Processing Engine |
| `-DUSE_QNN` | QNN | `.bin` | Qualcomm AI Engine Direct (context binary) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `test_batch.bin not found` | Run training first, or ensure `Training/data/cifar-10-batches-py/test_batch` exists |
| `Segmentation fault` after "Finished Executing Graphs" | Known QNN teardown bug — results are valid, script handles it |
| `ndk-build: not found` | Add NDK to PATH: `export PATH="/home/saikiran/NDK/android-ndk-r27d:$PATH"` |
| DSP backend crashes | V66 skel on V65 hardware — fall back to `libQnnCpu.so` or `libQnnGpu.so` |
| `numpy` errors in converter | Downgrade: `pip install numpy==1.26.4` |
| No `Result_*` dirs in output | Check `qnn-net-run` log output for errors during graph execution |
