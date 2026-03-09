# dsp_infer_accuracy — CIFAR-10 Accuracy with DSP Inference

Runs the full CIFAR-10 accuracy evaluation with **image loading and
preprocessing on ARM** and **neural network inference on the Hexagon DSP**
via Qualcomm FastRPC.

## Architecture

```
┌──────────────────────────────────────┐
│           ARM (Android)              │
│                                      │
│  1. Load test_batch.bin              │
│  2. Normalise pixels (CIFAR mean/std)│
│  3. For each image:                  │
│     ┌──── FastRPC ────┐              │
│     │  send CHW tensor │             │
│     │  to DSP          │             │
│     └──────────────────┘             │
│     ┌──── FastRPC ────┐              │
│     │  receive 10      │             │
│     │  logits back     │             │
│     └──────────────────┘             │
│  4. argmax → prediction              │
│  5. Accumulate accuracy              │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│          Hexagon DSP                 │
│                                      │
│  cifar10_infer_init()                │
│    → Load model (.dlc / .bin)        │
│                                      │
│  cifar10_infer_process()             │
│    → Run NN inference on tensor      │
│    → Return 10 float logits          │
│                                      │
│  cifar10_infer_deinit()              │
│    → Free model resources            │
└──────────────────────────────────────┘
```

## Folder Structure

```
dsp_infer_accuracy/
├── idl/
│   └── cifar10_infer.idl          # FastRPC interface definition
├── inc/
│   └── cifar10_infer.h            # Shared C header (manual qaic equivalent)
├── arm/
│   └── main.cpp                   # ARM host: image load + preprocess + accuracy
├── dsp/
│   └── cifar10_infer_imp.c        # DSP: model load + inference (SNPE / QNN / mock)
├── build_dsp.sh                   # Build libcifar10_infer_skel.so (Hexagon)
├── build_arm.sh                   # Build cifar10_dsp_accuracy (ARM64)
└── README.md
```

## Model Conversion (before deployment)

The DSP cannot run ONNX models directly. Convert first:

### Option A: SNPE (.dlc)

```bash
snpe-onnx-to-dlc \
    --input_network resnet18_cifar10.onnx \
    --output_path   resnet18_cifar10.dlc
```

### Option B: QNN (.bin context binary)

```bash
qnn-onnx-converter \
    --input_network resnet18_cifar10.onnx \
    --output_path   resnet18_cifar10.cpp

qnn-context-binary-generator \
    --model     resnet18_cifar10.cpp \
    --backend   libQnnHtp.so \
    --output_dir .
```

## Build

### 1. Build DSP skeleton (run first — generates the stub)

```bash
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK
./build_dsp.sh
```

Output: `build_dsp/libcifar10_infer_skel.so`

To use a real NN backend, uncomment the `EXTRA_DEFINES` / `EXTRA_LIBS`
lines in `build_dsp.sh` for SNPE or QNN.

### 2. Build ARM host

```bash
export NDK_ROOT=/path/to/android-ndk
./build_arm.sh
```

Output: `build_arm/cifar10_dsp_accuracy`

## Deploy & Run

```bash
# Push binaries
adb push build_arm/cifar10_dsp_accuracy      /data/local/tmp/
adb push build_dsp/libcifar10_infer_skel.so  /vendor/lib/rfsa/dsp/

# Push model and test data
adb push resnet18_cifar10.dlc                /data/local/tmp/
adb push test_batch.bin                      /data/local/tmp/

# Run
adb shell '/data/local/tmp/cifar10_dsp_accuracy \
    /data/local/tmp/resnet18_cifar10.dlc \
    /data/local/tmp/test_batch.bin'
```

## Expected Output

```
DSP backend version : 1
DSP num_classes     : 10
Model path          : /data/local/tmp/resnet18_cifar10.dlc

Test images loaded: 10000

[ 1000 / 10000]  running accuracy: XX.XX%
[ 2000 / 10000]  running accuracy: XX.XX%
...
[10000 / 10000]  running accuracy: XX.XX%

====== CIFAR-10 DSP Validation Results ======
Total images : 10000
Correct      : XXXX
Top-1 Accuracy: XX.XX%
Avg DSP infer : X.XXX ms  (includes FastRPC round-trip)
=============================================

Per-class accuracy:
  [0] airplane      XXX / 1000  (XX.XX%)
  [1] automobile    XXX / 1000  (XX.XX%)
  ...
```

## DSP Backend Selection

The DSP implementation (`dsp/cifar10_infer_imp.c`) supports three
compile-time backends:

| Flag | Runtime | Model format | Notes |
|---|---|---|---|
| (none) | Mock | N/A | Dummy logits; for testing FastRPC pipeline |
| `-DUSE_SNPE` | SNPE | `.dlc` | Snapdragon Neural Processing Engine |
| `-DUSE_QNN` | QNN | `.bin` | Qualcomm AI Engine Direct (HTP) |

The mock backend is compiled by default so you can verify the
end-to-end ARM ↔ DSP data flow before integrating a real NN runtime.
