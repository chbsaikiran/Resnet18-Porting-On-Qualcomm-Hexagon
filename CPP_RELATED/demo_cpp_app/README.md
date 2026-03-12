# Demo C++ App — Calling the .so Files Directly

This C++ application demonstrates **exactly how the `.so` files are
used**. It loads them via `dlopen()` at runtime and calls the QNN C API
to run ResNet18 inference.

## The Two .so Files

```
┌───────────────────────────────────────────────────┐
│  main.cpp  (THIS APPLICATION)                     │
│                                                   │
│  dlopen("libQnnCpu.so")           ← BACKEND       │
│    → QnnInterface_getProviders()                  │
│    → backendCreate()                              │
│    → contextCreate()                              │
│                                                   │
│  dlopen("libresnet18_cifar10.so") ← MODEL         │
│    → QnnModel_composeGraphs()                     │
│    → graphFinalize()                              │
│    → graphExecute(input, output)                  │
│                                                   │
│  Read output logits → argmax → "dog"              │
└───────────────────────────────────────────────────┘
```

## Build & Run

```bash
# Build
./build.sh

# Generate a test input (optional — one is included)
conda run -n mlpy310env python -c "
import pickle, numpy as np
with open('../../Training/data/cifar-10-batches-py/test_batch', 'rb') as f:
    batch = pickle.load(f, encoding='bytes')
img = batch[b'data'][42].reshape(3,32,32).astype(np.float32)/255.0
mean, std = [0.4914,0.4822,0.4465], [0.2470,0.2435,0.2616]
for c in range(3): img[c] = (img[c]-mean[c])/std[c]
img.transpose(1,2,0).tofile('test_inputs/img_0042.raw')
print('Ground truth: dog')
"

# Run inference
./build/cifar10_qnn_app test_inputs/img_0042.raw
```

## What the 10 Steps Do

| Step | QNN API Call | What it does |
|------|-------------|-------------|
| 1 | `dlopen("libQnnCpu.so")` | Load the backend engine |
| 1 | `QnnInterface_getProviders()` | Get the QNN function table |
| 2 | `backendCreate()` | Create a backend instance |
| 3 | `contextCreate()` | Create a resource context |
| 4 | `dlopen("libresnet18_cifar10.so")` | Load the model |
| 5 | `QnnModel_composeGraphs()` | Build the graph (all ops + weights) |
| 6 | `graphFinalize()` | Backend optimises for hardware |
| 7 | *(set up buffers)* | Prepare input/output data |
| 8 | `graphExecute()` | Run inference (the forward pass) |
| 9 | *(read output)* | argmax on logits → prediction |
| 10 | `contextFree()`, `backendFree()`, `dlclose()` | Cleanup |

## To run on DSP instead of CPU

Replace `libQnnCpu.so` with `libQnnDsp.so`:

```bash
./build/cifar10_qnn_app test_inputs/img_0042.raw /path/to/libQnnDsp.so
```

The application code stays exactly the same — only the backend `.so`
path changes. The QNN API abstracts away the hardware.
