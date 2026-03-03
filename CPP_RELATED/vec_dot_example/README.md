# Vector Dot Product – FastRPC DSP Example

Minimal example to verify ARM-to-DSP communication via FastRPC on a Snapdragon board.
No external SDK dependencies on the DSP side (no QNN, no ONNX Runtime) – just pure C
computing a dot product.

## Architecture

```
ARM (Apps Processor)                  FastRPC                  DSP (Hexagon)
┌──────────────────────┐                                ┌─────────────────────┐
│  main.cpp            │                                │  vec_dot_imp.c      │
│                      │                                │                     │
│  1. Fill vec_a, vec_b│   ION buffers                  │  dot = Σ a[i]*b[i]  │
│  2. Call process()   │ ──────────────────────────────► │                     │
│  3. Print result     │ ◄────────────────────────────── │  return dot         │
│                      │   ION buffer (1 float)          │                     │
│  Links:              │                                │  Links:             │
│   vec_dot_stub.c     │                                │   vec_dot_skel.c    │
│   libadsprpc.so      │                                │                     │
│   librpcmem.so       │                                │                     │
└──────────────────────┘                                └─────────────────────┘
```

## Interface

| Function    | Description                                        |
|-------------|----------------------------------------------------|
| `init`      | Mark DSP side as ready                             |
| `deinit`    | Cleanup                                            |
| `set_param` | Set vector length or toggle HVX                    |
| `get_param` | Query vector length, HVX flag, or version          |
| `process`   | Compute dot product: takes vec_a + vec_b, returns 1 float |

## Files

```
vec_dot_example/
├── idl/
│   └── vec_dot.idl          # FastRPC interface definition
├── inc/
│   └── vec_dot.h            # Shared header (interface + param IDs)
├── dsp/
│   └── vec_dot_imp.c        # DSP implementation (pure C, no deps)
├── arm/
│   └── main.cpp             # ARM host application
├── build_dsp.sh             # Build libvec_dot_skel.so
├── build_arm.sh             # Build vec_dot_arm
└── README.md
```

## Build & Run

### 1. Set environment

```bash
export HEXAGON_SDK_ROOT=/path/to/hexagon/sdk
export HEXAGON_ARCH=v68          # v68 for SM8350+, v73 for SM8550+
export NDK_ROOT=/path/to/android-ndk
```

### 2. Build DSP library (generates stub/skel + compiles)

```bash
./build_dsp.sh
```

### 3. Build ARM binary

```bash
./build_arm.sh
```

### 4. Deploy to device

```bash
adb push build_dsp/libvec_dot_skel.so /vendor/lib/rfsa/dsp/
adb push build_arm/vec_dot_arm /data/local/tmp/
```

### 5. Run

```bash
adb shell '/data/local/tmp/vec_dot_arm 1024'
```

### Expected output

```
=== Vector Dot Product – FastRPC DSP Example ===
Vector length : 1024 floats (4096 bytes)

DSP reports vec_len = 1024
DSP implementation version = 1

====== Results ======
DSP result    : 524800
ARM result    : 524800
Expected      : 524800
Match         : YES
DSP time (us) : <varies>
=====================

Running 100 iterations for timing ...
Avg round-trip (FastRPC + DSP compute): <varies> us
```

The test uses `a = [1, 1, 1, ...]` and `b = [1, 2, 3, ..., N]`, so the
expected dot product is `N * (N+1) / 2`.

## Next Steps

- Replace the scalar loop in `vec_dot_imp.c` with HVX intrinsics for SIMD
- Toggle via `set_param(VEC_DOT_PARAM_USE_HVX, 1)`
- Compare timing between scalar and HVX paths
