# Experiment 01: Hello HIP

This experiment introduces the fundamental pattern of GPU programming. Before diving into the code, let's establish the mental models that will guide your understanding throughout this course.

---

## The Host-Device Model

GPU programming operates on a fundamentally different model than traditional CPU programming. Understanding this distinction is crucial.

### Two Separate Processors, Two Separate Memories

```
┌─────────────────────────┐         ┌─────────────────────────┐
│         HOST            │         │        DEVICE           │
│         (CPU)           │         │        (GPU)            │
│                         │         │                         │
│  ┌─────────────────┐    │         │    ┌─────────────────┐  │
│  │   System RAM    │    │ PCIe    │    │   HBM / VRAM    │  │
│  │   (DDR4/DDR5)   │◄───┼─────────┼───►│   (192 GB)      │  │
│  │                 │    │         │    │                 │  │
│  └─────────────────┘    │         │    └─────────────────┘  │
│                         │         │                         │
│  Your main() runs here  │         │  Kernels run here       │
└─────────────────────────┘         └─────────────────────────┘
```

**Key insight**: The CPU and GPU have completely separate memory spaces. Data does not automatically transfer between them. This is why we need explicit `hipMemcpy` calls.

### Why Explicit Memory Management?

On a CPU, you might write:
```cpp
int* array = new int[1000];
process(array);  // Data is just... there
```

On a GPU, the same concept requires three steps:
```cpp
// 1. Allocate on device
int* d_array;
hipMalloc(&d_array, 1000 * sizeof(int));

// 2. Copy data to device
hipMemcpy(d_array, h_array, 1000 * sizeof(int), hipMemcpyHostToDevice);

// 3. Run computation
process_kernel<<<blocks, threads>>>(d_array);

// 4. Copy results back
hipMemcpy(h_array, d_array, 1000 * sizeof(int), hipMemcpyDeviceToHost);
```

This explicit management exists because:

1. **PCIe bandwidth is limited** — Moving data between host and device is slow (tens of GB/s) compared to GPU memory bandwidth (several TB/s). Making transfers explicit forces you to think about minimising them.

2. **GPU memory is fast but precious** — HBM has enormous bandwidth but limited capacity. Explicit allocation lets you control exactly what lives on the GPU.

3. **Asynchronous execution** — The GPU can work independently while the CPU does other things. Explicit transfers help manage synchronisation.

---

## Data Parallelism: The GPU's Superpower

CPUs excel at complex, sequential logic. GPUs excel at doing the same operation on millions of data points simultaneously.

### A Mental Model

Think of a CPU as a brilliant professor who can solve any problem but handles one student at a time. A GPU is like a thousand teaching assistants who can each handle simple problems simultaneously.

**CPU approach** (sequential):
```
for each pixel in image:
    apply_filter(pixel)    // One at a time
```

**GPU approach** (parallel):
```
launch 1,000,000 threads
each thread: apply_filter(my_pixel)   // All at once
```

### When GPUs Win

GPUs dramatically outperform CPUs when:
- The same operation applies to many data elements
- Operations are independent (no thread needs another thread's result)
- Data fits in GPU memory
- Enough parallelism exists to saturate the hardware

In this course, we'll eventually compute matrix multiplications where a single operation can keep 304 Compute Units busy with millions of floating-point operations per second.

---

## Thread Hierarchy: Grid, Blocks, Threads

HIP organises parallel execution in a three-level hierarchy:

```
┌─────────────────────────────────────────────────────────────┐
│                          GRID                                │
│  (All threads launched by one kernel call)                  │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Block 0    │  │  Block 1    │  │  Block 2    │   ...   │
│  │             │  │             │  │             │         │
│  │ ┌─┬─┬─┬─┐   │  │ ┌─┬─┬─┬─┐   │  │ ┌─┬─┬─┬─┐   │         │
│  │ │0│1│2│3│...│  │ │0│1│2│3│...│  │ │0│1│2│3│...│         │
│  │ └─┴─┴─┴─┘   │  │ └─┴─┴─┴─┘   │  │ └─┴─┴─┴─┘   │         │
│  │  Threads    │  │  Threads    │  │  Threads    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

**Grid**: The entire collection of threads. Size determined by your problem.

**Block**: A group of threads that can cooperate via shared memory (LDS). Limited to 1024 threads on MI300X.

**Thread**: A single execution unit. Identified by `threadIdx.x` within its block.

### Why This Hierarchy?

1. **Hardware mapping**: Blocks map to Compute Units. The GPU scheduler assigns blocks to available CUs.

2. **Synchronisation scope**: Threads within a block can synchronise (`__syncthreads()`). Threads in different blocks cannot.

3. **Resource sharing**: Threads in a block share fast LDS memory. Inter-block communication requires slow global memory.

---

## The Kernel Launch: What Actually Happens

When you write:
```cpp
my_kernel<<<num_blocks, threads_per_block>>>(args...);
```

Here's what occurs:

1. **Command queued**: The HIP runtime creates a command packet describing the kernel launch.

2. **Blocks scheduled**: The GPU's command processor assigns blocks to available Compute Units.

3. **Wavefronts created**: Each block is divided into wavefronts (groups of 64 threads). On MI300X with 256 threads per block, you get 4 wavefronts.

4. **Execution begins**: All threads in a wavefront execute the same instruction simultaneously (SIMD).

5. **Asynchronous return**: Control returns to the CPU immediately. The GPU works independently.

6. **Synchronisation**: `hipDeviceSynchronize()` blocks the CPU until all GPU work completes.

---

## Memory Transfer Mechanics

`hipMemcpy` is deceptively simple but involves significant hardware activity.

### What Happens During hipMemcpyHostToDevice

```
Host Memory                           Device Memory
    │                                      │
    ▼                                      │
┌─────────┐                                │
│  Data   │──► CPU reads data              │
└─────────┘         │                      │
                    ▼                      │
              ┌───────────┐                │
              │ DMA Engine│                │
              │           │                │
              └─────┬─────┘                │
                    │ PCIe Transfer        │
                    │ (16-32 GB/s)         │
                    ▼                      ▼
              ┌─────────────────────────────┐
              │         HBM Memory          │
              │   (up to 5.3 TB/s access)   │
              └─────────────────────────────┘
```

**Key numbers to remember**:
- PCIe 4.0 x16: ~32 GB/s theoretical, ~25 GB/s practical
- PCIe 5.0 x16: ~64 GB/s theoretical
- HBM3 on MI300X: ~5.3 TB/s

This 100× difference between transfer speed and compute speed is why we minimise data movement.

---

## Common Beginner Mistakes

### Mistake 1: Forgetting Bounds Checks

```cpp
// WRONG - threads beyond array size access invalid memory
__global__ void bad_kernel(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = 0;  // Crashes if tid >= n
}

// CORRECT - always check bounds
__global__ void good_kernel(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = 0;
    }
}
```

Why this matters: We often launch more threads than data elements (to fill wavefronts efficiently). Extra threads must do nothing.

### Mistake 2: Ignoring Error Codes

```cpp
// WRONG - error silently ignored
hipMalloc(&d_ptr, size);

// CORRECT - check every HIP call
hipError_t err = hipMalloc(&d_ptr, size);
if (err != hipSuccess) {
    printf("Allocation failed: %s\n", hipGetErrorString(err));
    exit(1);
}

// BETTER - use the HIP_CHECK macro from the code
HIP_CHECK(hipMalloc(&d_ptr, size));
```

GPU errors are silent by default. The `HIP_CHECK` macro catches them immediately.

### Mistake 3: Assuming Synchronous Execution

```cpp
// WRONG - reading results before kernel completes
my_kernel<<<blocks, threads>>>(d_output, d_input);
hipMemcpy(h_output, d_output, size, hipMemcpyDeviceToHost);  // Might get stale data!

// CORRECT - ensure kernel completion
my_kernel<<<blocks, threads>>>(d_output, d_input);
hipDeviceSynchronize();  // Wait for kernel
hipMemcpy(h_output, d_output, size, hipMemcpyDeviceToHost);

// ALSO CORRECT - hipMemcpy synchronises implicitly
my_kernel<<<blocks, threads>>>(d_output, d_input);
hipMemcpy(h_output, d_output, size, hipMemcpyDeviceToHost);  // This waits
```

Note: `hipMemcpy` is synchronous by default, so it implicitly waits. But explicit synchronisation makes intent clear.

---

## Exercises

After studying the code, try these modifications:

1. **Change the operation**: Modify the kernel to compute `input[tid] * 2 + 1` instead of squaring.

2. **Experiment with block sizes**: Try 64, 128, 256, 512 threads per block. Does performance change? Why?

3. **Add error injection**: Comment out the bounds check. What happens when `n` is not a multiple of block size?

4. **Measure transfer overhead**: Time the `hipMemcpy` calls separately from kernel execution. What percentage of total time is spent on transfers?

---

## What's Next

Experiment 02 explores wavefront execution—how those 64 threads execute in lockstep and why this matters for performance. The thread hierarchy you learned here becomes concrete when you see how wavefronts actually run.

---

## Quick Reference

| Function | Purpose |
|----------|---------|
| `hipMalloc(&ptr, size)` | Allocate device memory |
| `hipFree(ptr)` | Free device memory |
| `hipMemcpy(dst, src, size, direction)` | Copy between host/device |
| `hipDeviceSynchronize()` | Wait for all GPU work |
| `hipGetLastError()` | Check for kernel launch errors |
| `hipGetDeviceProperties(&props, device)` | Query device capabilities |

| Kernel Syntax | Meaning |
|---------------|---------|
| `__global__` | Function runs on GPU, called from CPU |
| `__device__` | Function runs on GPU, called from GPU |
| `__host__` | Function runs on CPU (default) |
| `<<<blocks, threads>>>` | Launch configuration |
