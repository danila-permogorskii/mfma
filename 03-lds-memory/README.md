# Experiment 03: LDS (Local Data Share) Memory

LDS is the key to high-performance GPU kernels. This fast, on-chip memory enables threads within a block to cooperate efficiently. Mastering LDS is prerequisite knowledge for implementing tiled matrix operations in later experiments.

---

## The Memory Hierarchy

GPU memory forms a hierarchy with dramatic differences in latency and bandwidth:

```
                          Latency        Bandwidth
                          (cycles)       (per CU)
    ┌─────────────┐
    │  Registers  │        ~0            Enormous
    │ (VGPR/AGPR) │                      (internal)
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │     LDS     │        20-30         ~3 TB/s
    │  (64 KB/CU) │                      (aggregate)
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  L1 Cache   │        ~50           ~12 TB/s
    │  (32 KB/CU) │                      (aggregate)
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  L2 Cache   │        100-200       ~6 TB/s
    │  (256 MB)   │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │    HBM      │        300+          ~5.3 TB/s
    │  (192 GB)   │                      (device total)
    └─────────────┘
```

### Key Insight

The performance gap between registers and HBM is over 1000×. LDS sits in the middle—10× slower than registers but 10× faster than global memory. Strategic use of LDS can dramatically accelerate memory-bound kernels.

---

## What Is LDS?

**Local Data Share** is a software-managed, on-chip memory:

- **64 KB per Compute Unit** on CDNA3
- **Shared among all threads in a block**
- **Not cached**—you control exactly what's there
- **Not persistent**—contents lost when block finishes

```
┌───────────────────────────────────────────────────────┐
│                    Compute Unit                       │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │                      LDS                        │ │
│  │                    64 KB                        │ │
│  │                                                 │ │
│  │  ┌───────────────────────────────────────────┐ │ │
│  │  │             Block 0 Data                  │ │ │
│  │  │       Visible to threads 0-255            │ │ │
│  │  └───────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐         │
│  │   WF 0    │  │   WF 1    │  │   WF 2    │   ...   │
│  │Lanes 0-63 │  │Lanes 0-63 │  │Lanes 0-63 │         │
│  └───────────┘  └───────────┘  └───────────┘         │
└───────────────────────────────────────────────────────┘
```

### Declaration Syntax

```cpp
__shared__ float shared_array[256];  // 256 floats = 1 KB
```

---

## The Cooperative Pattern

LDS enables a fundamental GPU programming pattern:

1. **Load** — Threads cooperatively load from global memory to LDS
2. **Sync** — Barrier ensures all loads complete
3. **Compute** — Threads read from LDS (possibly different locations)
4. **Sync** — Barrier ensures computation complete before next iteration

```cpp
__shared__ float tile[TILE_SIZE];

// Step 1: Each thread loads one element
tile[threadIdx.x] = global_data[blockIdx.x * TILE_SIZE + threadIdx.x];

// Step 2: Wait for all threads
__syncthreads();

// Step 3: Compute using potentially any element
float result = tile[some_other_index];

// Step 4: Sync before next iteration
__syncthreads();
```

### Why Synchronisation Is Critical

Without `__syncthreads()`, you have a **race condition**:

```cpp
// DANGEROUS - race condition!
tile[threadIdx.x] = input[tid];
output[tid] = tile[63 - threadIdx.x];  // Other thread might not have written yet!
```

The GPU executes wavefronts independently. Thread 0 might read `tile[63]` before thread 63 has written it.

---

## Bank Conflicts: The Hidden Performance Killer

LDS is organised into **32 banks**, each 4 bytes wide. Understanding bank conflicts is essential for performance.

### How Banks Work

```
LDS Address:     0    4    8   12   16   20   24   28   32   36  ...
Bank Number:     0    1    2    3    4    5    6    7    0    1  ...
                 └────────────────────────────────────────────────┘
                              32 banks × 4 bytes = 128 bytes
                              Pattern repeats every 128 bytes
```

Each bank can service **one request per cycle**. Multiple requests to the same bank serialise.

### Conflict Example

```cpp
__shared__ float data[64];  // 64 × 4 bytes

// Thread 0 accesses data[0]  → Bank 0
// Thread 1 accesses data[32] → Bank 0  (32 × 4 = 128 bytes apart)
// Thread 2 accesses data[64] → Bank 0  (would be, if array were larger)

// All hit Bank 0 → 3-way conflict → 3× slower!
```

### Conflict-Free Access

```cpp
// Stride-1 access: each thread hits different bank
data[threadIdx.x]  // Thread 0→Bank 0, Thread 1→Bank 1, Thread 2→Bank 2...

// Broadcast: all threads read same address (handled specially)
data[0]  // Hardware broadcasts to all lanes efficiently
```

### Visualising Conflicts

```
Stride-1 (No Conflicts):
Thread:  0  1  2  3  4  5  6  7  ...  31  32  33  ...
Address: 0  1  2  3  4  5  6  7  ...  31  32  33  ...
Bank:    0  1  2  3  4  5  6  7  ...  31   0   1  ...
         ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓       ✓   ✓   ✓

Stride-32 (Maximum Conflicts):
Thread:  0   1   2   3   4   ...
Address: 0  32  64  96  128 ...
Bank:    0   0   0   0   0  ...
         ✗ → All 64 threads hit Bank 0 = 64× slower!
```

---

## Matrix Transpose: A Classic LDS Problem

Transposing a matrix illustrates why LDS matters and how to avoid bank conflicts.

### Naive Transpose (Global Memory Only)

```cpp
// Reading: coalesced (row-major access)
// Writing: strided (column-major access) = SLOW

__global__ void naive_transpose(float* out, float* in, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    out[x * width + y] = in[y * width + x];  // Scattered writes!
}
```

### LDS-Optimised Transpose

```cpp
#define TILE 32

__global__ void lds_transpose(float* out, float* in, int width) {
    __shared__ float tile[TILE][TILE + 1];  // +1 avoids bank conflicts!
    
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    
    // Coalesced read from global → LDS
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    
    __syncthreads();
    
    // Transposed coordinates
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    
    // Coalesced write from LDS → global
    out[y * width + x] = tile[threadIdx.x][threadIdx.y];
}
```

### The "+1" Padding Trick

Without padding:
```
tile[0][0], tile[0][1], ..., tile[0][31]  → Banks 0-31
tile[1][0], tile[1][1], ..., tile[1][31]  → Banks 0-31 (repeats!)

Column access tile[0][0], tile[1][0], tile[2][0]...
All hit Bank 0 → CONFLICT!
```

With `tile[32][33]`:
```
tile[0][0..31] → Banks 0-31
tile[1][0..31] → Banks 1-32 (shifted by 1)
tile[2][0..31] → Banks 2-33 (shifted by 2)

Column access tile[0][0], tile[1][0], tile[2][0]...
Banks: 0, 1, 2, 3... → NO CONFLICT!
```

---

## When to Use LDS

### Good Use Cases

1. **Data reuse** — Same data accessed multiple times by different threads
2. **Access pattern transformation** — Like matrix transpose
3. **Thread communication** — Sharing intermediate results
4. **Reduction operations** — Combining values across threads
5. **Tiling for GEMM** — Loading matrix tiles for reuse (Experiments 04-05)

### When LDS Doesn't Help

1. **Single-pass streaming** — Data used once then discarded
2. **Small data** — Overhead of LDS management exceeds benefits
3. **Already cached** — If L1/L2 cache hit rate is high

---

## LDS Capacity Planning

With 64 KB per CU and potentially multiple blocks per CU, you must balance:

```
LDS per block = 64 KB / (number of concurrent blocks per CU)
```

| LDS per Block | Max Blocks per CU | Trade-off |
|---------------|-------------------|-----------|
| 64 KB | 1 | Maximum LDS, minimum occupancy |
| 32 KB | 2 | Balanced |
| 16 KB | 4 | High occupancy, limited LDS |
| 8 KB | 8 | Maximum occupancy |

More blocks mean more wavefronts to hide latency, but less LDS per block. This trade-off is kernel-specific.

---

## Common Patterns

### Pattern 1: Double Buffering

Overlap data loading with computation:

```cpp
__shared__ float buffer[2][TILE_SIZE];  // Two buffers
int current = 0;

// Load first tile
load_tile(buffer[current], ...);
__syncthreads();

for (int i = 1; i < num_tiles; i++) {
    // Load next tile into other buffer
    load_tile(buffer[1 - current], ...);
    
    // Compute on current buffer
    compute(buffer[current], ...);
    
    __syncthreads();
    current = 1 - current;  // Swap buffers
}

// Process final tile
compute(buffer[current], ...);
```

### Pattern 2: Parallel Reduction

Sum all elements efficiently:

```cpp
__shared__ float sdata[256];
sdata[tid] = input[global_tid];
__syncthreads();

// Reduce within block
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}

// Thread 0 has the sum
if (tid == 0) output[blockIdx.x] = sdata[0];
```

### Pattern 3: Histogram (Atomic LDS)

```cpp
__shared__ int histogram[256];

// Initialise to zero
if (threadIdx.x < 256) histogram[threadIdx.x] = 0;
__syncthreads();

// Atomic increment
atomicAdd(&histogram[input[tid]], 1);
__syncthreads();

// Write to global
if (threadIdx.x < 256) {
    atomicAdd(&global_histogram[threadIdx.x], histogram[threadIdx.x]);
}
```

---

## Connection to MFMA Kernels

In Experiments 04-05, you'll implement GEMM (matrix multiplication) using MFMA instructions. The pattern is:

```
1. Load tile of A from global → LDS
2. Load tile of B from global → LDS
3. __syncthreads()
4. Load from LDS → registers
5. Execute MFMA
6. Repeat for all tiles
7. Write results to global
```

LDS serves as a staging area, enabling:
- **Coalesced global loads** (each thread loads contiguous data)
- **Flexible local reads** (MFMA requires specific data layouts)
- **Data reuse** (same tile data used multiple times)

---

## Exercises

1. **Measure bank conflicts**: Implement two kernels—one with stride-1 access, one with stride-32. Compare throughput.

2. **Transpose performance**: Compare naive transpose (global only) vs LDS-optimised transpose. Measure speedup.

3. **Padding exploration**: Try the transpose with and without the "+1" padding. Measure the difference.

4. **LDS capacity**: Experiment with different LDS sizes and observe occupancy changes with `rocprof`.

5. **Reduction optimisation**: Implement parallel reduction with and without warp-level optimisations (shuffle in final steps).

---

## What's Next

Experiment 04 introduces MFMA instructions—the matrix computation engines at the heart of modern AI workloads. You'll combine everything learned so far: thread indexing, wavefront execution, and LDS management.

---

## Quick Reference

| Concept | Description |
|---------|-------------|
| LDS | 64 KB on-chip memory per CU, shared within block |
| Bank | One of 32 memory banks in LDS, 4 bytes each |
| Bank conflict | Multiple threads accessing same bank simultaneously |
| `__shared__` | Declaration specifier for LDS variables |
| `__syncthreads()` | Barrier synchronisation within block |

| Capacity | Per CU |
|----------|--------|
| Total LDS | 64 KB |
| Banks | 32 |
| Bank width | 4 bytes |
| Peak bandwidth | ~3 TB/s (aggregate) |

| Pattern | Use Case |
|---------|----------|
| Load-Sync-Compute-Sync | Basic cooperative pattern |
| Padding (+1) | Avoid column-access bank conflicts |
| Double buffering | Overlap load and compute |
| Parallel reduction | Efficient summation |
