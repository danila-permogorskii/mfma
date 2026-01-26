/**
 * =============================================================================
 * Experiment 03: LDS (Local Data Share) Memory
 * =============================================================================
 * 
 * PURPOSE:
 *   Master LDS - the fast on-chip memory shared within a workgroup.
 *   Critical for efficient MFMA kernels that tile large matrices.
 * 
 * KEY CONCEPTS:
 *   - LDS: 64KB per Compute Unit, shared by all threads in a block
 *   - MUCH faster than global memory (HBM)
 *   - Bank conflicts can kill performance (32 banks on CDNA3)
 *   - Essential for GEMM tiling strategies
 * 
 * MEMORY HIERARCHY (fastest to slowest):
 *   1. Registers (VGPRs, AGPRs, SGPRs) - ~0 cycles
 *   2. LDS (64KB per CU)               - ~20-30 cycles  
 *   3. L1 Cache (32KB)                 - ~50 cycles
 *   4. L2 Cache (256MB total)          - ~100-200 cycles
 *   5. HBM (192GB)                     - ~300+ cycles
 * 
 * ISA REFERENCE (MI300 ISA PDF, Section 11):
 *   "The local data share (LDS) is on-chip memory... 
 *    This memory is low-latency and high-bandwidth."
 * 
 * BUILD:
 *   hipcc --offload-arch=gfx942 -O3 -o lds_memory lds_memory.cpp
 * 
 * =============================================================================
 */

 #include <hip/hip_runtime.h>
 #include <stdio.h>
 #include <chrono>

 #define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            printf("HIP Error: %s at %s:%d\n", \
                hipGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0) \

/**
 * KERNEL 1: Basic LDS usage - declare shared memory
 * 
 * __shared__ marks memory as LDS (visible to all threads in block).
 * 
 * Pattern:
 *   1. Threads cooperatively load data from global → LDS
 *   2. __syncthreads() ensures all loads complete
 *   3. Threads read from LDS (potentially different locations)
 *   4. __syncthreads() before next iteration
 */
 __global__ void basic_lds_kernel(float* output, const float* input, int n) {
    // Declare shared memory - 64 floats, visible to all 64 threads
    // This is allocated in LDS (fast on-chip memory)
    __shared__ float shared_data[64];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    // Step 1: Each thread loads one element from global to shared
    if (tid < n) {
        shared_data[local_id] = input[tid];
    }

    // Step 2: BARRIER - wait for ALL threads to finish loading
    // Without this, some threads might read before others have written!
    __syncthreads();

    // Step 3: Each thread reads from a DIFFERENT location (reversed)
    // This demonstrates LDS's random access capability
    if (tid < n) {
        int read_idx = 63 - local_id; // Reverse order
        output[tid] = shared_data[read_idx] * 2.0f;
    }
 }

 /**
 * KERNEL 2: Demonstrate bank conflicts
 * 
 * LDS has 32 banks on CDNA3. Each bank can serve ONE request per cycle.
 * 
 * Bank assignment: address % 32 = bank number (for 4-byte elements)
 * 
 * CONFLICT: Multiple threads access SAME bank (different addresses)
 *           → Accesses serialise → Performance drops!
 * 
 * NO CONFLICT: 
 *   - Different threads access different banks
 *   - OR all threads access SAME address (broadcast)
 */

 // BAD: All threads in a wavefront access same bank (stride 32)
 __global__ void bank_conflict_kernel(float* output, int n, int iterations) {
    __shared__ float lds[2048]; // 8KB

    int tid = threadIdx.x;

    // Initialise LDS
    for (int i = tid; i < 2048; i += blockDim.x) {
        lds[i] = (float)i;
    }

    __syncthreads();

    float sum = 0.0f;

    // Access pattern with stride 32: ALL threads hit same bank!
    // Thread 0 -> lds[0] (bank 0)
    // Thread 1 -> lds[32] (bank 0) <- CONFLICT!
    // Thread 2 -> lds[64] (bank 0) <- CONFLICT!
    for (int iter = 0; iter < iterations; iter++) {
        int idx = tid * 32; // Stride of 32 -> all same bank!
        sum += lds[idx % 2048];
    }

    if (tid == 0) output[blockIdx.x] = sum;
 }

 // GOOD: Threads access consecutive addresses (no conflicts)
 __global__ void no_conflict_kernel(float* output, int n, int iterations) {
    __shared__ float lds[2048];

    int tid = threadIdx.x;

    // Initialise LDS
    for (int i = tid; i < 2048; i += blockDim.x) {
        lds[i] = (float)i;
    }

    __syncthreads();

    float sum = 0.0f;

    // Access pattern with stride 2: Perfect bank utilisation!
    // Thread 0 -> lds[0] (bank 0)
    // Thread 1 -> lds[1] (bank 0)
    // ...
    // Thread 31 -> lds[31] (bank 31)
    // Thread 32 -> lds[32] (bank 0) <- Different wavefront cycle
    for (int iter = 0; iter < iterations; iter++) {
        int idx = (tid + iter) % 2048; // Stride of 1 -> no conflicts
        sum += lds[idx];
    }

    if (tid == 0) output[blockDim.x] = sum;
 }

 
/**
 * KERNEL 3: Matrix transpose using LDS
 * 
 * Classic example: transpose a tile using shared memory.
 * This pattern is used in GEMM to ensure coalesced memory access.
 * 
 * Global memory prefers coalesced access (consecutive threads → consecutive addresses).
 * Transpose breaks this - LDS allows us to reorganise data.
 */

#define TILE_SIZE 32

__global__ void matrix_transpose_kernel(float* output, const float* input, int width, int height) {
    // Shared memory tile with padding to avoid bank conflicts
    // +1 padding: prevents conflicts when reading columns
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    // Global coordinates
    int x_in = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y_in = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load tile from global memory (coalesced read - threads read row)
    if (x_in < width && y_in < height) {
        tile[threadIdx.y][threadIdx.x] = input[y_in * width + x_in];
    }

    __syncthreads();

    // Output coordinates (transposed blocks)
    int x_out = blockIdx.y * TILE_SIZE + threadIdx.x;
    int y_out = blockIdx.x * TILE_SIZE + threadIdx.y;

    // Write transposed tile (coalesced write - threads write row)
    // Note: we READ tile[threadIdx.x][threadIdx.y] - transposed indexing!
    if (x_out < height && y_out < width) {
        output[y_out * height + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

void test_basic_lds() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Test 1: Basic LDS Usage                              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    const int N = 64;
    size_t bytes = N * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    float *d_input, *d_output;

    for (int i = 0; i < N; i++) h_input[i] = (float)i;

    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    HIP_CHECK(hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice));

    basic_lds_kernel<<<1, 64>>>(d_output, d_input, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost));

    printf(" Operation: Reverse array using LDS\n\n");
    printf(" Input: [");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_input[i]);
    printf("... %.0f]\n", h_input[N-1]);

    printf("  Output: [");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_output[i]);
    printf("... %.0f]\n\n", h_output[N-1]);
    
    // Verify: output[i] = input[63-i] * 2
    int correct = 1;
    for (int i = 0; i < N; i++) {
        float expected = h_input[63 - i] * 2.0f;
        if (h_output[i] != expected) correct = 0;
    }
    printf("  Result: %s\n\n", correct ? "PASS ✓" : "FAIL ✗");
    
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    free(h_input);
    free(h_output);
}

void test_bank_conflicts() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Test 2: Bank Conflicts Performance                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    const int iterations = 100000;
    float *d_output;
    HIP_CHECK(hipMalloc(&d_output, sizeof(float)));

    // Warm up
    bank_conflict_kernel<<<1, 64>>>(d_output, 1, 100);
    no_conflict_kernel<<<1, 64>>>(d_output, 1, 100);
    HIP_CHECK(hipDeviceSynchronize());

    // Time bank conflict version
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    bank_conflict_kernel<<<1, 64>>>(d_output, 1, iterations);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float conflict_ms;
    HIP_CHECK(hipEventElapsedTime(&conflict_ms, start, stop));

    // Time no conflict version
    HIP_CHECK(hipEventRecord(start));
    no_conflict_kernel<<<1, 64>>>(d_output, 1, iterations);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float no_conflict_ms;
    HIP_CHECK(hipEventElapsedTime(&no_conflict_ms, start, stop));

    printf("  LDS Bank Conflict Test (%d iterations):\n\n", iterations);
    printf("  ┌─────────────────────────┬────────────────┐\n");
    printf("  │ Access Pattern          │ Time (ms)      │\n");
    printf("  ├─────────────────────────┼────────────────┤\n");
    printf("  │ Stride 32 (conflicts)   │ %14.3f │\n", conflict_ms);
    printf("  │ Stride 1  (no conflicts)│ %14.3f │\n", no_conflict_ms);
    printf("  └─────────────────────────┴────────────────┘\n\n");
    printf("  Slowdown from conflicts: %.1fx\n\n", conflict_ms / no_conflict_ms);
    
    printf("  WHY THIS MATTERS FOR MFMA:\n");
    printf("  GEMM kernels load matrix tiles into LDS. Poor layout = bank conflicts.\n");
    printf("  Padding (+1 to row size) is a common fix.\n\n");
    
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_output));
}

void test_matrix_transpose() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Test 3: Matrix Transpose with LDS                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    const int WIDTH = 64;
    const int HEIGHT = 64;
    size_t bytes = WIDTH * HEIGHT * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    float *d_input, *d_output;

    // Initialise: input[y][x] = y * 100 + x
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            h_input[y * WIDTH + x] = (float)(y * 100 + x);
        }
    }

    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    HIP_CHECK(hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice));

    dim3 blocks((WIDTH + TILE_SIZE - 1) / TILE_SIZE, (HEIGHT + TILE_SIZE -1) / TILE_SIZE);
    dim3 threads(TILE_SIZE, TILE_SIZE);

    matrix_transpose_kernel<<<blocks, threads>>>(d_output, d_input, WIDTH, HEIGHT);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost));


    printf("  Original matrix (4x4 corner):\n");
    printf("  ┌─────────────────────────────────────┐\n");
    for (int y = 0; y < 4; y++) {
        printf("  │");
        for (int x = 0; x < 4; x++) {
            printf(" %5.0f", h_input[y * WIDTH + x]);
        }
        printf("  ...  │\n");
    }
    printf("  └─────────────────────────────────────┘\n\n");
    
    printf("  Transposed matrix (4x4 corner):\n");
    printf("  ┌─────────────────────────────────────┐\n");
    for (int y = 0; y < 4; y++) {
        printf("  │");
        for (int x = 0; x < 4; x++) {
            printf(" %5.0f", h_output[y * HEIGHT + x]);
        }
        printf("  ...  │\n");
    }
    printf("  └─────────────────────────────────────┘\n\n");
    
    // Verify: output[x][y] should equal input[y][x]
    int correct = 1;
    for (int y = 0; y < HEIGHT && correct; y++) {
        for (int x = 0; x < WIDTH && correct; x++) {
            float expected = h_input[x * WIDTH + y];  // Transposed!
            if (h_output[y * HEIGHT + x] != expected) correct = 0;
        }
    }
    printf("  Result: %s\n\n", correct ? "PASS ✓" : "FAIL ✗");

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    free(h_input);
    free(h_output);
}

int main() {
    printf("\n=== Experiment 03: LDS Memory ===\n\n");
    
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    printf("Device: %s\n", props.name);
    printf("LDS per block: %.0f KB\n", props.sharedMemPerBlock / 1024.0);
    printf("Max LDS per CU: 64 KB (hardware limit)\n\n");
    
    test_basic_lds();
    test_bank_conflicts();
    test_matrix_transpose();
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                     Summary                                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    printf("  ✓ LDS is fast on-chip memory shared within a block\n");
    printf("  ✓ 32 banks - avoid conflicts with stride != 32\n");
    printf("  ✓ __syncthreads() required between write and read phases\n");
    printf("  ✓ Padding (TILE_SIZE + 1) eliminates column-read conflicts\n\n");
    printf("NEXT: Experiment 04 - MFMA Introduction!\n\n");
    
    return 0;
}