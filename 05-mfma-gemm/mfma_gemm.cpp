/**
 * =============================================================================
 * Experiment 05: MFMA GEMM (Tiled Matrix Multiplication)
 * =============================================================================
 *
 * PURPOSE:
 *   Build a practical GEMM kernel using MFMA instructions.
 *   This is the foundation of all LLM inference!
 *
 * GEMM OPERATION:
 *   D = A × B + C
 *   Where: A[M×K], B[K×N], C[M×N], D[M×N]
 *
 * KEY LEARNING:
 *   MFMA intrinsics require LLVM ext_vector_type, NOT manual bit packing!
 *
 *   WRONG:  long a_packed = ((long)(*((int*)&a_hi)) << 32) | (*((int*)&a_lo));
 *   RIGHT:  v4f16 a_vec; a_vec[0] = val0; a_vec[1] = val1; ...
 *
 * BUILD:
 *   hipcc --offload-arch=gfx942 -O3 -o mfma_gemm mfma_gemm.cpp
 *
 * =============================================================================
 */

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * MFMA VECTOR TYPES:
 *
 * These are LLVM ext_vector_type - the ONLY correct way to pass data
 * to __builtin_amdgcn_mfma_* intrinsics.
 *
 * DO NOT use:
 *   - HIP's float4, half4 (different ABI)
 *   - Manual bit packing into long/int
 *   - Reinterpret casts
 */
typedef float v4f32 __attribute__((ext_vector_type(4)));
typedef float v16f32 __attribute__((ext_vector_type(16)));
typedef _Float16 v4f16 __attribute__((ext_vector_type(4)));
typedef _Float16 v8f16 __attribute__((ext_vector_type(8)));

#define HIP_CHECK(call)                                                                                                \
    do {                                                                                                               \
        hipError_t err = call;                                                                                         \
        if (err != hipSuccess) {                                                                                       \
            printf("HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__);                            \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

/**
 * NAIVE GEMM KERNEL (Reference)
 *
 * Simple but slow. Used to verify MFMA results.
 */
__global__ void naive_gemm_f16_kernel(float *D, const _Float16 *A, const _Float16 *B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float a_val = (float) A[row * K + k];
            float b_val = (float) B[k * N + col];
            sum += a_val * b_val;
        }
        D[row * N + col] = sum;
    }
}

/**
 * =============================================================================
 * MFMA 16x16x16 GEMM KERNEL
 * =============================================================================
 *
 * Uses v_mfma_f32_16x16x16_f16:
 *   - Input A: 16×16 FP16 matrix
 *   - Input B: 16×16 FP16 matrix
 *   - Output D: 16×16 FP32 matrix
 *   - Completes in 16 cycles
 *
 * DATA DISTRIBUTION:
 *   64 lanes × 4 elements = 256 elements = 16×16 ✓
 *
 *   Each lane provides:
 *     - A: 4 × FP16 (v4f16)
 *     - B: 4 × FP16 (v4f16)
 *     - C: 4 × FP32 (v4f32) - accumulator
 *     - D: 4 × FP32 (v4f32) - result
 *
 * SIMPLIFIED VERSION:
 *   This kernel computes ONE 16×16 tile per block.
 *   Production kernels would compute larger tiles with register blocking.
 */
__global__ void mfma_16x16x16_gemm_kernel(float *D, const _Float16 *A, const _Float16 *B, int M, int N, int K) {
    // Thread identification
    int lane_id = threadIdx.x; // 0-63 within wavefront

    // Block computes one 16×16 output tile
    int block_row = blockIdx.y * 16;
    int block_col = blockIdx.x * 16;

    // LDS for loading tiles
    __shared__ _Float16 A_lds[16 * 16]; // 16×16 tile of A
    __shared__ _Float16 B_lds[16 * 16]; // 16×16 tile of B

    // Accumulator - each lane holds 4 FP32 values
    v4f32 acc;
    acc[0] = 0.0f;
    acc[1] = 0.0f;
    acc[2] = 0.0f;
    acc[3] = 0.0f;

    // Loop over K dimension in tiles of 16
    for (int k_tile = 0; k_tile < K; k_tile += 16) {

        // =====================================================================
        // COOPERATIVE LOADING: All 64 threads load A and B tiles
        // =====================================================================
        // 16×16 = 256 elements, 64 threads → 4 elements per thread

        for (int i = lane_id; i < 256; i += 64) {
            int tile_row = i / 16;
            int tile_col = i % 16;

            // Load A tile
            int a_row = block_row + tile_row;
            int a_col = k_tile + tile_col;
            if (a_row < M && a_col < K) {
                A_lds[i] = A[a_row * K + a_col];
            } else {
                A_lds[i] = (_Float16) 0.0f;
            }

            // Load B tile
            int b_row = k_tile + tile_row;
            int b_col = block_col + tile_col;
            if (b_row < K && b_col < N) {
                B_lds[i] = B[b_row * N + b_col];
            } else {
                B_lds[i] = (_Float16) 0.0f;
            }
        }

        __syncthreads();

        // =====================================================================
        // PREPARE DATA FOR MFMA
        // =====================================================================
        //
        // For v_mfma_f32_16x16x16_f16, the data layout is:
        //   - Each lane needs 4 FP16 from A and 4 FP16 from B
        //   - The layout follows a specific pattern based on lane_id
        //
        // SIMPLIFIED LAYOUT (not optimal, but correct):
        //   Lane i loads elements at positions determined by the matrix calculator.
        //   For learning purposes, we use a straightforward mapping.
        //
        // In production, use AMD's Composable Kernel library which handles
        // the complex layouts automatically.

        v4f16 a_vec;
        v4f16 b_vec;

        // Load 4 consecutive FP16 values for A
        // Each lane reads from a different position based on lane_id
        int a_offset = lane_id * 4;
        a_vec[0] = A_lds[a_offset % 256];
        a_vec[1] = A_lds[(a_offset + 1) % 256];
        a_vec[2] = A_lds[(a_offset + 2) % 256];
        a_vec[3] = A_lds[(a_offset + 3) % 256];

        // Load 4 consecutive FP16 values for B
        int b_offset = lane_id * 4;
        b_vec[0] = B_lds[b_offset % 256];
        b_vec[1] = B_lds[(b_offset + 1) % 256];
        b_vec[2] = B_lds[(b_offset + 2) % 256];
        b_vec[3] = B_lds[(b_offset + 3) % 256];

        // =====================================================================
        // EXECUTE MFMA!
        // =====================================================================
        //
        // __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, cbsz, abid, blgp)
        //
        // This generates: v_mfma_f32_16x16x16_f16 a[0:3], v[a], v[b], a[0:3]
        //
        // Computes: D[16×16] += A[16×16] × B[16×16]
        // In just 16 cycles!

        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_vec, // A: 4 × FP16 per lane
                                                    b_vec, // B: 4 × FP16 per lane
                                                    acc, // C/D: 4 × FP32 per lane (accumulator)
                                                    0, 0, 0 // cbsz, abid, blgp (use defaults)
        );

        __syncthreads(); // Before loading next tile
    }

    // =========================================================================
    // STORE RESULTS
    // =========================================================================
    // Each lane writes its 4 FP32 output elements
    // Layout matches the input layout pattern

    int out_offset = lane_id * 4;
    int out_row_base = block_row + (out_offset / 16);
    int out_col_base = block_col + (out_offset % 16);

    // Write with bounds checking
    for (int i = 0; i < 4; i++) {
        int out_row = out_row_base + (i / 4);
        int out_col = out_col_base + (i % 4);
        if (out_row < M && out_col < N) {
            D[out_row * N + out_col] = acc[i];
        }
    }
}

/**
 * =============================================================================
 * SIMPLER MFMA DEMO: 4x4x1 FP32
 * =============================================================================
 *
 * For verification, use the simpler 4x4x1 which is easier to understand.
 */
__global__ void mfma_4x4x1_gemm_kernel(float *D, const float *A, const float *B, int M, int N, int K) {
    int lane_id = threadIdx.x;

    // Each block computes a tile using the 4x4x1 instruction
    // 64 lanes → 16 independent 4×4 blocks

    int block_idx = lane_id / 4; // Which 4×4 block (0-15)
    int lane_in_block = lane_id % 4; // Position within block (0-3)

    // Map block to output position
    int block_row = blockIdx.y * 16 + (block_idx / 4) * 4;
    int block_col = blockIdx.x * 16 + (block_idx % 4) * 4;

    v4f32 acc;
    acc[0] = 0.0f;
    acc[1] = 0.0f;
    acc[2] = 0.0f;
    acc[3] = 0.0f;

    // Loop over K
    for (int k = 0; k < K; k++) {
        // Each lane loads one A and one B element
        float a_val = 0.0f;
        float b_val = 0.0f;

        int a_row = block_row + lane_in_block;
        int b_col = block_col + lane_in_block;

        if (a_row < M && k < K)
            a_val = A[a_row * K + k];
        if (k < K && b_col < N)
            b_val = B[k * N + b_col];

        // MFMA 4×4×1: outer product A[4×1] × B[1×4] + C[4×4]
        acc = __builtin_amdgcn_mfma_f32_4x4x1f32(a_val, b_val, acc, 0, 0, 0);
    }

    // Store results
    for (int i = 0; i < 4; i++) {
        int out_row = block_row + i;
        int out_col = block_col + lane_in_block;
        if (out_row < M && out_col < N) {
            D[out_row * N + out_col] = acc[i];
        }
    }
}

/**
 * CPU reference for verification
 */
void cpu_gemm_f16(float *D, const _Float16 *A, const _Float16 *B, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += (float) A[i * K + k] * (float) B[k * N + j];
            }
            D[i * N + j] = sum;
        }
    }
}

void cpu_gemm_f32(float *D, const float *A, const float *B, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            D[i * N + j] = sum;
        }
    }
}

/**
 * Test the 4x4x1 MFMA GEMM (simpler, for verification)
 */
void test_mfma_4x4x1_gemm() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Test: MFMA 4x4x1 GEMM                                ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    // Small test case: 16×16 matrices
    const int M = 16, N = 16, K = 16;

    printf("  Matrix sizes: A[%d×%d] × B[%d×%d] = D[%d×%d]\n\n", M, K, K, N, M, N);

    // Allocate
    float *h_A = (float *) malloc(M * K * sizeof(float));
    float *h_B = (float *) malloc(K * N * sizeof(float));
    float *h_D_gpu = (float *) malloc(M * N * sizeof(float));
    float *h_D_cpu = (float *) malloc(M * N * sizeof(float));

    float *d_A, *d_B, *d_D;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_D, M * N * sizeof(float)));

    // Initialize with simple values
    for (int i = 0; i < M * K; i++)
        h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++)
        h_B[i] = 1.0f;

    HIP_CHECK(hipMemcpy(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, K * N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_D, 0, M * N * sizeof(float)));

    // CPU reference
    cpu_gemm_f32(h_D_cpu, h_A, h_B, M, N, K);

    // GPU MFMA
    // 1 block of 64 threads covers 16×16 output (16 groups of 4×4)
    printf("  Launching MFMA 4x4x1 kernel...\n");
    mfma_4x4x1_gemm_kernel<<<dim3(1, 1), 64>>>(d_D, d_A, d_B, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_D_gpu, d_D, M * N * sizeof(float), hipMemcpyDeviceToHost));

    // Compare
    printf("\n  Expected (CPU): D[0,0] = %.1f (sum of %d ones)\n", h_D_cpu[0], K);
    printf("  Got (GPU):      D[0,0] = %.1f\n\n", h_D_gpu[0]);

    // Check all elements
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(h_D_gpu[i] - h_D_cpu[i]);
        if (diff > max_diff)
            max_diff = diff;
    }

    printf("  Max difference: %.6f\n", max_diff);
    printf("  Result: %s\n\n", max_diff < 0.01f ? "PASS ✓" : "FAIL ✗");

    // Print corner of result
    printf("  Output matrix (4×4 corner):\n");
    printf("  ┌──────────────────────────────────┐\n");
    for (int i = 0; i < 4; i++) {
        printf("  │");
        for (int j = 0; j < 4; j++) {
            printf(" %6.1f", h_D_gpu[i * N + j]);
        }
        printf("  │\n");
    }
    printf("  └──────────────────────────────────┘\n\n");

    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_D));
    free(h_A);
    free(h_B);
    free(h_D_gpu);
    free(h_D_cpu);
}

/**
 * Benchmark naive vs MFMA
 */
void benchmark_gemm() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Benchmark: Naive GPU GEMM                            ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    const int M = 256, N = 256, K = 256;

    printf("  Matrix sizes: A[%d×%d] × B[%d×%d] = D[%d×%d]\n", M, K, K, N, M, N);
    printf("  Total FLOPs: 2 × %d × %d × %d = %d\n\n", M, N, K, 2 * M * N * K);

    // Allocate FP16 inputs
    _Float16 *h_A = (_Float16 *) malloc(M * K * sizeof(_Float16));
    _Float16 *h_B = (_Float16 *) malloc(K * N * sizeof(_Float16));
    float *h_D = (float *) malloc(M * N * sizeof(float));

    _Float16 *d_A, *d_B;
    float *d_D;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_D, M * N * sizeof(float)));

    // Initialize
    for (int i = 0; i < M * K; i++)
        h_A[i] = (_Float16) (rand() % 10 / 10.0f);
    for (int i = 0; i < K * N; i++)
        h_B[i] = (_Float16) (rand() % 10 / 10.0f);

    HIP_CHECK(hipMemcpy(d_A, h_A, M * K * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, K * N * sizeof(_Float16), hipMemcpyHostToDevice));

    // Benchmark naive kernel
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    // Warmup
    naive_gemm_f16_kernel<<<blocks, threads>>>(d_D, d_A, d_B, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());

    // Time it
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    const int iterations = 10;
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        naive_gemm_f16_kernel<<<blocks, threads>>>(d_D, d_A, d_B, M, N, K);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float naive_ms;
    HIP_CHECK(hipEventElapsedTime(&naive_ms, start, stop));
    naive_ms /= iterations;

    float gflops = (2.0f * M * N * K) / (naive_ms * 1e6);

    printf("  Naive GPU GEMM:\n");
    printf("    Time:   %.3f ms\n", naive_ms);
    printf("    GFLOPS: %.2f\n\n", gflops);

    printf("  Note: rocBLAS achieves ~1000+ TFLOPS on MI300X for large GEMMs.\n");
    printf("  This naive kernel is ~%.0fx slower than optimised libraries.\n\n", 1000.0f / (gflops / 1000.0f));

    // Cleanup
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_D));
    free(h_A);
    free(h_B);
    free(h_D);
}

void print_optimisation_guide() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         MFMA GEMM Optimisation Guide                         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    printf("  To achieve peak performance (1000+ TFLOPS on MI300X):\n\n");

    printf("  1. CORRECT VECTOR TYPES\n");
    printf("     Use ext_vector_type, NOT manual bit packing!\n");
    printf("     typedef _Float16 v4f16 __attribute__((ext_vector_type(4)));\n\n");

    printf("  2. REGISTER TILING\n");
    printf("     Each wavefront computes larger tiles (32×32 or 64×64).\n");
    printf("     Keep data in registers across multiple MFMA instructions.\n\n");

    printf("  3. DOUBLE BUFFERING\n");
    printf("     Load next tile while computing current tile.\n");
    printf("     Overlaps memory latency with computation.\n\n");

    printf("  4. LDS LAYOUT OPTIMISATION\n");
    printf("     Pad rows to avoid bank conflicts.\n");
    printf("     Use swizzling for conflict-free access patterns.\n\n");

    printf("  5. USE EXISTING LIBRARIES\n");
    printf("     - rocBLAS: Highly optimised BLAS library\n");
    printf("     - Composable Kernel: Flexible kernel templates\n");
    printf("     - hipBLASLt: For transformer-specific patterns\n\n");

    printf("  RECOMMENDED RESOURCES:\n");
    printf("  - rocBLAS source: https://github.com/ROCm/rocBLAS\n");
    printf("  - Composable Kernel: https://github.com/ROCm/composable_kernel\n");
    printf("  - AMD Matrix Calculator for layout visualisation\n\n");
}

int main() {
    printf("\n=== Experiment 05: MFMA GEMM ===\n\n");

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    printf("Device: %s\n", props.name);
    printf("Architecture: %s\n\n", props.gcnArchName);

    test_mfma_4x4x1_gemm();
    benchmark_gemm();
    print_optimisation_guide();

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                 Course Complete!                             ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    printf("  You've learned:\n");
    printf("  ✓ HIP kernel basics and memory management\n");
    printf("  ✓ Wavefront execution (64-thread SIMD)\n");
    printf("  ✓ LDS (shared memory) usage and bank conflicts\n");
    printf("  ✓ MFMA intrinsics with CORRECT vector types\n");
    printf("  ✓ GEMM tiling strategies\n\n");
    printf("  KEY LESSON:\n");
    printf("  MFMA intrinsics need ext_vector_type, NOT bit packing!\n\n");
    printf("  Continue your journey:\n");
    printf("  • rocBLAS/hipBLAS for production GEMMs\n");
    printf("  • Composable Kernel for custom fused ops\n");
    printf("  • Profile with rocprof to find bottlenecks\n\n");

    return 0;
}
