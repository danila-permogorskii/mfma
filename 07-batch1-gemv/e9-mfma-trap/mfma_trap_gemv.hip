/**
 * =============================================================================
 * Experiment E9: The MFMA trap, demonstrated rather than asserted
 * =============================================================================
 *
 * PURPOSE:
 *   Pad the batch dimension from 1 to 16 to fill an MFMA tile, and compare
 *   against the E2/E4 VALU kernel on identical data. Weight traffic is
 *   unchanged by the padding — the honest prediction is comparable time,
 *   which is the point: MFMA is pointless here, not slow.
 *
 * KEY CONCEPTS:
 *   - v_mfma_f32_16x16x16_f16 tiling with 15/16 padded rows
 *   - AGPR pressure from accumulator tiles
 *   - Bytes-moved accounting as the real metric, not instruction count
 *
 * SEE: README.md in this directory, and ../../04-mfma-intro/mfma_intro.cpp /
 * ../../05-mfma-gemm/mfma_gemm.cpp for the MFMA vector-type conventions this
 * exercise reuses.
 *
 * BUILD:
 *   amdclang++ -x hip --offload-arch=gfx942 -O3 -o mfma_trap_gemv mfma_trap_gemv.cpp -I../common
 *
 * =============================================================================
 */

#include "../common/gemv_common.hpp"
#include <hip/hip_fp16.h>
#include <cstdio>

// MFMA vector types — LLVM ext_vector_type, per module 04/05's convention.
// DO NOT use manual bit packing (see 05-mfma-gemm/mfma_gemm.cpp's warning).
typedef float v4f32 __attribute__((ext_vector_type(4)));
typedef float v16f32 __attribute__((ext_vector_type(16)));
typedef _Float16 v4f16 __attribute__((ext_vector_type(4)));
typedef _Float16 v8f16 __attribute__((ext_vector_type(8)));

// -----------------------------------------------------------------------------
// TODO(E9), README Step 3.2: padded-batch MFMA kernel. Row 0 of the 16-row
// activation tile holds the real batch-1 activation; rows 1-15 are zero.
// -----------------------------------------------------------------------------
__global__ void mfma_trap_gemv(const half *W, const half *x, float *y, int rows, int cols) {
    // TODO(E9): implement per README Step 3.2
}

// -----------------------------------------------------------------------------
// Comparison baseline — E2's wave-per-row VALU kernel. Paste your working E2
// implementation here (or #include its .cpp) so both kernels run against
// identical data in the same binary.
// -----------------------------------------------------------------------------
__global__ void wave_gemv_dpp(const half *W, const half *x, float *y, int rows, int cols) {
    // TODO(E9): paste your working E2 implementation here
}

int main(int argc, char **argv) {
    int rows = 8192, cols = 8192;
    printf("mfma_trap_gemv: rows=%d cols=%d\n", rows, cols);

    half *d_x;
    float *d_y;
    HIP_CHECK(hipMalloc(&d_x, cols * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_y, rows * sizeof(float)));

    // TODO(E9), README Step 3.3: benchmark both kernels against the SAME
    // weight matrix and activation vector, cache-cold, and report GB/s + MBU
    // for each. Then follow README Step 3.4 for the ISA/counter verification.
    printf("TODO(E9): benchmark loop not yet implemented — see README.md Step 3.3\n");

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    return 0;
}
