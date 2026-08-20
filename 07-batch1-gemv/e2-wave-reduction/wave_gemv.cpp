/**
 * =============================================================================
 * Experiment E2: Wave-per-row GEMV with cross-lane reduction
 * =============================================================================
 *
 * PURPOSE:
 *   The canonical correct structure: one wave64 per output row, coalesced
 *   reads, DPP (or LDS-tree) cross-lane reduction. This is the working
 *   baseline for E3, E4, E5, E9, and E10.
 *
 * KEY CONCEPTS:
 *   - Coalesced access: 64 lanes read 64 consecutive elements
 *   - DPP butterfly reduction vs LDS-tree reduction
 *   - x staged once into LDS, never re-read from global memory per element
 *
 * SEE: README.md in this directory for the full guide.
 *
 * BUILD:
 *   amdclang++ -x hip --offload-arch=gfx942 -O3 -o wave_gemv wave_gemv.cpp -I../common
 *
 * =============================================================================
 */

#include "../common/gemv_common.hpp"
#include <hip/hip_fp16.h>
#include <cstdio>

// -----------------------------------------------------------------------------
// TODO(E2), README Steps 3.2-3.3: one wave (block of 64 threads) per row,
// x staged into LDS, DPP butterfly reduction.
// -----------------------------------------------------------------------------
__global__ void wave_gemv_dpp(const half *W, const half *x, float *y, int rows, int cols) {
    // TODO(E2): implement per README Steps 3.2-3.3
}

// -----------------------------------------------------------------------------
// TODO(E2), README Step 3.4: same structure, LDS-tree reduction instead of DPP.
// Write this one AFTER wave_gemv_dpp works, so you have a baseline to diff against.
// -----------------------------------------------------------------------------
__global__ void wave_gemv_lds(const half *W, const half *x, float *y, int rows, int cols) {
    // TODO(E2): implement per README Step 3.4
}

int main(int argc, char **argv) {
    int rows = 8192, cols = 8192;
    if (argc >= 3) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
    }
    printf("wave_gemv: rows=%d cols=%d (%.1f MB per weight matrix)\n", rows, cols,
           rows * (double) cols * sizeof(half) / (1024.0 * 1024.0));

    half *d_x;
    float *d_y;
    HIP_CHECK(hipMalloc(&d_x, cols * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_y, rows * sizeof(float)));

    // TODO(E2), README Step 3.6: launch <<<rows, 64, cols*sizeof(half)>>> for
    // each variant, through RotatingBuffers<half>(rows*cols, count) sized
    // well over 512MB, time >=110 iterations (discard first 10), and print
    // median+IQR GB/s and MBU for wave_gemv_dpp and wave_gemv_lds separately.
    printf("TODO(E2): benchmark loop not yet implemented — see README.md Step 3.6\n");

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    return 0;
}
