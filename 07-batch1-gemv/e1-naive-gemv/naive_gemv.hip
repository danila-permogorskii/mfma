/**
 * =============================================================================
 * Experiment E1: Naive GEMV (deliberately bad baseline)
 * =============================================================================
 *
 * PURPOSE:
 *   One thread per output row, walking its row sequentially. Catastrophically
 *   uncoalesced by construction — this is what makes E2's fix legible.
 *
 * KEY CONCEPTS:
 *   - Memory coalescing across the 64 lanes of a wavefront
 *   - Cache-line utilisation (useful bytes / bytes transferred)
 *
 * SEE: README.md in this directory for the full guide.
 *
 * BUILD:
 *   amdclang++ --offload-arch=gfx942 -O3 -o naive_gemv naive_gemv.hip -I../common
 *
 * =============================================================================
 */

#include "../common/gemv_common.hpp"
#include <hip/hip_fp16.h>
#include <cstdio>
#include <vector>

// -----------------------------------------------------------------------------
// TODO(E1), README Step 3.2: one thread per output row, sequential walk.
// -----------------------------------------------------------------------------
__global__ void naive_gemv(const half *W, const half *x, float *y, int rows, int cols) {
    // TODO(E1): implement per README Step 3.2
}

int main(int argc, char **argv) {
    int rows = 8192, cols = 8192; // ~2 * 8192 * 8192 = 128 MB weight matrix (adjust as needed)
    if (argc >= 3) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
    }
    printf("naive_gemv: rows=%d cols=%d (%.1f MB per weight matrix)\n", rows, cols,
           rows * (double) cols * sizeof(half) / (1024.0 * 1024.0));

    // x and y are tiny relative to W; single buffers are fine for these.
    half *d_x;
    float *d_y;
    HIP_CHECK(hipMalloc(&d_x, cols * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_y, rows * sizeof(float)));

    // TODO(E1), README Step 3.3: size RotatingBuffers<half> for W so the total
    // exceeds 512 MB (pick `count` accordingly), warm up, then time >=110
    // iterations, discard first 10, report median+IQR GB/s via median_iqr().
    printf("TODO(E1): benchmark loop not yet implemented — see README.md Step 3.3\n");

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    return 0;
}
