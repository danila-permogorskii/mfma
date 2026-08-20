/**
 * =============================================================================
 * Experiment E3: Packed dot product (v_dot2_f32_f16)
 * =============================================================================
 *
 * PURPOSE:
 *   Halve the VALU instruction count on the multiply-accumulate and check
 *   whether that changes wall-clock time at batch 1 (it should not, if E2's
 *   kernel is correctly bandwidth-bound — that null result IS the finding).
 *
 * KEY CONCEPTS:
 *   - Packed FP16x2 dot product instruction (v_dot2_f32_f16)
 *   - Compilers often decline to auto-form this from scalar code — verify,
 *     don't assume (see README.md Step 3.4)
 *
 * SEE: README.md in this directory for the full guide.
 *
 * BUILD:
 *   amdclang++ -x hip --offload-arch=gfx942 -O3 -o dot2_gemv dot2_gemv.cpp -I../common
 *
 * =============================================================================
 */

#include "../common/gemv_common.hpp"
#include <hip/hip_fp16.h>
#include <cstdio>

// -----------------------------------------------------------------------------
// TODO(E3), README Steps 3.1-3.2: start from E2's wave_gemv_dpp (paste it),
// then replace the scalar multiply-add with the packed dot product.
// -----------------------------------------------------------------------------
__global__ void dot2_gemv(const half *W, const half *x, float *y, int rows, int cols) {
    // TODO(E3): implement per README Steps 3.1-3.2
}

int main(int argc, char **argv) {
    int rows = 8192, cols = 8192;
    if (argc >= 3) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
    }
    printf("dot2_gemv: rows=%d cols=%d\n", rows, cols);

    half *d_x;
    float *d_y;
    HIP_CHECK(hipMalloc(&d_x, cols * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_y, rows * sizeof(float)));

    // TODO(E3), README Step 3.3: same benchmark harness shape as E2 — compare
    // GB/s against E2's wave_gemv_dpp result directly.
    printf("TODO(E3): benchmark loop not yet implemented — see README.md Step 3.3\n");

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    return 0;
}
