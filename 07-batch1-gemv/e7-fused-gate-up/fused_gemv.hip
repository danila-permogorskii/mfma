/**
 * =============================================================================
 * Experiment E7 (optional): Fused gate/up projection
 * =============================================================================
 *
 * PURPOSE:
 *   Compare two separate launches vs one fused-sequential kernel vs one
 *   fused-interleaved kernel for a SwiGLU-style gate+up projection sharing
 *   one input activation — isolating launch-overhead elimination from
 *   memory-level-parallelism overlap.
 *
 * KEY CONCEPTS:
 *   - Kernel launch overhead
 *   - Interleaving two independent load streams for overlap
 *
 * SEE: README.md in this directory for the full guide.
 *
 * BUILD:
 *   amdclang++ -x hip --offload-arch=gfx942 -O3 -o fused_gemv fused_gemv.cpp -I../common
 *
 * =============================================================================
 */

#include "../common/gemv_common.hpp"
#include <hip/hip_fp16.h>
#include <cstdio>

// TODO(E7), README Step 3.2: reuse E2's wave_gemv_dpp for arrangement (a) —
// two separate launches, one against W_gate, one against W_up.
__global__ void wave_gemv_dpp(const half *W, const half *x, float *y, int rows, int cols) {
    // TODO(E7): paste E2's implementation here (or #include it)
}

// -----------------------------------------------------------------------------
// TODO(E7), README Step 3.3: arrangement (b) — one kernel, gate then up,
// fully sequential internally. Launch overhead paid once; no overlap.
// -----------------------------------------------------------------------------
__global__ void fused_sequential(const half *W_gate, const half *W_up, const half *x,
                                  float *y_gate, float *y_up, int rows, int cols) {
    // TODO(E7): implement per README Step 3.3
}

// -----------------------------------------------------------------------------
// TODO(E7), README Step 3.4: arrangement (c) — one kernel, one shared loop,
// gate-chunk and up-chunk loads issued together so both are in flight.
// -----------------------------------------------------------------------------
__global__ void fused_interleaved(const half *W_gate, const half *W_up, const half *x,
                                   float *y_gate, float *y_up, int rows, int cols) {
    // TODO(E7): implement per README Step 3.4
}

int main(int argc, char **argv) {
    int rows = 8192, cols = 8192;
    printf("fused_gemv: rows=%d cols=%d\n", rows, cols);

    // TODO(E7), README Step 3.5: allocate W_gate, W_up, x, y_gate, y_up and
    // measure total time for arrangements (a), (b), (c) under identical
    // conditions. Also measure a minimal empty-kernel launch loop to get
    // your own reference figure for launch overhead on this hardware.
    printf("TODO(E7): benchmark loop not yet implemented — see README.md Step 3.5\n");

    return 0;
}
