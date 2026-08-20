/**
 * =============================================================================
 * Experiment E6 (optional): XCD-aware weight placement for GEMV
 * =============================================================================
 *
 * PURPOSE:
 *   Test whether matching weight-row placement to XCD assignment moves the
 *   MBU number for a bandwidth-bound GEMV, reusing module 06's XCD
 *   identification approach.
 *
 * KEY CONCEPTS:
 *   - 8 XCDs / 4 IODs topology (see ../../06-xcd-awareness/README.md)
 *   - NPS mode and its effect on what "controlled placement" means
 *
 * SEE: README.md in this directory, and ../../06-xcd-awareness/README.md
 * for the topology background this exercise reuses.
 *
 * BUILD:
 *   amdclang++ --offload-arch=gfx942 -O3 -o xcd_gemv xcd_gemv.hip -I../common
 *
 * =============================================================================
 */

#include "../common/gemv_common.hpp"
#include <hip/hip_fp16.h>
#include <cstdio>

// -----------------------------------------------------------------------------
// TODO(E6), README Step 3.1: reuse XCD-identification logic from
// 06-xcd-awareness/06a-xcd-discovery/xcd_discovery.cpp (e.g. via
// __builtin_amdgcn_read_exec or the HW-id register approach used there).
// -----------------------------------------------------------------------------
__device__ int get_xcd_id() {
    // TODO(E6): implement per module 06's approach
    return 0;
}

// -----------------------------------------------------------------------------
// TODO(E6), README Step 3.3: E2-style wave-per-row GEMV, run against both a
// default (round-robin) weight allocation and an XCD-matched one.
// -----------------------------------------------------------------------------
__global__ void xcd_gemv(const half *W, const half *x, float *y, int rows, int cols) {
    // TODO(E6): implement per README Step 3.3 (reuse E2's structure)
}

int main(int argc, char **argv) {
    int rows = 8192, cols = 8192;
    printf("xcd_gemv: rows=%d cols=%d\n", rows, cols);

    // TODO(E6), README Step 3.2: allocate W in row-partitioned chunks and
    // attempt XCD-matched placement; state your NPS mode before drawing
    // conclusions (see README Failure modes).
    printf("TODO(E6): placement + benchmark not yet implemented — see README.md Steps 3.2-3.3\n");

    return 0;
}
