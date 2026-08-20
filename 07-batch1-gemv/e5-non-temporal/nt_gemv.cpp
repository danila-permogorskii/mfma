/**
 * =============================================================================
 * Experiment E5: Non-temporal hints on weight loads
 * =============================================================================
 *
 * PURPOSE:
 *   Weights at batch 1 are read once and never touched again. Hint the
 *   weight loads as non-temporal and measure the effect on a CO-RESIDENT
 *   synthetic working set (a KV-cache stand-in) — the hint protects the
 *   other tenant, not the weight stream itself. See README.md.
 *
 * KEY CONCEPTS:
 *   - CDNA3 scope bits (sc0/sc1) on load instructions
 *   - Cache pollution from single-use data
 *
 * SEE: README.md in this directory for the full guide.
 *
 * BUILD:
 *   amdclang++ -x hip --offload-arch=gfx942 -O3 -o nt_gemv nt_gemv.cpp -I../common
 *
 * =============================================================================
 */

#include "../common/gemv_common.hpp"
#include <hip/hip_fp16.h>
#include <cstdio>

// -----------------------------------------------------------------------------
// TODO(E5), README Step 3.2: wave-per-row GEMV (start from E2's structure),
// with weight loads hinted non-temporal and a synthetic resident_set that is
// read normally (cached) every iteration, standing in for a KV cache.
// -----------------------------------------------------------------------------
__global__ void nt_gemv(const half *W, const half *x, float *y, float *resident_set,
                         size_t resident_elems, int rows, int cols, bool hint_weights) {
    // TODO(E5): implement per README Steps 3.2-3.3.
    // - weight loads: non-temporal hint IF hint_weights, ordinary load otherwise
    //   (keep both code paths so Configuration A/B can toggle it at runtime,
    //   or compile two variants — your choice, document which you picked)
    // - x load: ordinary, cached, unchanged
    // - resident_set access: ordinary, cached, unchanged — this is what you're measuring
}

int main(int argc, char **argv) {
    int rows = 8192, cols = 8192;
    size_t resident_mb = 32; // synthetic KV-cache stand-in size, tune per README Step 3.3
    printf("nt_gemv: rows=%d cols=%d, resident_set=%zu MB\n", rows, cols, resident_mb);

    half *d_x;
    float *d_y, *d_resident;
    HIP_CHECK(hipMalloc(&d_x, cols * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_y, rows * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_resident, resident_mb * 1024 * 1024));

    // TODO(E5), README Step 3.3: run Configuration A (weights alone, hinted
    // vs unhinted) and Configuration B (weights + resident_set, hinted vs
    // unhinted), reporting resident_set hit rate via rocprofv3/L2 counters
    // for Configuration B specifically.
    printf("TODO(E5): benchmark loop not yet implemented — see README.md Step 3.3\n");

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_resident));
    return 0;
}
