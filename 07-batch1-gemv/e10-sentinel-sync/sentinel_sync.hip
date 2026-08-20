/**
 * =============================================================================
 * Experiment E10: Sentinel synchronisation versus a naive barrier
 * =============================================================================
 *
 * PURPOSE:
 *   Three cross-block synchronisation schemes over the same producer/consumer
 *   dependency: naive global barrier, cooperative-groups grid sync, and a
 *   NaN-sentinel scheme where the data itself is the readiness flag. Includes
 *   the prefetch-across-a-dependency idea (issuing stage 2's loads before
 *   stage 1's completion is confirmed).
 *
 * KEY CONCEPTS:
 *   - CDNA3's same-type vector memory retirement-order guarantee (read the
 *     ISA guide's relevant section before relying on this)
 *   - Memory scope bits for cross-XCD visibility
 *   - LICM hazard on a polling loop
 *
 * SEE: README.md in this directory for the full guide.
 *
 * BUILD:
 *   amdclang++ -x hip --offload-arch=gfx942 -O3 -o sentinel_sync sentinel_sync.cpp -I../common
 *
 * =============================================================================
 */

#include "../common/gemv_common.hpp"
#include <cstdio>
#include <cfloat>
#include <cmath>

// -----------------------------------------------------------------------------
// TODO(E10), README Step 3.2: scheme (a) — atomic counter + spin.
// -----------------------------------------------------------------------------
__device__ int g_barrier_counter = 0;

__global__ void producer_consumer_naive_barrier(float *payload, int n_blocks) {
    // TODO(E10): implement per README Step 3.2
}

// -----------------------------------------------------------------------------
// TODO(E10), README Step 3.3: scheme (b) — cooperative-groups grid sync.
// Requires launch via hipLaunchCooperativeKernel; see README Step 3.3.
// -----------------------------------------------------------------------------
__global__ void producer_consumer_grid_sync(float *payload, int n_blocks) {
    // TODO(E10): implement per README Step 3.3 (#include <cooperative_groups.h>)
}

// -----------------------------------------------------------------------------
// TODO(E10), README Steps 3.4-3.6: scheme (c) — NaN sentinel. Consumer polls
// the DATA, not a separate flag. Force the poll load to stay in the loop
// (volatile or inline asm) so LICM cannot hoist it — verify with `make isa`.
// -----------------------------------------------------------------------------
__global__ void producer_consumer_sentinel(volatile float *payload, int n_blocks) {
    // TODO(E10): implement per README Steps 3.4-3.6, including the optional
    // early-issue prefetch of "stage 2" work described in Step 3.4.
}

int main(int argc, char **argv) {
    int n_blocks = 256;
    printf("sentinel_sync: n_blocks=%d\n", n_blocks);

    float *d_payload;
    HIP_CHECK(hipMalloc(&d_payload, n_blocks * sizeof(float)));

    // TODO(E10), README Step 3.7: for each scheme, initialise state
    // appropriately (NaN for scheme (c) specifically), run >=110 iterations,
    // record per-dependency latency via device_timestamp(), report
    // median+IQR. Then repeat with one producer block artificially delayed
    // and compare how each scheme's latency responds (the straggler
    // experiment).
    printf("TODO(E10): benchmark loop not yet implemented — see README.md Step 3.7\n");

    HIP_CHECK(hipFree(d_payload));
    return 0;
}
