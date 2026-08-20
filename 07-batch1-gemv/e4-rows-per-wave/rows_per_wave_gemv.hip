/**
 * =============================================================================
 * Experiment E4: Rows per wave, and the register wall
 * =============================================================================
 *
 * PURPOSE:
 *   Process R output rows per wave, reusing one load of x across all R rows.
 *   Sweep R and find the occupancy/register-pressure knee.
 *
 * KEY CONCEPTS:
 *   - Reuse of the small operand (x) — the only reuse available at batch 1
 *   - VGPR pressure vs occupancy vs latency-hiding capacity
 *   - Register spilling and how to see it in the ISA
 *
 * SEE: README.md in this directory for the full guide.
 *
 * BUILD:
 *   amdclang++ --offload-arch=gfx942 -O3 -o rows_per_wave_gemv rows_per_wave_gemv.hip -I../common
 *
 * =============================================================================
 */

#include "../common/gemv_common.hpp"
#include <hip/hip_fp16.h>
#include <cstdio>

// -----------------------------------------------------------------------------
// TODO(E4), README Step 3.2: R output rows per wave, x loaded once and
// reused R times. Template on R so each instantiation gets its own tight
// register allocation.
// -----------------------------------------------------------------------------
template <int R>
__global__ void rows_per_wave_gemv(const half *W, const half *x, float *y, int rows, int cols, int row_base) {
    // TODO(E4): implement per README Step 3.2, including the R-way DPP reduction
}

// Explicit instantiations so `make isa` / -Rpass-analysis can report register
// usage per R value independently.
template __global__ void rows_per_wave_gemv<1>(const half *, const half *, float *, int, int, int);
template __global__ void rows_per_wave_gemv<2>(const half *, const half *, float *, int, int, int);
template __global__ void rows_per_wave_gemv<4>(const half *, const half *, float *, int, int, int);
template __global__ void rows_per_wave_gemv<8>(const half *, const half *, float *, int, int, int);
template __global__ void rows_per_wave_gemv<16>(const half *, const half *, float *, int, int, int);

int main(int argc, char **argv) {
    int rows = 8192, cols = 8192;
    if (argc >= 3) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
    }
    printf("rows_per_wave_gemv: rows=%d cols=%d\n", rows, cols);

    half *d_x;
    float *d_y;
    HIP_CHECK(hipMalloc(&d_x, cols * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_y, rows * sizeof(float)));

    // TODO(E4), README Step 3.3: sweep R in {1,2,4,8,16}, launching
    // rows_per_wave_gemv<R> with grid = rows/R blocks. For each R, report
    // MBU (cache-cold) plus VGPR count / occupancy — see README Step 3.4 for
    // how to read those without a full benchmark run.
    printf("TODO(E4): sweep not yet implemented — see README.md Step 3.3\n");

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    return 0;
}
