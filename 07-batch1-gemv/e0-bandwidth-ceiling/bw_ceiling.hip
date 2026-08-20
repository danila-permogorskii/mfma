/**
 * =============================================================================
 * Experiment E0: Bandwidth Ceiling
 * =============================================================================
 *
 * PURPOSE:
 *   Establish the denominator for every MBU (memory bandwidth utilization)
 *   number reported in the rest of this module: the sustained HBM bandwidth
 *   this GPU/partition actually delivers, measured by sweep, not quoted from
 *   a spec sheet.
 *
 * KEY CONCEPTS:
 *   - Load width (dword / dwordx2 / dwordx4) vs bytes-in-flight
 *   - Waves-per-CU vs latency hiding
 *   - Cache-cold measurement via RotatingBuffers<T> (see ../common/gemv_common.hpp)
 *
 * SEE: README.md in this directory for the full step-by-step guide. The
 * TODO(E0) blocks below are explained there — don't fill them in without
 * reading the corresponding step.
 *
 * BUILD:
 *   amdclang++ --offload-arch=gfx942 -O3 -o bw_ceiling bw_ceiling.hip -I../common
 *
 * =============================================================================
 */

#include "../common/gemv_common.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

// -----------------------------------------------------------------------------
// TODO(E0), README Step 3.2: streaming-read kernels, one per load width.
// Start with this dword version filled in as a worked example; write the
// dwordx2 and dwordx4 (and optionally dwordx3-via-asm) variants yourself
// following the same shape.
// -----------------------------------------------------------------------------
__global__ void stream_read_dword(const float* in, float* sink, size_t n_floats)
{
    size_t i  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for(size_t off = i; off < n_floats; off += (size_t)gridDim.x * blockDim.x)
    {
        acc += in[off];
    }
    if(acc == -1.0f)
        sink[i] = acc; // keeps the load alive without adding real traffic
}

// TODO(E0), README Step 3.2: __global__ void stream_read_dwordx2(...)
// TODO(E0), README Step 3.2: __global__ void stream_read_dwordx4(...)
// TODO(E0), README Step 3.3: non-temporal variant(s)

__global__ void stream_read_dwordx2(const float* in, float* sink, size_t n_vec2)
{
    size_t i   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float2 acc = {0, 0, 0, 0};
    for(size_t off = i; off < n_vec2; off += (size_t)gridDim.x * blockDim.x)
    {
        float2 v = in[off];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == -1.0f) sink[i] = acc.x;
}

__global__ void stream_read_dwordx4(const float *in, float *sink, size_t n_vec4) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    float4 acc = {0, 0, 0, 0};
    for (size_t off =i; off < n_vec4; off += (size_t)gridDix.x * blockDim.x) {
        float4 v = in[off];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == -1.0f) sink[i] = acc.x;
}

// -----------------------------------------------------------------------------
// TODO(E0), README Step 3.4: the sweep driver.
// For each (width, waves/CU, cached/NT) combination: run >=110 iterations
// through RotatingBuffers sized well over 512MB, discard the first 10,
// bracket the kernel region with device_timestamp(), and report median+IQR
// GB/s via median_iqr() from gemv_common.hpp.
// -----------------------------------------------------------------------------
void run_sweep()
{
    // TODO(E0): implement the sweep described in README Step 3.4.
    printf("TODO(E0): sweep not yet implemented — see README.md Step 3.4\n");
}

int main(int argc, char** argv)
{
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if(device_count == 0)
    {
        printf("No HIP devices found.\n");
        return 1;
    }
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (gfx%s)\n", prop.name, "942");

    run_sweep();

    return 0;
}
