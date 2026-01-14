/**
* =============================================================================
 * Experiment 02: Wavefront Basics
 * =============================================================================
 *
 * PURPOSE:
 *   Understand wavefront execution - the fundamental unit of parallelism on AMD GPUs.
 *   This is CRITICAL for writing efficient MFMA kernels later.
 *
 * KEY CONCEPTS:
 *   - Wavefront: 64 threads executing in SIMD lockstep
 *   - Lane ID: thread's position within wavefront (0-63)
 *   - All lanes execute same instruction at same time
 *   - Divergent branches cause serialisation (avoid!)
 *
 * AMD vs NVIDIA TERMINOLOGY:
 *   AMD         | NVIDIA
 *   ------------|------------
 *   Wavefront   | Warp
 *   64 threads  | 32 threads
 *   Lane        | Lane
 *   CU          | SM
 *   LDS         | Shared Memory
 *
 * ISA REFERENCE (MI300 ISA PDF, Section 3):
 *   "CDNA processor groups 64 work-items into a wavefront"
 *   "Threads within a wavefront execute in lockstep (SIMD)"
 *
 * BUILD:
 *   hipcc --offload-arch=gfx942 -O3 -o wavefront_basics wavefront_basics.cpp
 *
 * =============================================================================
 */

#include <hip/hip_runtime.h>
#include <stdio.h>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hupSuccess) { \
            printf("HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
            exit(1);
        } \
    } while (0)

/**
 * KERNEL 1: Understand thread indexing within wavefronts
 *
 * Demonstrates:
 *   - Global thread ID (unique across entire grid)
 *   - Wavefront ID (which wavefront this thread belongs to)
 *   - Lane ID (position within wavefront, 0-63)
 *
 * The relationship:
 *   global_tid = wavefront_id * 64 + lane_id
 */

__global__ void thread_indexing_kernel(int* global_ids, int* wavefront_ids, int* lane_ids, int n) {
            // Global thread ID across entire grid
            int global_tid = blockIdx.x * blockDim.x + threadId.x;

            // Which wavefront within this block?
            // threadIdx.x / 64 gives wavefront number within block
            int wavefront_in_block = threadIdx.x / 64;

            // Which wavefront globally?
            int wavefronts_per_block = blockDim.x / 64;
            int wavefront_id = blockIdx.x * wavefronts_per_block + wavefront_in_block;

            // Lane ID: position within wavefront (0-63)
            // This is what matters for MFMA register layouts!
            int lane_id = threadIdx.x % 64;

            if (global_tid < n) {
                global_ids[global_tid]  = global_tid;
                wavefront_ids[wavefront_id] = wavefront_id;
                lane_ids[global_tid] = lane_id;
            }
        }

void test_thread_indexing() {

        }