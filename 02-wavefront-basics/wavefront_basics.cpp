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
if (err != hipSuccess) { \
printf("HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
exit(1); \
} \
} while(0)

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
            int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

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
            printf("╔══════════════════════════════════════════════════════════════╗\n");
            printf("║         Test 1: Thread Indexing Within Wavefronts            ║\n");
            printf("╚══════════════════════════════════════════════════════════════╝\n\n");

            const int N = 256; // 4 wavefronts
            size_t bytes = N * sizeof(int);

            int *h_global, *h_wavefront, *h_lane;
            int *d_global, *d_wavefront, *d_lane;

            h_global = (int*)malloc(bytes);
            h_wavefront = (int*)malloc(bytes);
            h_lane = (int*)malloc(bytes);

            HIP_CHECK(hipMalloc(&d_global, bytes));
            HIP_CHECK(hipMalloc(&d_wavefront, bytes));
            HIP_CHECK(hipMalloc(&d_lane, bytes));

            // Launch with 256 threads = 4 wavefronts in 1 block
            thread_indexing_kernel<<<1, 256>>>(d_global, d_wavefront, d_lane, N);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(h_global, d_global, bytes, hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h_wavefront, d_wavefront, bytes, hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h_lane, d_lane, bytes, hipMemcpyDeviceToHost));

            printf("Block 0, 256 threads = 4 wavefronts:\n\n");
            printf("  Thread │ Global ID │ Wavefront │ Lane │ Formula Check\n");
            printf("  ───────┼───────────┼───────────┼──────┼─────────────────\n");

            // Show first and last thread of each wavefront
            for (int wf = 0; wf < 4; wf++) {
                int first = wf * 64;
                int last = first + 63;
                printf(" %6d | %9d | %9d | %4d | %d*64+%d = %d \n",
                    first, h_global[first], h_wavefront[first], h_lane[first],
                    h_wavefront[first], h_lane[first], h_global[first]);
                printf(" %6d | %9d | %9d | %4d | %d*64+%d = %d \n",
                    last, h_global[last], h_wavefront[last], h_lane[last],
                    h_wavefront[last], h_lane[last], h_global[last]);
                if (wf < 3) printf("  ───────┼───────────┼───────────┼──────┼─────────────────\n");
            }

            printf("\n KEY INSIGHT: Lane ID (0-63) determines which matrix element\n");
            printf("  each thread computes in MFMA instructions!\n\n");

            HIP_CHECK(hipFree(d_global));
            HIP_CHECK(hipFree(d_wavefront));
            HIP_CHECK(hipFree(d_lane));
            free(h_global);
            free(h_wavefront);
            free(h_lane);
        }

int main() {
            printf("\n=== Experiment 02: Wavefront basics ===\n\n");

            hipDeviceProp_t props;
            HIP_CHECK(hipGetDeviceProperties(&props, 0));
            printf("Device: %s (wavefront size = %d)\n\n", props.name, props.warpSize);

            test_thread_indexing();

            return 0;
        }
