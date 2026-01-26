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

/**
 * KERNEL 2: Demonstrate wavefront-uniform vs lane-specific values
 *
 * Key insight for MFMA:
 *   - Scalar values (same across wavefront) use SGPRs (Scalar GPRs)
 *   - Per-lane values use VGPRs (Vector GPRs) or AGPRs (Accumulation GPRs)
 *
 * This kernel shows the difference between:
 *   - blockIdx.x: Same for all threads in block (can be SGPR)
 *   - threadIdx.x: Different per thread (must be VGPR)
 */

__global__ void uniform_vs_divergent_kernel(int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        // blockIdx.x is UNIFORM across all threads in the block
        // The compiler can put this in an SGPR (efficient!)
        int uniform_value = blockIdx.x * 1000;

        // threadIdx.x is DIVERGENT - different per thread
        // Must be in VGPR (one value per lane)
        int divergent_value = threadIdx.x;

        output[tid] = uniform_value + divergent_value;
    }
}

/**
 * KERNEL 3: Wavefront-level reduction using shuffle operations
 *
 * AMD provides __shfl_* intrinsics for cross-lane communication.
 * This is FAST because it doesn't use memory!
 *
 * For MFMA, understanding lane communication is essential because
 * matrix elements are distributed across lanes.
 *
 * Key shuffle operations:
 *   __shfl(val, lane)     : Get value from specific lane
 *   __shfl_xor(val, mask) : XOR current lane with mask to get source lane
 *   __shfl_down(val, delta): Get value from lane + delta
 *   __shfl_up(val, delta) : Get value from lane - delta
 */
__global__ void wavefront_reduction_kernel(int* input, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 64;
    int wavefront_id = tid / 64;

    // Each thread loads its value
    int val = (tid < n) ? input[tid] : 0;

    // Parallel reduction withing wavefront using butterfly pattern
    // This sums all 64 values in log2(64) = 6 steps!
    //
    // Step 1: Lanes 0-31 get sum with lanes 32-63
    val += __shfl_xor(val, 32);
    // Step 2: Lanes 0-15,32-47 get sum with lanes 16-13,48-63
    val += __shfl_xor(val, 16);
    // Step 3:
    val += __shfl_xor(val, 8);
    // Step 4:
    val += __shfl_xor(val, 4);
    // Step 5:
    val += __shfl_xor(val, 2);
    // Step 6:
    val += __shfl_xor(val, 1);

    // Now all lanes hav the sum! Lane 0 writes it out.
    if (lane_id == 0 && wavefront_id < (n + 63) / 64) {
        output[wavefront_id] = val;
    }
}

/**
 * KERNEL 4: Demonstrate lane masking with EXEC
 * 
 * AMD GPUs use an EXEC (execute) mask to control which lanes are active.
 * When you have an if-statement, inactive lanes have their EXEC bit = 0.
 * 
 * This is important for understanding MFMA output:
 *   - All 64 lanes participate
 *   - Each lane computes a specific element of the output matrix
 */

 __global__ void exec_mask_demo_kernel(int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 64;

    if (tid < n) {
        // Different code paths based on lane_id
        // This causes wavefront divergence - all paths execute serially!
        if (lane_id < 16) {
            output[tid] = lane_id * 10; // Lanes 0-15
        } else if (lane_id < 32) {
            output[tid] = lane_id * 100; // Lanes 15-31
        } else if (lane_id < 48) {
            output[tid] = lane_id * 1000; // Lanes 32-47
        } else {
            output[tid] = lane_id * 10000; // Lanes 48-53
        }
        // In the ISA, this compiles to multiple instructions with different EXCE masks
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

void test_wavefront_reduction() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Test 2: Wavefront Reduction (Cross-Lane Ops)         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    const int N = 64;  // One wavefront
    size_t in_bytes = N * sizeof(int);
    size_t out_bytes = sizeof(int);

    int *h_input = (int*)malloc(in_bytes);
    int *h_output = (int*)malloc(out_bytes);
    int *d_input, *d_output;

    // Initialize: each lane has value = lane_id
    // Sum should be 0+1+2+...+63 = 64*63/2 = 2016
    for (int i = 0; i < N; i++) {
        h_input[i] = i;
    }

    HIP_CHECK(hipMalloc(&d_input, in_bytes));
    HIP_CHECK(hipMalloc(&d_output, out_bytes));
    HIP_CHECK(hipMemcpy(d_input, h_input, in_bytes, hipMemcpyHostToDevice));

    wavefront_reduction_kernel<<<1, 64>>>(d_input, d_output, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_output, d_output, out_bytes, hipMemcpyDeviceToHost));

    int expected = 64 * 63 / 2;  // Sum of 0..63
    printf("  Input:    lane_id values [0, 1, 2, ..., 63]\n");
    printf("  Expected: 0+1+2+...+63 = %d\n", expected);
    printf("  Got:      %d\n", h_output[0]);
    printf("  Result:   %s\n\n", h_output[0] == expected ? "PASS ✓" : "FAIL ✗");

    printf("  HOW IT WORKS (butterfly reduction):\n");
    printf("  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │ Step 1: val += __shfl_xor(val, 32)  // lanes 0↔32, 1↔33 │\n");
    printf("  │ Step 2: val += __shfl_xor(val, 16)  // lanes 0↔16, 1↔17 │\n");
    printf("  │ Step 3: val += __shfl_xor(val, 8)   // lanes 0↔8,  1↔9  │\n");
    printf("  │ Step 4: val += __shfl_xor(val, 4)   // lanes 0↔4,  1↔5  │\n");
    printf("  │ Step 5: val += __shfl_xor(val, 2)   // lanes 0↔2,  1↔3  │\n");
    printf("  │ Step 6: val += __shfl_xor(val, 1)   // lanes 0↔1,  2↔3  │\n");
    printf("  └─────────────────────────────────────────────────────────┘\n");
    printf("  Result: All 64 values summed in just 6 steps!\n\n");

    printf("  KEY INSIGHT: MFMA uses similar cross-lane data movement.\n");
    printf("  Matrix A and B elements are distributed across lanes!\n\n");

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    free(h_input);
    free(h_output);
}

void test_uniform_vs_divergent() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Test 2: Uniform vs Divergent Values                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    const int N = 256; // 4 wavefronts across 4 blocks
    size_t bytes = N * sizeof(int);

    int *h_output = (int*)malloc(bytes);
    int *d_output;

    HIP_CHECK(hipMalloc(&d_output, bytes));

    // Launch: 4 blocks * 64 threads = 256 threads total
    // Each block has blockIdx.x = 0, 1, 2, or 3
    uniform_vs_divergent_kernel<<<4, 64>>>(d_output, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost));

    printf("  Operation: output[tid] = blockIdx.x * 1000 + threadIdx.x\n\n");
    printf("  blockIdx.x is UNIFORM within a block (same for all 64 threads)\n");
    printf("  threadIdx.x is DIVERGENT (different per thread)\n\n");
    
    printf("  ┌─────────┬───────────┬────────────┬────────────┬─────────────┐\n");
    printf("  │ Block   │ Thread    │ Uniform    │ Divergent  │ Output      │\n");
    printf("  │ (idx.x) │ (idx.x)   │ (blk*1000) │ (thr)      │             │\n");
    printf("  ├─────────┼───────────┼────────────┼────────────┼─────────────┤\n");

    // Show firt and last thread of each block
    for (int block = 0; block < 4; block++) {
        int first_tid = block * 64;
        int last_tid = first_tid + 63;

        int uniform_part = block * 1000;

        printf("    | %7d | %9d | %10d | %10d | %11d |\n",
            block, 0, uniform_part, 0, h_output[first_tid]);
        printf("    | %7d | %9d | %10d | %10d | %11d |\n",
            block, 63, uniform_part, 63, h_output[last_tid]);

        if (block < 3) {
            printf("  ├─────────┼───────────┼────────────┼────────────┼─────────────┤\n");
        }
    }
    printf("  └─────────┴───────────┴────────────┴────────────┴─────────────┘\n\n");

    // Verify
    int correct = 1;
    for (int i = 0; i < N; i++) {
        int block = i / 64;
        int thread = i % 64;
        int expected = block * 1000 + thread;
        if (h_output[i] != expected) correct = 0;
    }

    printf("  Result: %s\n\n", correct ? "PASS ✓" : "FAIL ✗");
    
    printf("  WHY THIS MATTERS:\n");
    printf("  ┌─────────────────────────────────────────────────────────────┐\n");
    printf("  │ UNIFORM values (same across wavefront) → stored in SGPRs   │\n");
    printf("  │ DIVERGENT values (different per lane)  → stored in VGPRs   │\n");
    printf("  │                                                             │\n");
    printf("  │ SGPRs are scarce (104 max) but shared across 64 lanes.     │\n");
    printf("  │ VGPRs are plentiful (256 per lane) but use more power.     │\n");
    printf("  │                                                             │\n");
    printf("  │ For MFMA: loop counters → SGPR, matrix elements → VGPR     │\n");
    printf("  └─────────────────────────────────────────────────────────────┘\n\n");
    
    HIP_CHECK(hipFree(d_output));
    free(h_output);
}

void test_exec_mask() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Test 3: EXEC Mask and Divergent Branches             ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    const int N = 64; // One wavefront
    size_t bytes = N * sizeof(int);

    int *h_output = (int*)malloc(bytes);
    int *d_output;

    HIP_CHECK(hipMalloc(&d_output, bytes));

    // Launch exactly 1 wavefront (64 threads)
    exec_mask_demo_kernel<<<1, 64>>>(d_output, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost));

    printf("  Operation: Different multipliers based on lane_id range\n\n");
    printf("  Code structure:\n");
    printf("  ┌─────────────────────────────────────────────────────────────┐\n");
    printf("  │ if (lane_id < 16)           output = lane_id * 10;         │\n");
    printf("  │ else if (lane_id < 32)      output = lane_id * 100;        │\n");
    printf("  │ else if (lane_id < 48)      output = lane_id * 1000;       │\n");
    printf("  │ else                        output = lane_id * 10000;      │\n");
    printf("  └─────────────────────────────────────────────────────────────┘\n\n");
    
    printf("  Results by lane group:\n\n");
    printf("  ┌──────────────┬──────────┬────────────┬───────────────────────┐\n");
    printf("  │ Lane Range   │ Lane ID  │ Multiplier │ Output                │\n");
    printf("  ├──────────────┼──────────┼────────────┼───────────────────────┤\n");

    // Show samples from each group
    int samples[] = {0, 15, 16, 31, 32, 47, 48, 63};
    int multipliers[] = {10, 10, 100, 100, 1000, 1000, 10000, 10000};
    const char* groups[] = {"[0-15]", "[0-15]", "[16-31]", "[16-31]",
                            "[32-47]", "[32-47]", "[48-63]", "[48-63]"};

    for (int i = 0; i < 8; i++) {
        int lane = samples[i];
        printf("    | %-12s | %8d | %10d | %21d |\n",
        groups[i], lane, multipliers[i], h_output[lane]);
    }

    printf("  └──────────────┴──────────┴────────────┴───────────────────────┘\n\n");

    // Verify
    int correct = 1;
    for (int i = 0; i < N; i++) {
        int expected;
        if (i < 16)       expected = i * 10;
        else if (i < 32)  expected = i * 100;
        else if (i < 48)  expected = i * 1000;
        else              expected = i * 10000;
        
        if (h_output[i] != expected) correct = 0;
    }
    printf("  Result: %s\n\n", correct ? "PASS ✓" : "FAIL ✗");
    
    printf("  HOW EXEC MASK WORKS:\n");
    printf("  ┌─────────────────────────────────────────────────────────────┐\n");
    printf("  │ All 64 lanes start with EXEC = 0xFFFFFFFFFFFFFFFF (all 1s) │\n");
    printf("  │                                                             │\n");
    printf("  │ Branch 1 (lane < 16):                                       │\n");
    printf("  │   EXEC = 0x000000000000FFFF  (lanes 0-15 active)           │\n");
    printf("  │   → Execute: output = lane_id * 10                         │\n");
    printf("  │                                                             │\n");
    printf("  │ Branch 2 (lane 16-31):                                      │\n");
    printf("  │   EXEC = 0x00000000FFFF0000  (lanes 16-31 active)          │\n");
    printf("  │   → Execute: output = lane_id * 100                        │\n");
    printf("  │                                                             │\n");
    printf("  │ Branch 3 (lane 32-47):                                      │\n");
    printf("  │   EXEC = 0x0000FFFF00000000  (lanes 32-47 active)          │\n");
    printf("  │   → Execute: output = lane_id * 1000                       │\n");
    printf("  │                                                             │\n");
    printf("  │ Branch 4 (lane 48-63):                                      │\n");
    printf("  │   EXEC = 0xFFFF000000000000  (lanes 48-63 active)          │\n");
    printf("  │   → Execute: output = lane_id * 10000                      │\n");
    printf("  └─────────────────────────────────────────────────────────────┘\n\n");
    
    printf("  PERFORMANCE IMPACT:\n");
    printf("  ┌─────────────────────────────────────────────────────────────┐\n");
    printf("  │ This 4-way branch executes ALL 4 paths sequentially!       │\n");
    printf("  │ Total cycles ≈ 4× compared to uniform branch.              │\n");
    printf("  │                                                             │\n");
    printf("  │ AVOID divergent branches in performance-critical code.     │\n");
    printf("  │ Use predication or restructure algorithms instead.         │\n");
    printf("  │                                                             │\n");
    printf("  │ For MFMA: All 64 lanes participate in matrix ops,          │\n");
    printf("  │ so divergence is rarely an issue inside MFMA itself.       │\n");
    printf("  └─────────────────────────────────────────────────────────────┘\n\n");
    
    HIP_CHECK(hipFree(d_output));
    free(h_output);
}

int main() {
    printf("\n=== Experiment 02: Wavefront Basics ===\n\n");
    
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    printf("Device: %s (wavefront size = %d)\n\n", props.name, props.warpSize);
    
    test_thread_indexing();
    test_uniform_vs_divergent();
    test_exec_mask();
    test_wavefront_reduction();
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                     Summary                                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    printf("  ✓ AMD wavefronts have 64 threads (not 32 like NVIDIA)\n");
    printf("  ✓ Lane ID (0-63) determines data layout in MFMA\n");
    printf("  ✓ Uniform values → SGPRs, Divergent values → VGPRs\n");
    printf("  ✓ EXEC mask controls which lanes execute each instruction\n");
    printf("  ✓ Cross-lane operations (__shfl_*) enable fast reductions\n");
    printf("  ✓ Divergent branches serialise - avoid in hot paths!\n\n");
    printf("NEXT: Experiment 03 - LDS (Local Data Share) Memory\n\n");
    
    return 0;
}