/**
 * =============================================================================
 * Experiment 06a: XCD Discovery — Förstå MI300X Chiplet-arkitektur
 * =============================================================================
 *
 * SYFTE (PURPOSE):
 *   Understand how workgroups are distributed across the 8 XCDs in MI300X.
 *   Learn the foundation for NUMA-aware kernel design.
 *   Förstå hur arbetsgrupper fördelas över de 8 XCD:erna i MI300X.
 *
 * MI300X ARCHITECTURE — ARKITEKTUR:
 *
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │                        MI300X (8 XCDs)                              │
 *   ├─────────────────────────────────────────────────────────────────────┤
 *   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                   │
 *   │  │  XCD 0  │ │  XCD 1  │ │  XCD 2  │ │  XCD 3  │    4 XCDs        │
 *   │  │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │    on IOD 0-1    │
 *   │  │  4MB L2 │ │  4MB L2 │ │  4MB L2 │ │  4MB L2 │                   │
 *   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘                   │
 *   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                   │
 *   │  │  XCD 4  │ │  XCD 5  │ │  XCD 6  │ │  XCD 7  │    4 XCDs        │
 *   │  │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │    on IOD 2-3    │
 *   │  │  4MB L2 │ │  4MB L2 │ │  4MB L2 │ │  4MB L2 │                   │
 *   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘                   │
 *   │                                                                     │
 *   │  Total: 304 CUs, 32 MB L2 Cache, 192 GB HBM3                       │
 *   │  Totalt: 304 beräkningsenheter, 32 MB L2-cache, 192 GB HBM3        │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 * KEY CONCEPT — NYCKELKONCEPT:
 *
 *   In SPX mode (Single Partition), workgroups are distributed ROUND-ROBIN
 *   across XCDs. This means:
 *   I SPX-läge fördelas arbetsgrupper ROUND-ROBIN över XCD:er. Detta innebär:
 *
 *     Workgroup 0 → XCD 0
 *     Workgroup 1 → XCD 1
 *     Workgroup 2 → XCD 2
 *     ...
 *     Workgroup 7 → XCD 7
 *     Workgroup 8 → XCD 0  (wraps around / går runt)
 *     Workgroup 9 → XCD 1
 *     ...
 *
 *   PROBLEM: If workgroups 0 and 8 share data, they're on the SAME XCD
 *            and can share L2 cache. But workgroups 0 and 1 are on
 *            DIFFERENT XCDs and cannot share L2 cache!
 *
 *   PROBLEM: Om arbetsgrupp 0 och 8 delar data, är de på SAMMA XCD
 *            och kan dela L2-cache. Men arbetsgrupp 0 och 1 är på
 *            OLIKA XCD:er och kan inte dela L2-cache!
 *
 * WHY THIS MATTERS — VARFÖR DETTA ÄR VIKTIGT:
 *
 *   L2 cache hit rates can vary from 1% to 97% depending on how you
 *   organise your workgroups! (See paper: https://arxiv.org/html/2511.02132)
 *   L2-cache-träfffrekvensen kan variera från 1% till 97% beroende på
 *   hur du organiserar dina arbetsgrupper!
 *
 * BUILD — KOMPILERA:
 *   hipcc --offload-arch=gfx942 -O3 -o xcd_discovery xcd_discovery.cpp
 *
 * RUN — KÖR:
 *   ./xcd_discovery
 *
 * =============================================================================
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/**
 * HIP_CHECK macro
 *
 * Always check HIP API calls for errors!
 */
#define HIP_CHECK(call)                                                                          \
    do                                                                                           \
    {                                                                                            \
        hipError_t err = call;                                                                   \
        if(err != hipSuccess)                                                                    \
        {                                                                                        \
            fprintf(                                                                             \
                stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
            exit(1);                                                                             \
        }                                                                                        \
    } while(0)

/**
 * NUM_XCDS - Number of XCDs in MI300X
 * Antal XCD:er i MI300X
 *
 * This is a constant for the MI300X architecture.
 */
constexpr int NUM_XCDS = 8;

/**
 * CUS_PER_XCD - Compute Units per XCD
 *
 * MI300X has 38 active CUs per XCD (40 total, 2 disabled for yield).
 */
constexpr int CUS_PER_XCD = 38;

/**
 * This kernel records which "logical XCD" each workgroup runs on.
 *
 * HOW IT WORKS
 *
 * We cannot directly query "which XCD am I on?" in HIP. Howewer, we CAN
 * infer it from the workgroup ID using the round-robin scheduling pattern.
 *
 * Logical XCD = workgroup_id % NUM_XCDS
 *
 * PARAMETERS:
 * @param workgroups_id     Output: workgroup ID for each workgroup
 * @param xcd_assignments   Output: inferred XCD assignment (0-7)
 * @param thread_counts     Output: number of threads that ran in each workgroup
 * @param num_workgroups    Total number of workgroups to launch
 */

__global__ void k_discover_xcd_assignment(int* workgroup_ids,
                                          int* xcd_assignments,
                                          int* thread_counts,
                                          int num_workgroups)
{
    /**
     * Get workgroup (block) ID
     *
     * In HIP/CUDA terminology:
     *   blockIdx.x = workgroup ID
     *   threadIdx.x = thread ID within workgroup
     */

    int wg_id = blockIdx.x;
    int tid   = threadIdx.x;

    /**
     * Only thread 0 of each workgroup records the data
     * This avoids race conditions (kapplöpningsförhållanden) where multiple
     * threads try to write to the same memory location.
     */

    if(tid == 0 && wg_id < num_workgroups)
    {
        // Record the workgroup
        workgroups[wg_id] = wg_id;

        /**
         * Infer XCD assignment using round-robin pattern
         *
         * In SPX mode, the hardware scheduler assigns workgroups to XCDs
         * in a simple round-robin fashion: 0,1,2,3,4,6,7,0,1,2,3,...
         */

        xcd_assignments[wg_id] = wg_id % NUM_XCDS;

        // Record block size
        thread_counts[wg_id] = blockDim.x;
    }
}

/**
 * KERNEL: k_demonstrate_naive_vs_swizzled
 *
 *  This kernel demonstrates the difference between NAIVE and SWIZZLED
 *  workgroup assignment patterns.
 *
 * SCENARIO:
 *  Imagine a 2D grid of tiles (like in GEMM or attention).
 *
 *  Grid: num_heads x num_blocks_per_head
 *
 * NAVIE MAPPING:
 *  Linear workgroup ID -> (head, block) in row-major order
 *
 *     WG 0 → Head 0, Block 0 → XCD 0
 *     WG 1 → Head 0, Block 1 → XCD 1
 *     WG 2 → Head 0, Block 2 → XCD 2
 *     ...
 *     WG 8 → Head 1, Block 0 → XCD 0
 *
 * PROBLEM: Blocks of the SAME head end up on DIFFERENT XCDs!
 *          They share K, V data but can't share L2 cache!
 *
 * SWIZZLED MAPPING:
 *  Remap workgroup IDs so that blocks of the same head stay on the same XCD.
 *
 *     WG 0 → Head 0, Block 0 → XCD 0
 *     WG 1 → Head 0, Block 1 → XCD 0  (SAME XCD! / SAMMA XCD!)
 *     WG 2 → Head 0, Block 2 → XCD 0
 *     ...
 *     WG 8 → Head 1, Block 0 → XCD 1  (Different head → different XCD)
 *
 * RESULT: L2 cache hit rate 80-97% instead of 1%!
 */

__global__ void k_demonstrate_naive_vs_swizzled(int* naive_head,
                                                int* naive_block,
                                                int* naive_xcd,
                                                int* swizzled_head,
                                                int* swizzled_block,
                                                int* swizzled_xcd,
                                                int num_heads,
                                                int blocks_per_head)
{
    int wg_id = blockIdx.x;
    int tid   = threadIdx.x;

    if(tid == 0)
    {
        /**
         * NAIVE MAPPING
         *
         * Simple linear mapping: iterate through blocks, then heads.
         *
         * Formula:
         *   head = wg_id / blocks_per_head
         *   block = wg_id % blocks_per_head
         */
        int n_head  = wg_id / blocks_per_head;
        int n_block = wg_id % blocks_per_head;
        int n_xcd   = wg_id % NUM_XCDS; // Round-robin XCD assignment

        naive_head[wg_id]  = n_head;
        naive_block[wg_id] = n_block;
        naive_xcd[wg_id]   = n_xcd;

        /**
         * SWIZZLED HEAD-FIRST MAPPING
         *
         * From paper https://arxiv.org/html/2511.02132 Figure 11:
         *
         * The goal is to assign ALL blocks of the same head to the SAME XCD.
         *
         * Key insight:
         *   head_per_xcd = num_heads / NUM_XCDS
         *   We group heads so each XCD handles (num_heads / 8) heads.
         *
         * Swizzling formula:
         *   target_xcd = head / heads_per_xcd
         *   Within that XCD, we process all blocks of each head sequentially.
         */
        int heads_per_xcd  = (num_threads + NUM_XCDS - 1) / NUM_XCDS; // Ceiling division
        int blocks_per_xcd = head_per_xcd * blocks_per_head;

        // Calculate which XCD this workgroup should run on
        int target_xcd = wg_id / blocks_per_xcd;
        if(target_xcd >= NUM_XCDS)
            target_xcd = NUM_XCDS - 1;

        // Calculate head and block within this XCD's allocation
        int local_id = wg_id % blocks_per_xcd;
        int s_head   = target_xcd * heads_per_xcd + (local_id / blocks_per_head);
        int s_block  = local_id % blocks_per_head;

        // Clamp head to valid range
        if(s_head >= num_heads)
            s_head = num_heads - 1;

        swizzled_head[wg_id]  = s_head;
        swizzled_block[wg_id] = s_block;
        swizzled_xcd[wg_id]   = target_xcd;
    }
}

/**
 * KERNEL: k_cache_locality_demo
 *
 * This kernel demonstrates cache locality effects by having workgroups
 * access shared data.
 *
 * SETUP:
 *  - We have NUM_HEADS "attention heads"
 *  - Each head has a KEY buffer that all its blocks share
 *
 * MEASUREMENT:
 *  - Run with rocprof to measure L2 cache hit rate
 *  - Each head has a KEY buffer that all its blocks share
 */

__global__ void
k_cache_locality_demo(float* shared_keys, // [num_heads][key_size] - shared data per head
                      float* output;      // [num_workgroups] - output per workgroup
                      int num_heads,
                      int blocks_per_head,
                      int key_size,
                      int use_swizzling // 0 = naive, 1 = swizzled

)
{
    int wg_id = blockIdx.x;
    int tid   = threadIdx.x;

    int head_idx;

    if(use_swizzling)
    {
        // SWIZZLED: Claculate head using swizzled mapping
        int head_per_xcd   = (num_heads + NUM_XCDS - 1) / NUM_XCDS;
        int blocks_per_xcd = heads_per_xcd * blocks_per_head;
        int target_xcd     = wg_id / blocks_per_head;
        if(target_xcd >= NUM_XCDS)
            target_xcd = NUM_XCDS - 1;
        int local_id = wg_id % blocks_per_xcd;
        head_idx     = target_xcd * heads_per_xcd + (local_id / blocks_per_head);
        if(head_idx >= num_heads)
            head_idx = num_heads - 1;
    }
    else
    {
        // NAIVE: Simple linear mapping
        heap_idx = wg_id / blocks_per_head;
        if(head_idx >= num_threads)
            head_idx = num_heads - 1;
    }

    /**
     * Access the KEY data for this head
     *
     * All workgroups processing the same head will access thesame KEY data.
     * If they're on the same XCD, this data will be in L2 cache!
     */

    float* key_ptr = shared_keys + head_idx * key_size;

    // Simulate reading KEY data (like in attention)
    float sum = 0.0f;
    for(int i = tid; i < key_size; i += blockDim.x)
    {
        sum += key_ptr[i];
    }

    // Reduce within workgroup (simplified)
    __shared__ float partial_sums[256];
    partial_sums[tid] = sum;
    __syncthreads();

    if(tid == 0)
    {
        float total = 0.0f;
        for(int i = 0; i < blockDim.x && i < 256; i++)
        {
            total += partial_sums[i];
        }
        output[wg_id] = total;
    }
}

void print_header()
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Experiment 06a: XCD Discovery — Förstå MI300X Chiplet-arkitektur            ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

void print_architecture_info()
{
    printf("┌──────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  MI300X Architecture Overview — Arkitekturöversikt                           │\n");
    printf("├──────────────────────────────────────────────────────────────────────────────┤\n");
    printf("│                                                                              │\n");
    printf("│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                            │\n");
    printf("│  │  XCD 0  │ │  XCD 1  │ │  XCD 2  │ │  XCD 3  │   ← 4 XCDs on IOD 0-1     │\n");
    printf("│  │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │                            │\n");
    printf("│  │  4MB L2 │ │  4MB L2 │ │  4MB L2 │ │  4MB L2 │   ← Private L2 per XCD    │\n");
    printf("│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                            │\n");
    printf("│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                            │\n");
    printf("│  │  XCD 4  │ │  XCD 5  │ │  XCD 6  │ │  XCD 7  │   ← 4 XCDs on IOD 2-3     │\n");
    printf("│  │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │                            │\n");
    printf("│  │  4MB L2 │ │  4MB L2 │ │  4MB L2 │ │  4MB L2 │                            │\n");
    printf("│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                            │\n");
    printf("│                                                                              │\n");
    printf("│  Key Point (Nyckelpunkt):                                                    │\n");
    printf("│  Each XCD has its OWN L2 cache. Data cached on XCD 0 is NOT visible         │\n");
    printf("│  to XCD 1! This is the NUMA effect we must optimize for.                    │\n");
    printf("│  Varje XCD har sin EGEN L2-cache. Data cachad på XCD 0 är INTE synlig       │\n");
    printf("│  för XCD 1! Detta är NUMA-effekten vi måste optimera för.                   │\n");
    printf("│                                                                              │\n");
    printf("└──────────────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
}

void print_round_robin_explanation()
{
    printf("┌──────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  Round-Robin Scheduling — Round-Robin-schemaläggning                         │\n");
    printf("├──────────────────────────────────────────────────────────────────────────────┤\n");
    printf("│                                                                              │\n");
    printf("│  In SPX mode, workgroups are assigned to XCDs in round-robin order:         │\n");
    printf("│  I SPX-läge tilldelas arbetsgrupper till XCD:er i round-robin-ordning:      │\n");
    printf("│                                                                              │\n");
    printf("│    WG 0 → XCD 0    WG 8  → XCD 0    WG 16 → XCD 0                           │\n");
    printf("│    WG 1 → XCD 1    WG 9  → XCD 1    WG 17 → XCD 1                           │\n");
    printf("│    WG 2 → XCD 2    WG 10 → XCD 2    WG 18 → XCD 2                           │\n");
    printf("│    WG 3 → XCD 3    WG 11 → XCD 3    WG 19 → XCD 3                           │\n");
    printf("│    WG 4 → XCD 4    WG 12 → XCD 4    WG 20 → XCD 4                           │\n");
    printf("│    WG 5 → XCD 5    WG 13 → XCD 5    WG 21 → XCD 5                           │\n");
    printf("│    WG 6 → XCD 6    WG 14 → XCD 6    WG 22 → XCD 6                           │\n");
    printf("│    WG 7 → XCD 7    WG 15 → XCD 7    WG 23 → XCD 7                           │\n");
    printf("│                                                                              │\n");
    printf("│  Pattern: XCD = workgroup_id %% 8                                            │\n");
    printf("│  Mönster: XCD = arbetsgrupp_id %% 8                                          │\n");
    printf("│                                                                              │\n");
    printf("└──────────────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
}
