# E2 — Wave-per-row GEMV with cross-lane reduction

**Depends on:** E1
**Priority:** compulsory — this is the working baseline for E3, E4, E5, E9, E10.

## 1. Objective

Fix E1's coalescing problem by inverting the assignment: one **wavefront** per output row, not one
thread. The 64 lanes read 64 consecutive elements together (coalesced), each lane accumulates its
own partial sum, and then the wave needs one final step E1 didn't: combining 64 partial sums into
the single scalar output. That combining step — cross-lane reduction — is the new piece of
machinery this exercise introduces.

## 2. Concept primer: coalesced reads + a reduction tree

```
 Same matrix W, same row-major layout. Now ONE WAVE owns row r, not one thread.

 Instant 1 (all 64 lanes execute their first load together):

   lane 0  → W[r][0]  ┐
   lane 1  → W[r][1]  │  64 CONSECUTIVE elements — 128 bytes total for FP16,
   lane 2  → W[r][2]  │  exactly one cache line, one transaction, fully used
   ...                │
   lane 63 → W[r][63] ┘

   Useful bytes / bytes transferred = 128/128 = 100%

 Loop over the row in chunks of 64 (or 64 × elements-per-lane if you unroll):
   chunk 0: lanes read W[r][0..63],   chunk 1: lanes read W[r][64..127], ...
   Each lane accumulates its own running partial sum as it goes.

 After the loop, 64 partial sums live in 64 different lanes — one per lane —
 and need to become ONE number. That's the reduction:

   DPP butterfly (register-to-register, no LDS, no barrier):

     step 1: lane i += lane (i XOR 1)   — pairs combine:  (0,1)(2,3)(4,5)...
     step 2: lane i += lane (i XOR 2)   — pairs of pairs:  (0..3)(4..7)...
     step 3: lane i += lane (i XOR 4)
     step 4: lane i += lane (i XOR 8)
     step 5: lane i += lane (i XOR 16)
     step 6: lane i += lane (i XOR 32)
     → after 6 steps (log2 64), every lane holds the full sum; lane 0 writes it out.

   LDS tree (the alternative — write partials to shared memory, barrier,
   have half the threads add the other half's value, barrier, repeat):

     partials[tid] = my_sum;  __syncthreads();
     for (stride = 32; stride > 0; stride >>= 1) {
         if (tid < stride) partials[tid] += partials[tid + stride];
         __syncthreads();
     }
```

The DPP path never touches LDS and never needs a barrier — it operates entirely on registers using
`v_add` with `row_shr`/`row_bcast`-style DPP modifiers. On CDNA it should win over the LDS tree,
but "should" is not "measured" — build both and compare, per the acceptance criteria below.

## 3. Step-by-step build

### Step 3.1 — files **[paste this]**

`wave_gemv.cpp` stub reuses the buffer-setup/timing shape from E1 (paste it forward) with two
kernel slots: `wave_gemv_dpp` and `wave_gemv_lds`, both `TODO(E2)`.

### Step 3.2 — where `x` lives **[type this, read first]**

`x` is small (a few KB) and every wave reads all of it. **Do not re-read `x` from global memory
inside the loop** — that's the single most common way this kernel goes wrong, because it's easy to
write `x[c]` inside the same loop as `W[...][c]` without noticing you're now paying for two global
reads per element instead of one. Load the relevant slice of `x` into LDS once per block (or
registers, if it fits), at kernel start, before the row loop:

```cpp
extern __shared__ half x_shared[];
for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    x_shared[c] = x[c];
}
__syncthreads();
```

### Step 3.3 — the kernel body, DPP variant **[type this]**

```cpp
__global__ void wave_gemv_dpp(const half *W, const half *x_shared_src, float *y, int rows, int cols) {
    int row = blockIdx.x;          // one block == one wave == one row
    int lane = threadIdx.x;        // 0..63

    extern __shared__ half x_shared[];
    for (int c = lane; c < cols; c += 64) x_shared[c] = x_shared_src[c];
    __syncthreads();

    float acc = 0.0f;
    for (int c = lane; c < cols; c += 64) {
        acc += (float)W[(size_t)row * cols + c] * (float)x_shared[c];
    }

    // DPP butterfly reduction — accumulate in FP32 always, see failure modes.
    // TODO(E2): implement the 6-step DPP reduction from the schematic above,
    // using __builtin_amdgcn_update_dpp or amd's DPP intrinsics. Check the
    // ISA guide's DPP section for the exact intrinsic signature available in
    // your ROCm version — this is worth reading rather than guessing.

    if (lane == 0) y[row] = acc;
}
```

### Step 3.4 — the kernel body, LDS variant **[type this]**

Same structure, but the reduction step uses the LDS tree from the schematic instead of DPP. Write
this one second, after the DPP version works, so you have something to compare it against.

### Step 3.5 — verify in the ISA **[paste this, then read it]**

```bash
make isa
grep -E "dpp|v_add.*row_shr" wave_gemv.isa.txt   # confirm DPP modifiers appear
grep "s_waitcnt vmcnt" wave_gemv.isa.txt          # confirm multiple loads can be in flight
```

Two things to check specifically:
- The DPP modifiers actually appear (not lowered to something slower — a plain cross-lane shuffle
  through LDS, for instance, if the compiler couldn't form the DPP instruction from your source).
- `s_waitcnt vmcnt` placement: naive code often serialises — issue one load, wait immediately,
  issue the next — which throws away most of the benefit of coalescing because now only one
  transaction is in flight at a time. Correct code lets several loads issue before the first
  `s_waitcnt`, so the memory system has several outstanding requests to work with. If your `.s`
  shows a `s_waitcnt vmcnt(0)` immediately after every single load, that's the bottleneck to fix —
  restructure the loop to unroll a few iterations before waiting.

### Step 3.6 — measure and compare **[paste this]**

```bash
make run
```

Compare GB/s and MBU: E1 vs E2, and DPP vs LDS within E2.

## 4. Acceptance criteria

- [ ] MBU meaningfully above E1's number, cache-cold.
- [ ] Both reduction variants implemented and measured; DPP vs LDS ratio reported (don't just
      assert DPP wins — show the number).
- [ ] ISA confirmation that DPP modifiers survived and that multiple loads are in flight before the
      first `s_waitcnt`.
- [ ] A `s_waitcnt` story you can narrate: where in your `.s` file the compiler (or you, by hand)
      let loads overlap, and by how much that changed the number versus a version that didn't.

## 5. Failure modes

- **Accumulating in FP16.** `half` accumulation over a long row loses precision catastrophically at
  the tail of long rows (magnitude growth swamps small increments). Accumulate in FP32 always, even
  though the inputs and output storage may be FP16 elsewhere.
- **Reducing across the wrong lane group.** If your block has more than 64 threads (e.g. multiple
  waves per block for other reasons), make sure the DPP reduction combines exactly the 64 lanes of
  one wavefront, not lanes from a different wave that happen to share a block.
- **Assuming wave32.** CDNA3 is wave64. Code copied from RDNA (wave32) tutorials will silently
  reduce only half the lanes.
- **Re-reading `x` from global memory inside the loop** (Step 3.2's warning) — the most common
  regression when refactoring this kernel later in E3/E4.

## 6. Checkpoint question

Why does putting `x` in LDS (or registers) instead of re-reading it from global memory help *more*
at large `cols` than at small `cols`? Think about what's actually being amortised.

## 7. What you can now explain

You have a working baseline you understand at the instruction level — where the compiler placed
`s_waitcnt`, why DPP beat (or didn't beat) LDS, and why `x` belongs in LDS. E3, E4, E5, E9, and E10
all start from this kernel.

Next: [`../e3-packed-dot/README.md`](../e3-packed-dot/README.md) (recommended) or
[`../e4-rows-per-wave/README.md`](../e4-rows-per-wave/README.md) (compulsory)
