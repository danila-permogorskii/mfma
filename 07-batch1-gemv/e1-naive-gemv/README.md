# E1 — The naive GEMV, kept deliberately

**Depends on:** E0
**Priority:** compulsory

## 1. Objective

Build the obvious first implementation of `y = W x` at batch 1 — one thread per output row — and
measure it. It will be bad. That's the point: E2's improvement is only legible against a baseline
that shows what "bad" looks like in both the numbers and the memory trace, and skipping straight to
the good kernel would hide exactly the mechanism (coalescing) that makes E2 correct.

## 2. Concept primer: 64 lanes, one full row apart

A wavefront is 64 lanes executing in lockstep. When lane *L* and lane *L+1* issue a load in the
same instruction, the memory system can coalesce those two requests into one transaction **only if
the addresses are close together** — ideally within the same 128-byte cache line. One thread per
output row breaks this completely:

```
 Matrix W, row-major, shape [rows × cols]. Thread t owns row t.

 Instant 1 (all 64 lanes execute their first load together):

   lane 0  → W[0][0]  ┐
   lane 1  → W[1][0]  │  addresses are `cols` elements apart —
   lane 2  → W[2][0]  │  for cols=8192, FP16: 16 KB apart per lane
   ...                │
   lane 63 → W[63][0] ┘

   64 lanes, 64 different 128-byte cache lines touched, ONE instruction.
   Each line delivers 128 bytes; each lane needs 2 bytes of it.
   Useful bytes / bytes transferred = 2/128 ≈ 1.6%
```

Compare with what E2 will do — 64 lanes reading 64 *consecutive* elements, one cache line shared by
the whole wave. That structural difference, not anything about instruction choice, is the entire
story of why E1 is slow and E2 isn't.

## 3. Step-by-step build

### Step 3.1 — files **[paste this]**

`naive_gemv.hip` stub has `main()`, matrix/vector allocation via `RotatingBuffers<half>`, and the
timing harness already wired. The kernel is `TODO(E1)`.

### Step 3.2 — the kernel **[type this]**

```cpp
__global__ void naive_gemv(const half *W, const half *x, float *y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float acc = 0.0f;
    for (int c = 0; c < cols; ++c) {
        acc += (float)W[(size_t)row * cols + c] * (float)x[c];
    }
    y[row] = acc;
}
```

Notice what this kernel does *not* do: no attempt at coalescing, no LDS staging of `x`, no wave
reduction — each thread is fully self-sufficient and fully independent. That simplicity is exactly
why it's the natural first thing to write, and exactly why it's slow.

### Step 3.3 — launch and time **[type this]**

```cpp
int threads = 256;
int blocks = (rows + threads - 1) / threads;
// warm-up, then >=110 timed iterations through RotatingBuffers<half> for W,
// discard first 10, report median+IQR GB/s (see 00-measurement-methodology.md).
// Bytes moved per iteration = rows * cols * sizeof(half)  (x and y are negligible by comparison)
```

### Step 3.4 — verify with rocprofv3 **[paste this]**

```bash
make profile-v3
```

Look at the trace for indicators of pathological memory behaviour — high `req` counts relative to
useful bytes delivered, or (if you have `rocprof-compute`/L2 counters available, per module 06's
approach) a low L2 hit rate despite `x` being tiny and reused. The instruction mix itself is
unremarkable; the story here is entirely in the memory system, not the ISA — you don't need to
disassemble this one the way E2 and E3 require.

## 4. Acceptance criteria

- [ ] A GB/s and MBU number (against E0's denominator), cache-cold.
- [ ] A one-paragraph explanation of *why*, in terms of coalescing and the 2/128-bytes-useful
      arithmetic above — not just "it's slow," but the specific mechanism.
- [ ] Confirmation via rocprofv3 (or L2 counters if available) that the request pattern matches the
      prediction: many small, poorly-utilised transactions.

Expect single-digit MBU percentages. If you're seeing 30%+, something about your launch
configuration is accidentally coalescing — check your thread-to-row mapping.

## 5. Failure modes

- **Accidentally coalesced.** If you mapped threads to columns instead of rows, or transposed your
  mental model of row-major storage, you may get a much better number than intended — which means
  you built E2 by accident. Re-check the indexing against the schematic above.
- **Compiler auto-vectorising the inner loop in a way that masks the effect.** Unlikely at this
  access pattern, but if your number looks suspiciously good, check the ISA anyway.

## 6. Checkpoint question

If you doubled `cols` (a wider matrix, same row count), would you expect the MBU percentage to
change? Would you expect the *absolute* GB/s to change? Answer both before moving on — they're not
the same question.

## 7. What you can now explain

You can point at a rocprofv3 trace and connect a specific access pattern to a specific, quantified
cache-line-utilisation cost — not "uncoalesced access is bad" as received wisdom, but as something
you measured yourself.

Next: [`../e2-wave-reduction/README.md`](../e2-wave-reduction/README.md)
