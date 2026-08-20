# E7 — Two matrices, one x: the fused gate/up projection (optional)

**Depends on:** E2
**Priority:** optional — cut this one first if the week runs long.

## 1. Objective

In a SwiGLU MLP, the gate and up projections both consume the same input activation vector. Load
`x` once and compute both projections in one kernel, interleaving the two weight streams so loads
from both are in flight simultaneously — a first step toward "one activation, several consumers"
thinking, which is the structural idea behind fusing more of a decode layer into fewer kernels.

## 2. Concept primer: three ways to spend the same launch budget

```
 (a) Two separate kernel launches:

   launch(gate_gemv) ──wait── launch(up_gemv) ──wait──
   │←── launch overhead ──→│  │←── launch overhead ──→│
   Two full launch costs, weight streams never overlap.

 (b) One fused kernel, sequential internally:

   launch(fused) { gate work; up work; }
   │←── one launch overhead ──→│
   Launch cost paid once. Weight streams still don't overlap —
   still fully serial inside the kernel.

 (c) One fused kernel, interleaved:

   launch(fused) { for each chunk: gate_chunk load; up_chunk load; (both in flight) }
   │←── one launch overhead ──→│
   Launch cost paid once, AND both weight streams have outstanding
   requests simultaneously — more memory-level parallelism.
```

The three-way comparison isolates two different, easily-conflated effects: launch overhead
elimination (b vs a) and overlap from interleaving (c vs b). A lot of "fusion helps" claims blur
these together; this exercise measures them separately.

## 3. Step-by-step build

### Step 3.1 — files **[paste this]**

Two copies of E2's `wave_gemv_dpp` — one reading `W_gate`, one reading `W_up`, same `x`.

### Step 3.2 — arrangement (a): two launches **[paste this]**

```cpp
wave_gemv_dpp<<<rows, 64, cols*sizeof(half)>>>(W_gate, x, y_gate, rows, cols);
wave_gemv_dpp<<<rows, 64, cols*sizeof(half)>>>(W_up,   x, y_up,   rows, cols);
```

### Step 3.3 — arrangement (b): fused, sequential **[type this]**

```cpp
__global__ void fused_sequential(const half *W_gate, const half *W_up, const half *x,
                                  float *y_gate, float *y_up, int rows, int cols) {
    // stage x into LDS once (shared by both halves)
    // TODO(E7): compute the gate projection for this row (same body as E2), then
    //           compute the up projection for this row (same body again) — fully
    //           sequential, just inside one kernel instead of two.
}
```

### Step 3.4 — arrangement (c): fused, interleaved **[type this]**

```cpp
__global__ void fused_interleaved(const half *W_gate, const half *W_up, const half *x,
                                   float *y_gate, float *y_up, int rows, int cols) {
    // stage x into LDS once
    // TODO(E7): in the SAME loop over c, issue the gate-chunk load and the
    // up-chunk load before either result is consumed, so both have requests
    // outstanding at once — this is the memory-level-parallelism difference
    // from arrangement (b). Two separate accumulators, one shared loop.
}
```

### Step 3.5 — measure all three **[paste this]**

```bash
make run
```

Report total time for (a), (b), and (c). The interesting comparison is (c) vs (b), since it
isolates overlap from launch-overhead elimination, which (b) vs (a) already captures.

## 4. Acceptance criteria

- [ ] All three arrangements measured under identical conditions (same rows/cols, same regime).
- [ ] A quantified launch-overhead figure: (a) minus (b), compared against a reference range for
      kernel launch cost on this hardware that you either measure directly (e.g. via a
      minimal empty-kernel launch loop) or find independently — don't take a number from memory
      without checking it against your own machine.
- [ ] A statement about whether interleaving (c vs b) bought anything beyond what fusing alone did.

## 5. Failure modes

- **Attributing all of the (a)→(b) gain to "fusion" without separating out plain launch-overhead
  elimination**, which is a much simpler effect than memory-level-parallelism overlap.
- **Measuring (c) with too small a `cols`** to give the scheduler room to actually overlap the two
  streams — if each row is short, there's little to interleave.

## 6. Checkpoint question

If (c) shows no improvement over (b), what would that tell you about where this kernel's actual
bottleneck already was?

## 7. What you can now explain

A concrete, decomposed opinion on where a "fusion" gain in a real inference engine actually comes
from — launch elimination versus genuine overlap — rather than treating "fusing kernels helps" as
one undifferentiated claim.

Next: back to the compulsory exercises — [`../e9-mfma-trap/README.md`](../e9-mfma-trap/README.md) / [`../e10-sentinel-sync/README.md`](../e10-sentinel-sync/README.md)
