# E4 — Rows per wave, and the register wall

**Depends on:** E2
**Priority:** compulsory

## 1. Objective

Process R output rows per wave instead of one, reusing the *same* load of `x` across all R rows.
Sweep R and find the point where more reuse stops helping because the registers needed to hold R
accumulators start crowding out the occupancy that was hiding HBM latency in the first place. That
crossover point — the "knee" — is the register-pressure story this exercise produces.

## 2. Concept primer: reuse of the small operand, and the wall it hits

```
 E2: one wave, one row. x is read once per wave, W is read once per wave. No reuse.

 E4: one wave, R rows. x is read ONCE and reused across all R rows — this is the
 only reuse available at batch 1 (there's only one x; there's no batching to
 amortise W across). Cuts activation traffic and per-row instruction overhead.
 Does NOT reduce weight traffic — every element of W is still read exactly once.

   lane holds: acc[0] (row 0's partial sum)
               acc[1] (row 1's partial sum)
               ...
               acc[R-1]
   for each chunk of x (loaded ONCE):
       for r in 0..R:  acc[r] += W[row_base+r][c] * x_shared[c]

 More R = more accumulator registers held live simultaneously = fewer waves
 fit per SIMD = less latency-hiding capacity:

   R=1:  VGPRs/wave low  → many waves resident → HBM latency well hidden
         SIMD: [wave][wave][wave][wave][wave][wave][wave][wave]  (8 waves)

   R=8:  VGPRs/wave high → few waves resident  → latency exposed, or SPILLS
         SIMD: [ wave ][ wave ]                                  (2 waves)
                    ↑ if accumulators don't fit even at 2 waves,
                      the allocator spills to scratch — visible in the ISA
                      as store/load instructions inside your inner loop,
                      and fatal to performance.

               MBU
                │        ___----‾‾‾--___
                │      /                ‾--__        ← the knee: MBU rises
                │    /                        ‾--_      with R while occupancy
                │  /                               ‾-_   still covers latency,
                │/                                     ‾  then falls once it
                └───────────────────────────────────────► R    doesn't.
                  1    2    4    8    16
```

## 3. Step-by-step build

### Step 3.1 — files **[paste this]**

Start from E2's `wave_gemv_dpp`. The block/wave structure and `x`-staging logic carry over
unchanged.

### Step 3.2 — the R-rows-per-wave kernel **[type this]**

```cpp
template <int R>
__global__ void rows_per_wave_gemv(const half *W, const half *x, float *y, int rows, int cols, int row_base) {
    int lane = threadIdx.x;
    extern __shared__ half x_shared[];
    for (int c = lane; c < cols; c += 64) x_shared[c] = x[c];
    __syncthreads();

    float acc[R];
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;

    for (int c = lane; c < cols; c += 64) {
        half xv = x_shared[c];              // loaded ONCE, reused R times below
        for (int r = 0; r < R; ++r) {
            int row = row_base + blockIdx.x * R + r;
            acc[r] += (float)W[(size_t)row * cols + c] * (float)xv;
        }
    }

    // TODO(E4): DPP-reduce each of the R accumulators (R independent
    // reductions, or interleave them — try both and see which the compiler
    // schedules better).
}
```

Templating on `R` (rather than a runtime loop bound) lets the compiler allocate exactly the right
number of accumulator registers per configuration and lets you compile one binary per R for a
clean comparison — instantiate `rows_per_wave_gemv<1>`, `<2>`, `<4>`, `<8>`, `<16>` explicitly.

### Step 3.3 — the sweep **[type this]**

Sweep R ∈ {1, 2, 4, 8, 16} and, orthogonally, the unroll depth (elements of the row consumed per
loop iteration before the next `s_waitcnt`). Record, for each point: MBU, VGPR count (from the
compiler's own output, see Step 3.4), and achieved occupancy (from `rocprof-compute` or
`rocprofv3`).

### Step 3.4 — read the register allocation **[paste this, then read it]**

```bash
amdclang++ --offload-arch=gfx942 -O3 -Rpass-analysis=kernel-resource-usage \
  -c -o /dev/null rows_per_wave_gemv.hip -I../common 2>&1 | grep -A5 "rows_per_wave_gemv"
```

This prints VGPR/SGPR/AGPR counts and occupancy estimate per kernel instantiation without a full
link — useful for sweeping R quickly. Cross-check with `make isa` and search for `scratch` in the
disassembly:

```bash
make isa
grep -i scratch rows_per_wave_gemv.isa.txt
```

Any `scratch` load/store inside the per-element loop (not just at kernel entry/exit) means the
allocator spilled — that R value is past the wall, not just slower.

## 4. Acceptance criteria

- [ ] A table or plot: R vs MBU, with VGPR count and measured occupancy at each point.
- [ ] A visible knee — MBU rising, then falling — with the R at the knee named explicitly.
- [ ] Confirmation (from the ISA, not inference) of where spilling starts, if it does within your
      sweep range.
- [ ] An explanation of the mechanism at the knee, not just its location: fewer resident waves ⇒
      less latency hidden ⇒ HBM round-trips become visible in the timing.

## 5. Failure modes

- **Concluding "more rows is always better" from a cache-warm measurement.** Cache-warm hides the
  latency-hiding story entirely, since MALL's latency is far lower than HBM's — this exercise's
  finding only shows up cache-cold. Report both regimes but draw the knee conclusion from
  cache-cold only.
- **Ignoring the occupancy drop.** Going from 8 waves/SIMD to 2 waves/SIMD isn't just "fewer
  threads" — it's a direct reduction in the machine's ability to hide the exact latency this whole
  module is fighting. Say so explicitly rather than treating occupancy as a side detail.

## 6. Checkpoint question

Why does this reuse only help with the *activation* traffic (x) and not the *weight* traffic (W) —
and why does that asymmetry matter more at batch 1 than it would at, say, batch 32?

## 7. What you can now explain

You have a register-pressure story with a named knee and ISA evidence for where spilling begins —
not "occupancy matters" as received wisdom, but a specific R value on this specific kernel where it
started to.

Next: [`../e5-non-temporal/README.md`](../e5-non-temporal/README.md) (recommended), or continue to
[`../e9-mfma-trap/README.md`](../e9-mfma-trap/README.md) / [`../e10-sentinel-sync/README.md`](../e10-sentinel-sync/README.md) (compulsory)
