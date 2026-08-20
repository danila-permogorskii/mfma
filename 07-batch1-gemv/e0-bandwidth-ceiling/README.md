# E0 — The ceiling: what does this machine actually give me?

**Depends on:** [`00-measurement-methodology.md`](../00-measurement-methodology.md) (read it first)
**Priority:** compulsory

## 1. Objective

Every exercise after this one reports a percentage — "MBU: 61%" — and a percentage is only as
honest as its denominator. This exercise measures that denominator: the actual sustained HBM
bandwidth this specific GPU (and this specific partition, if it's shared or virtualised) delivers,
not the number on the spec sheet. Once you have it, you defend it with a sweep, not a citation.

## 2. Concept primer: why the load width and wave count both matter

A single wavefront issuing one 4-byte load per lane per instruction cannot saturate HBM — there
just aren't enough bytes in flight. Two independent levers widen the pipe:

```
 Lever 1 — bytes per instruction (load width)
 ┌──────────────────────────────────────────────────────────┐
 │  dword    : 4 bytes/lane   × 64 lanes = 256 B/instruction │
 │  dwordx2  : 8 bytes/lane   × 64 lanes = 512 B/instruction │
 │  dwordx4  : 16 bytes/lane  × 64 lanes = 1024 B/instruction│
 └──────────────────────────────────────────────────────────┘

 Lever 2 — waves in flight per CU (hides latency behind more work)
 ┌──────────────────────────────────────────────────────────┐
 │  1 wave/CU :  latency of a load is fully exposed          │
 │              CU idle │████░░░░░░░░░░│ idle │████░░░...    │
 │                       load issued    waiting  data back   │
 │                                                            │
 │  8 waves/CU: while wave A waits on HBM, waves B..H issue  │
 │              their own loads — the CU is never idle       │
 │              CU busy  │████████████████████████████│      │
 └──────────────────────────────────────────────────────────┘
```

Both levers matter because HBM has a fixed number of outstanding-request slots and a fixed
per-request latency. Too few in-flight bytes (either because loads are narrow or because too few
waves are resident) and the memory controller sits idle waiting for the next request, even though
HBM itself could deliver far more. The sweep in this exercise finds the point where both levers
stop mattering — that plateau is your ceiling.

## 3. Step-by-step build

### Step 3.1 — directory and files **[paste this]**

You already have `e0-bandwidth-ceiling/bw_ceiling.cpp` and `Makefile` as stubs. Open
`bw_ceiling.cpp` — the `main()`, argument handling, and buffer setup via
`RotatingBuffers<float4>` from `common/gemv_common.hpp` are already written. What's missing is the
kernel itself and the sweep driver, both marked `TODO(E0)`.

### Step 3.2 — the streaming-read kernel **[type this]**

The kernel does no useful arithmetic — its only job is to issue loads and prevent the compiler
from deleting them, since a load whose result is never used is dead code and *will* be eliminated
at `-O3`. A cheap sink (write one value to a tiny output buffer) keeps the loads alive without
adding meaningful extra traffic:

```cpp
// dword (4B/lane) variant
__global__ void stream_read_dword(const float *in, float *sink, size_t n_floats) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (size_t off = i; off < n_floats; off += (size_t)gridDim.x * blockDim.x) {
        acc += in[off];              // one 4-byte load per iteration
    }
    if (acc == -1.0f) sink[i] = acc; // branch almost never taken; keeps `acc` live
}
```

The `if (acc == -1.0f)` trick is a standard "convince the compiler this matters" idiom: `acc` can
never legitimately equal exactly `-1.0f` given the input data, so the store almost never executes,
but the compiler cannot prove that at compile time and must keep the accumulation around. Verify
this worked in Step 3.5 — don't take it on faith.

Now write the `dwordx2` and `dwordx4` variants the same way, using `float2`/`float4` as the load
type instead of `float`. Same loop shape, wider element type:

```cpp
__global__ void stream_read_dwordx4(const float4 *in, float *sink, size_t n_vec4) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float4 acc = {0, 0, 0, 0};
    for (size_t off = i; off < n_vec4; off += (size_t)gridDim.x * blockDim.x) {
        float4 v = in[off];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == -1.0f) sink[i] = acc.x;
}
```

For `dwordx3` (12 bytes/lane) HIP has no built-in vector type — reproduce it with inline asm if you
want the full sweep; it's the same trick the monokernel-style engines use to get an odd-width load
the compiler wouldn't generate on its own. This one is optional; the width sweep is informative
even without it.

### Step 3.3 — the non-temporal variant **[type this]**

Same loop, but hint the load as non-temporal so the read doesn't pollute cache on the way through.
On CDNA3 this is expressed via a builtin/intrinsic or inline-asm scope bit rather than a keyword —
check the ISA manual's section on cache/scope bits for the exact builtin name available in your
ROCm version, since this has changed across generations; don't copy syntax from a gfx90a
(MI200-era) example.

### Step 3.4 — the sweep driver **[type this]**

```cpp
// TODO(E0): for each combination of
//   width      in { dword, dwordx2, dwordx4 (, dwordx3) }
//   waves/CU   in { 1, 2, 4, 8 }             // control via blockDim / grid size
//   cached vs NT
// run >= 110 iterations through RotatingBuffers<...>(elems, count) sized so
// total > 512 MB, discard the first 10, and record achieved GB/s using
// device_timestamp() bracketing the kernel region (see 00-measurement-methodology.md §3).
// Print one line per configuration: width, waves/CU, cached/NT, median GB/s, IQR.
```

Waves per CU is controlled indirectly: pick a block size and grid size such that the number of
concurrently resident blocks per CU corresponds to the wave count you want to test (a block of 256
threads = 4 wavefronts of 64; residency depends on register/LDS usage too, so confirm the actual
resident wave count with `rocprofv3` rather than assuming your launch configuration guarantees it).

### Step 3.5 — verify in the ISA **[paste this, then read it]**

```bash
amdclang++ -x hip --offload-arch=gfx942 -O3 -save-temps -o bw_ceiling bw_ceiling.cpp -I../common
llvm-objdump -d --offloading bw_ceiling > bw_ceiling.isa.txt
grep -E "dwordx4|dwordx2|global_load" bw_ceiling.isa.txt
```

Confirm the width you intended survived. Compilers routinely narrow `dwordx4` to two `dwordx2`s or
four `dword`s when they can't prove 16-byte alignment on the pointer — if your `dwordx4` kernel's
disassembly doesn't show `dwordx4` loads, that's why, and the fix is usually an explicit alignment
annotation on the pointer or the allocation.

### Step 3.6 — run and profile **[paste this]**

```bash
make run
make profile-v3
```

## 4. Acceptance criteria

- [ ] A sweep table: width × waves/CU × cached/NT → median GB/s (IQR reported).
- [ ] The configuration that maximises achieved GB/s, named explicitly.
- [ ] Confirmation from the ISA that the intended load width actually survived compilation for
      each width you tested.
- [ ] One number you're willing to write down as "the bandwidth this partition delivers" — this
      becomes the denominator for every MBU in every later exercise.
- [ ] If this number is below the 5.3 TB/s datasheet figure (likely, especially on a shared or
      SR-IOV partition), say so explicitly and by how much.

## 5. Failure modes

- **The loads got eliminated.** Symptom: suspiciously fast, suspiciously flat timing across widths.
  Check the ISA — if the load instructions aren't there at all, the sink trick didn't work; make
  the "almost never true" condition depend on more of the accumulated state, or write to a
  `volatile` sink unconditionally on one thread per block.
- **Too few waves in flight.** Symptom: bandwidth well below what wider loads should achieve.
  Check resident wave count with `rocprofv3`, not just your launch configuration.
- **First-touch page-fault cost included in the timing.** Symptom: iteration 1 is an outlier.
  This is exactly what "discard the first 10 iterations" in the methodology doc handles — confirm
  you're actually discarding them.

## 6. Checkpoint question

Before moving to E1: if you doubled the grid size without changing block size or per-block
resource usage, would you expect bandwidth to keep rising, plateau, or fall — and why, in terms of
in-flight requests versus the memory controller's actual capacity?

## 7. What you can now explain

You have a defensible number for "the bandwidth this machine gives me," obtained by sweeping, not
quoted from a spec sheet — and you know which lever (width or occupancy) was the binding one on
this partition. Every MBU percentage from E1 onward divides by this number.

Next: [`../e1-naive-gemv/README.md`](../e1-naive-gemv/README.md)
