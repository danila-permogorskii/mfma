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

You already have `e0-bandwidth-ceiling/bw_ceiling.hip` and `Makefile` as stubs. Open
`bw_ceiling.hip` — the `main()`, argument handling, and buffer setup via
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
type instead of `float`. Same loop shape, wider element type — **but the sink trick needs a change
once there's more than one component, or you will silently measure the wrong width:**

```cpp
__global__ void stream_read_dwordx4(const float4 *in, float *sink, size_t n_vec4) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float4 acc = {0, 0, 0, 0};
    for (size_t off = i; off < n_vec4; off += (size_t)gridDim.x * blockDim.x) {
        float4 v = in[off];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    float s = acc.x + acc.y + acc.z + acc.w;   // sum ALL components — see warning below
    if (s == -1.0f) sink[i] = s;
}
```

**Why the sum matters — a live example of "the compiler did something you didn't ask for."** If
you write `if (acc.x == -1.0f) sink[i] = acc.x;` instead — the same pattern that's correct for the
scalar `dword` kernel above — the compiler can prove `acc.y`, `acc.z`, and `acc.w` are never
observed anywhere, and eliminates them. Once those accumulations are gone, three of the four loaded
components are dead too, and `-O3` narrows the load from `dwordx4` down to reading only `.x` — you
end up compiling something that disassembles identically to the `dword` kernel, with no warning,
no error, just a suspiciously-similar number if you'd skipped Step 3.5. Summing every component
into the value that actually gets checked and stored is what forces the compiler to keep (and
therefore load) all of it. This is not a hypothetical — compiling the `if (acc.x == ...)` version
and disassembling it shows exactly one `global_load_dword`; the summed version shows
`global_load_dwordx4`. Check this yourself in Step 3.5 rather than taking it on faith either way.

For `dwordx3` (12 bytes/lane), HIP *does* have a built-in vector type — `float3`, same family as
`float2`/`float4` in `amd_hip_vector_types.h` — no inline asm required. Write
`stream_read_dwordx3` the same way, with `const float3 *in` and the same summed-sink pattern; the
compiler emits a genuine `global_load_dwordx3` for it. (An earlier draft of this guide claimed
`float3` didn't exist and asked you to reach for inline asm — it does exist and you don't need to;
that claim was wrong and has been corrected here.) This one is still optional if you're short on
time — the dword/dwordx2/dwordx4 sweep is informative on its own — but it's no longer harder to get
than the others.

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
amdclang++ --offload-arch=gfx942 -O3 -save-temps -o bw_ceiling bw_ceiling.hip -I../common
llvm-objdump -d --offloading bw_ceiling > bw_ceiling.isa.txt
grep -E "dwordx4|dwordx3|dwordx2|global_load" bw_ceiling.isa.txt
```

Confirm the width you intended survived — and check it once per kernel, not just once overall,
since each width can narrow for a different reason. Two distinct causes to check for, both real,
both silent:

1. **Alignment.** Compilers narrow `dwordx4` to two `dwordx2`s or four `dword`s when they can't
   prove 16-byte alignment on the pointer. Fix with an explicit alignment annotation on the
   pointer/allocation.
2. **Dead-component elimination** (Step 3.2's warning). If your sink only observes `.x`, the
   compiler narrows the load to just `.x`'s width regardless of alignment — no alignment fix will
   help here, because alignment was never the problem. Fix by summing all components into whatever
   the sink actually checks/stores.

If a width narrowed and you're not sure which cause it was, alignment failures usually still show
a load wider than `dword` (e.g. `dwordx4` narrowed to `dwordx2`), while dead-component elimination
typically narrows all the way to plain `dword` regardless of the original width — that asymmetry is
a useful first clue before you dig further.

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

- **The loads got eliminated entirely.** Symptom: suspiciously fast, suspiciously flat timing
  across widths, with no `global_load` at all in the disassembly for that kernel. The sink didn't
  keep anything alive; make the "almost never true" condition depend on the accumulated state, or
  write to a `volatile` sink unconditionally on one thread per block.
- **The loads got *narrowed*, not eliminated — the width-specific version of the above.** Symptom:
  a `dwordx4` (or `dwordx2`/`dwordx3`) kernel disassembles with a plain `global_load_dword`, and its
  timing suspiciously matches the scalar `dword` kernel's. This is the Step 3.2/3.5 dead-component
  trap: the sink observed only one component (typically `.x`), so the compiler proved the rest dead
  and shrank the load to match. Distinct from full elimination — you still get *a* load, just the
  wrong width — and the fix is different too: sum every component into whatever the sink actually
  checks and stores, not just watch for a missing load instruction.
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
