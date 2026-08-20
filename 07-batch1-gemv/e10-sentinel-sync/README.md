# E10 — Sentinel synchronisation versus a naive barrier

**Depends on:** E2
**Priority:** compulsory
**Note:** this guide also covers the idea originally split out as "E8" (prefetching across a
dependency) — implementing it is the natural first step before the three-scheme comparison below,
since sentinel synchronisation *is* how you make a safely-overlapped prefetch correct.

## 1. Objective

A persistent, multi-stage kernel needs blocks to signal each other when a value is ready, without
returning to the host between stages. This exercise builds three different ways to do that signal —
a naive global barrier, a cooperative-groups grid sync, and a data-is-the-flag "sentinel" scheme —
and measures the latency and straggler-sensitivity of each. It also reproduces, in miniature, the
prefetch-across-a-dependency idea: issuing stage 2's weight loads before stage 1 has finished,
which only works safely if the synchronisation between the stages is precise enough not to either
stall unnecessarily or read a value before it's actually ready.

## 2. Concept primer: three ways to say "the data is ready"

```
 (a) Naive global barrier — atomic counter + spin:

   producer:  write data; atomicAdd(&counter, 1)
   consumer:  while (atomicLoad(&counter) < expected) { }   // spins on a COUNTER
   → every block waits for EVERY other block, regardless of whether it
     actually depends on it. One straggler delays everyone.

 (b) Cooperative-groups grid sync — the framework-supplied version of (a):

   grid.sync();
   → same "everyone waits for everyone" semantics, different implementation.

 (c) NaN sentinel — the data IS the flag:

   producer:  buffer[i] = real_value;             // ordinary store
   consumer:  while (isnan(buffer[i])) { }         // polls the DATA itself
              value = buffer[i];                    // no separate flag ever existed
   → consumer initialised buffer[i] to NaN before the producer runs. No
     atomic counter, no separate flag write, no peer poll of shared state —
     just "keep reading this address until it's not NaN anymore."
     Per-dependency (only waits on what it actually needs) and per-CU
     (doesn't require agreement across the whole grid).

  Timeline comparison (illustrative, not to scale):

   (a)/(b)  producer0 ──write──┐
            producer1 ──write────┐         consumer must wait for
            producer2 ────write────┐  ◄──  the SLOWEST producer,
            producerN ──────write────┐     even if it only needed
                                       ▼    producer0's value
                                  consumer proceeds (after ALL)

   (c)      producer0 ──write──┐
                                 ▼
                            consumer0 proceeds (as soon as ITS value is ready,
                                                 independent of producer1..N)
```

The correctness of (c) rests on a specific hardware guarantee: **vector memory operations of the
same type retire in program order** on CDNA3, so a later value cannot become visible to another
wave before an earlier value the same lane wrote. Read that section of the ISA guide before writing
the kernel — this is a correctness argument, not an optimisation detail, and it's worth being able
to cite precisely.

## 3. Step-by-step build

### Step 3.1 — files **[paste this]**

`sentinel_sync.cpp` stub sets up a persistent kernel with a producer/consumer dependency between
blocks (e.g. block *i* produces a value block *i+1* consumes) and buffer allocation for all three
schemes.

### Step 3.2 — scheme (a): naive barrier **[type this]**

```cpp
__device__ int counter = 0;
// producer: write payload; __threadfence(); atomicAdd(&counter, 1);
// consumer: while (atomicAdd(&counter, 0) < expected_count) { }
```

### Step 3.3 — scheme (b): cooperative groups **[type this]**

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
// grid.sync() inside a kernel launched via hipLaunchCooperativeKernel
```

### Step 3.4 — scheme (c): NaN sentinel **[type this]**

```cpp
// consumer buffer initialised to NaN before the producer stage runs
// producer: buffer[idx] = real_value;   // ordinary store, no fence needed
//           given CDNA3's same-type retirement-order guarantee — cite the
//           ISA guide section here in your own README/findings, the way
//           this repo's asm-experiments track cites section numbers.
// consumer: while (isnan(buffer[idx])) { /* LICM hazard — see Step 3.6 */ }
//           float value = buffer[idx];
```

**Prefetch across the dependency (the "E8" idea):** once scheme (c) works, issue stage 2's weight
loads (into registers or LDS) *before* polling confirms stage 1 is done, so the load latency
overlaps with the wait rather than following it. Measure the overlap directly: total time with
early issue vs. issuing the load only after the poll succeeds. Whatever limits how much you can
issue early — almost certainly available register or LDS capacity — name it explicitly; that's the
finding this sub-step produces.

### Step 3.5 — cross-chiplet visibility: scope bits **[type this]**

The producer's store and the consumer's load both need the correct memory **scope** for the
sentinel to be visible across XCDs — a workgroup-scoped store is invisible to a consumer on a
different XCD, and the kernel will appear to work by accident if you happen to test with blocks
that land on the same XCD. Use device scope explicitly; check the ISA guide's scope-bit section for
the exact builtin/intrinsic in your ROCm version.

### Step 3.6 — the LICM hazard **[paste this, then read it]**

```bash
make isa
grep -B3 -A3 "isnan\|global_load" sentinel_sync.isa.txt
```

A compiler's loop-invariant-code-motion pass will *try* to hoist the poll load out of the loop —
after all, from the compiler's local view, nothing in the loop body writes `buffer[idx]`. A hoisted
poll reads the value once and then spins forever comparing a stale register against itself. Force
the load to remain in the loop (a `volatile` qualifier on the pointer, or an explicit inline-asm
load) and confirm in the disassembly that the load instruction is actually inside the loop body,
not before it.

### Step 3.7 — measure all three, plus the straggler case **[type this]**

Measure per-dependency latency (median + IQR, per `00-measurement-methodology.md`) for each scheme
at several block counts. Then run the straggler experiment: artificially delay one producer block
(e.g. a busy-wait of known length before it writes) and measure how each scheme's latency responds.
Scheme (c), being per-dependency, should degrade only for the consumer that actually depends on the
delayed producer; schemes (a)/(b) should degrade globally.

## 4. Acceptance criteria

- [ ] Median + IQR latency for all three schemes, at several block counts.
- [ ] The straggler experiment: one producer delayed, effect on each scheme measured and compared —
      this is arguably the more interesting result of the three.
- [ ] ISA confirmation that the poll load was not hoisted by LICM (ordinary or volatile-forced, but
      verified, not assumed).
- [ ] Confirmation that device-scope bits are set on both the producer's store and the consumer's
      load, with a note on what happens if you test only within one XCD (the bug that looks like
      success).
- [ ] A paragraph on the NaN hazard: a genuine NaN in the real data stream is indistinguishable from
      "not yet written," and the consumer would wait forever. Discuss the options — constraining
      the arithmetic so NaN cannot arise, encoding readiness in a payload bit instead of the full
      value, or choosing a sentinel value that's illegal for that specific tensor's numeric range —
      and state which you chose and why, as an open question in your findings document.

## 5. Failure modes

- **LICM eating the poll.** Symptom: the kernel hangs. Check the ISA before assuming it's a
  correctness bug elsewhere.
- **Missing device-scope bits, working "by accident" on a single XCD.** Test explicitly with blocks
  forced onto different XCDs (reuse module 06's XCD-identification approach) before trusting a
  passing single-XCD test.
- **Too few blocks for the naive barrier's cost to be visible.** The straggler effect and the raw
  barrier cost both need enough concurrent blocks to show up meaningfully — a 2-block test won't
  distinguish these schemes well.

## 6. Checkpoint question

Why is scheme (c) "per-dependency and per-CU rather than global" a structural property of the
scheme, not just an implementation detail — what would you have to add to schemes (a) or (b) to get
the same property, and what would that cost?

## 7. What you can now explain

Three measured numbers for the cost of telling another block "the data is ready," an explanation of
*why* the cheapest of the three is safe (the retirement-order guarantee, cited precisely), and a
reasoned position on the NaN-sentinel hazard rather than just an awareness that it exists.

This is the last exercise in the module. Return to [`../README.md`](../README.md) to fill in
`FINDINGS_TEMPLATE.md` and `LAB_LOG_TEMPLATE.md`.
