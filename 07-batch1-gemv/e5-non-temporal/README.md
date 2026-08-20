# E5 — Non-temporal hints, or: weights are read once

**Depends on:** E2, and conceptually E0
**Priority:** recommended

## 1. Objective

At batch 1, every weight is read once, used once, and never touched again this token. Nothing
about that access pattern benefits from being cached. Worse: caching it anyway *evicts* things that
genuinely are reused — the KV cache, ongoing activations. This exercise applies a non-temporal hint
to weight loads specifically, and measures the effect on a *co-resident* tenant of the cache, not
on the weight stream itself.

## 2. Concept primer: the hint protects the other tenant, not the streamer

```
 WITHOUT non-temporal hint:

  ┌───────────────────────────────────────────────────────────┐
  │  Cache (MALL)                                              │
  │  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
  │  │ KV cache   │  │ weight     │  │ weight     │  ← streamed │
  │  │ (reused    │  │ chunk N    │  │ chunk N+1  │    once,    │
  │  │  every     │  │ (never     │  │ evicts     │    evicts   │
  │  │  token)    │◄─┤  read      │◄─┤  KV cache  │    useful   │
  │  │            │  │  again)    │  │  lines!    │    data     │
  │  └────────────┘  └────────────┘  └────────────┘            │
  └───────────────────────────────────────────────────────────┘
   Every weight chunk that passes through competes for the SAME
   cache lines the KV cache needs — and wins some of that fight,
   even though the weight chunk itself gets zero benefit from it.

 WITH non-temporal hint on weight loads only:

  ┌───────────────────────────────────────────────────────────┐
  │  Cache (MALL)                                              │
  │  ┌────────────┐                    ┌ ─ ─ ─ ─ ─ ┐           │
  │  │ KV cache   │   weight chunks    │  (weight   │           │
  │  │ (stays     │   flow THROUGH     │  chunks    │           │
  │  │  resident, │   without being     │  bypass or │           │
  │  │  never     │   installed as       │  are      │           │
  │  │  evicted   │   long-lived         │  evicted   │           │
  │  │  by        │   entries            │  first)   │           │
  │  │  weights)  │                    └ ─ ─ ─ ─ ─ ┘           │
  │  └────────────┘                                            │
  └───────────────────────────────────────────────────────────┘
   KV cache's hit rate goes UP, even though nothing changed about
   how the KV cache itself is accessed. The win is second-order.
```

This framing — the hint protects the *other* tenant — is the correct one, and it's easy to get
backwards: measuring only the weight stream in isolation (no co-resident data) will show little or
no effect, because there's nothing else competing for those cache lines in that measurement. The
real experiment needs a synthetic KV-cache stand-in.

## 3. Step-by-step build

### Step 3.1 — files **[paste this]**

Start from E2's `wave_gemv_dpp`. Add a second buffer representing a synthetic "resident" working
set — a few tens of MB, re-read every iteration, standing in for a KV cache.

### Step 3.2 — apply the hint to weight loads only **[type this]**

On CDNA3 this is expressed via scope bits (`sc0`/`sc1`) on the load instruction, or via a
compiler builtin/intrinsic that sets them — check the ISA guide's section on cache/scope control
for the exact spelling in your ROCm version rather than copying syntax written for gfx90a
(MI200-era); the bit layout and available builtins have changed across generations.

```cpp
// Weight load: hinted non-temporal (exact intrinsic/builtin per your ROCm
// version's ISA guide — do not guess the bit pattern from memory).
half w = load_nontemporal(&W[(size_t)row * cols + c]);

// x load and the synthetic resident-set load: ordinary, cached, UNCHANGED.
half xv = x_shared[c];
float kv = resident_set[kv_index];
```

Leave `x` and the output alone — only the weight loads get the hint. This is deliberate: the point
is to see whether *not* installing weights protects something else, not to speed up the weight
stream itself.

### Step 3.3 — two configurations, measure both **[type this]**

**Configuration A — weights alone.** No competing resident data. Measure GB/s hinted vs unhinted.
Expect a small or null effect — there's nothing to protect yet.

**Configuration B — weights streaming alongside a resident working set.** Allocate `resident_set`
(a few tens of MB) and re-read it every iteration from every wave, interleaved with the weight
stream. Measure the resident set's own hit rate (via `rocprofv3`/L2 counters, as in module 06)
**with and without** the hint on the weight stream. This is the real experiment.

### Step 3.4 — verify the scope bits landed where intended **[paste this, then read it]**

```bash
make isa
grep -B2 -A2 "global_load" nt_gemv.isa.txt | grep -i "sc0\|sc1\|nt"
```

Confirm the scope bits are set on the weight-load instructions specifically, and *not* on the `x`
or `resident_set` loads — a hint applied everywhere by accident defeats the point of the
experiment.

## 4. Acceptance criteria

- [ ] Configuration A measured: weight-stream GB/s, hinted vs unhinted (expect little difference).
- [ ] Configuration B measured: resident-set hit rate, hinted-weights vs unhinted-weights (this is
      the number that should move).
- [ ] ISA confirmation that scope bits landed only on weight loads.
- [ ] A written conclusion framed correctly: the hint's value is in what it protects, not in making
      the weight stream itself faster.

## 5. Failure modes

- **Measuring only Configuration A and concluding "non-temporal hints don't matter here."** This is
  the single most common way to get this exercise backwards — see the concept primer above.
- **Hinting `x` or the output by accident**, which can slow things down for no interesting reason
  and muddies the measurement of what the hint is actually protecting.

## 6. Checkpoint question

Why might the *size* of the synthetic resident working set change how large an effect you measure —
and what would happen to the effect if the resident set were also larger than MALL itself?

## 7. What you can now explain

You have a precise, measured answer to "why do non-temporal loads matter when the kernel is already
bandwidth-saturated" — a question that sounds contradictory until you frame it around the tenant
being protected rather than the stream being hinted.

Next: [`../e9-mfma-trap/README.md`](../e9-mfma-trap/README.md) or [`../e10-sentinel-sync/README.md`](../e10-sentinel-sync/README.md) (both compulsory), or the optional [`../e6-xcd-placement/README.md`](../e6-xcd-placement/README.md) / [`../e7-fused-gate-up/README.md`](../e7-fused-gate-up/README.md)
