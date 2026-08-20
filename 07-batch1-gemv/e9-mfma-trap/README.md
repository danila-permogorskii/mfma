# E9 — The MFMA trap, demonstrated rather than asserted

**Depends on:** E2, E4
**Priority:** compulsory

## 1. Objective

You've spent modules 04–05 of this repo building MFMA (matrix-core) kernels. The natural instinct
when facing a new matrix operation is to reach for them again. This exercise proves, by
measurement rather than assertion, that matrix cores are the wrong instrument for batch-1 GEMV —
and, just as importantly, states *precisely* why, because the sloppy version of this conclusion
("MFMA is slower here") is wrong and the precise version ("MFMA is pointless here, for a specific
reason") is not the same claim.

## 2. Concept primer: filling a 16×16 tile with one real row

```
 v_mfma_f32_16x16x16_f16 wants a 16-row batch of activations to multiply
 against a 16×16 tile of weights. At batch 1, you have ONE row of real data:

  ┌────────────────────────────────────────┐
  │ row 0:  [ real activation data ]        │  ← the only row that matters
  │ row 1:  [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ] │
  │ row 2:  [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ] │  ← 15 rows of padding,
  │  ...                                    │     computed anyway, because
  │ row 15: [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ] │     the instruction operates
  └────────────────────────────────────────┘     on the whole 16×16 tile

  15/16 of the matrix core's issued work produces zeros you already knew.
```

Critically: **padding does not increase weight traffic.** The weight matrix W is read exactly
once either way — MFMA or the E2/E4 VALU kernel. Since this whole module has been establishing that
weight traffic, not arithmetic, is what gates batch-1 GEMV, the honest prediction *before you
measure* is: comparable wall-clock time, because both kernels are bound by the same bytes. If MFMA
comes back dramatically slower, that's a signal you've measured something else (a cache-warm run,
a bad tiling) — not evidence that matrix cores are inherently bad here.

## 3. Step-by-step build

### Step 3.1 — files **[paste this]**

Reuse module 04/05's MFMA vector types (`v4f16`, `v4f32`, etc.) and intrinsic-call pattern as your
starting point for the padded-batch kernel; reuse E2's `wave_gemv_dpp` as the comparison baseline.

### Step 3.2 — the padded-batch MFMA kernel **[type this]**

```cpp
// TODO(E9): implement the same projection as a GEMM with the batch
// dimension padded from 1 to 16, using v_mfma_f32_16x16x16_f16 tiles.
// Only row 0 of the 16-row activation tile holds real data; rows 1-15
// are zero. Follow module 04/05's MFMA vector-type conventions
// (ext_vector_type, NOT manual bit packing) for A/B/C matrix layout.
```

### Step 3.3 — measure both, identical data **[paste this]**

```bash
make run
```

Compare against your E2/E4 VALU kernel result on the *same* weight matrix and activation vector.

### Step 3.4 — verify in the ISA and via counters **[paste this, then read it]**

```bash
make isa
grep v_mfma mfma_trap_gemv.isa.txt
rocprof -i <(echo "pmc: SQ_INSTS_MFMA, SQ_INSTS_VALU") -o e9.csv ./mfma_trap_gemv
```

The important number here isn't time or even `SQ_INSTS_MFMA` — it's **bytes actually moved**.
Confirm (from your own accounting of the kernel's memory traffic, not from the instruction count)
that the padding didn't add weight-matrix reads. If it did — check your tiling; that's a bug, not
a finding.

### Step 3.5 — check the second-order effects **[paste this]**

Look at VGPR/AGPR usage (the accumulator tiles for MFMA live in AGPRs) and occupancy, the same way
you did in E4. These are the real costs MFMA imposes here — not arithmetic slowness, but register
and issue-slot pressure spent on a resource (arithmetic throughput) that was already idle.

## 4. Acceptance criteria

- [ ] Time and MBU measured for both kernels, same data, same regime.
- [ ] `SQ_INSTS_MFMA` and bytes-moved both reported, with bytes-moved confirmed unchanged by padding.
- [ ] The conclusion stated **precisely**: MFMA is not slower here in any interesting sense — the
      likely result is comparable time, because both kernels are bandwidth-bound on the same bytes.
      MFMA is *pointless* here, not *bad* here — it buys arithmetic throughput the kernel doesn't
      need, at a real (if secondary) cost in registers and occupancy.
- [ ] The second-order costs (register pressure from accumulator tiles, any occupancy loss) named
      explicitly, not glossed over.

## 5. Failure modes

- **Overclaiming.** If your measurement shows MFMA as catastrophically slower, you have very likely
  measured a cache-warm run, a bad tiling, or an unfair comparison (e.g. different working-set
  sizes between the two kernels) — go find the confound before writing down the conclusion, not
  after.
- **Conflating "pointless" with "bad."** These are different claims with different implications for
  when you *would* reach for MFMA (batch sizes large enough that arithmetic actually becomes the
  constraint — see the checkpoint question).

## 6. Checkpoint question

At what batch size would you expect the crossover — the point where MFMA stops being pointless and
starts being the right instrument? What would you measure to find that crossover experimentally
rather than guess it?

## 7. What you can now explain

A precise, measured answer to "how would you use the matrix cores here?" that resists the tempting
but wrong instinct to reach for the tool you know best — backed by a number, not an assertion.

Next: [`../e10-sentinel-sync/README.md`](../e10-sentinel-sync/README.md)
