# Measurement methodology — read this before E0

Every exercise from here on reports a bandwidth number. If the methodology behind that number is
wrong, every exercise that follows inherits the mistake silently — the code can be correct and the
conclusion can still be false. This document is the one place that methodology is explained; every
exercise guide links back here instead of repeating it.

---

## 1. The trap: your GPU has a 256 MB cache you didn't ask for

MI300X sits **256 MB of Infinity Cache (MALL — Memory Attached Last Level cache)** between the
compute dies and HBM. It is transparent: nothing in your kernel source mentions it, nothing in the
HIP API exposes it directly, and a naive benchmark loop will happily run *faster* than HBM
bandwidth allows without erroring, warning, or otherwise telling you why.

Picture a typical benchmark loop:

```
for iteration in 1..100:
    launch kernel(weight_matrix)     # always the SAME weight_matrix pointer
    wait
    record elapsed time
```

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │  iteration 1: weight_matrix not in any cache                        │
 │                                                                      │
 │   Compute Die ──miss──▶ MALL ──miss──▶ HBM  (slow: real transfer)   │
 │                            │                                        │
 │                            └──── copy also lands in MALL ───────────┼──▶ now resident
 │                                                                      │
 │  iteration 2..100: SAME weight_matrix pointer, still in MALL        │
 │                                                                      │
 │   Compute Die ──HIT──▶ MALL   (fast: never touches HBM again)       │
 └─────────────────────────────────────────────────────────────────────┘
```

A 2048 × 8192 FP16 weight matrix is 32 MB — small enough that the *entire* matrix survives in a
256 MB cache. From iteration 2 onward you are measuring **MALL bandwidth**, not **HBM bandwidth**.
The two numbers can differ by a large factor, and your reported "MBU" (memory bandwidth
utilization) can quietly read as anything up to and past 100% without any single step in the
computation being wrong. Nothing crashes. Nothing warns you. You have to be the one looking for it.

---

## 2. Three defences — use all three, not just one

### Defence 1 — report two regimes, explicitly, always

Don't report one bandwidth number per exercise. Report two, labeled:

- **cache-cold**: working set exceeds 256 MB, or buffers are rotated so no iteration reuses the
  bytes the previous iteration touched. This is the number that reflects HBM.
- **cache-warm**: the same weight matrix reused every iteration, deliberately, so you can see and
  report how much MALL is doing for you.

Both numbers are real measurements of real things. They answer different questions. Reporting only
one — especially only the flattering cache-warm one — is the single most common way a GPU
micro-benchmark misleads its own author.

### Defence 2 — rotate buffers

`RotatingBuffers<T>` in `common/gemv_common.hpp` implements this once for the whole module:
allocate N device copies of the weight matrix, sized so the *total* across all N comfortably
exceeds 512 MB (not 256 MB — leave margin so partial residency doesn't muddy the result), and call
`.next()` once per benchmark iteration. Iteration *i* then reads bytes that iteration *i-1* never
touched, so MALL never gets a chance to serve iteration *i* from what iteration *i-1* left behind.

```cpp
RotatingBuffers<half> weights(rows * cols, /*count=*/24);  // 24 copies, well over 512 MB total
for (int i = 0; i < iterations; ++i) {
    half *w = weights.next();          // a buffer nothing has touched recently
    kernel<<<grid, block>>>(w, ...);
}
```

### Defence 3 — non-temporal hints on weight loads

Covered in full in E5. The short version: if a load is tagged non-temporal, the hardware is told
"don't bother keeping this around" — which is both a correctness-of-measurement tool (it makes
cache-cold behaviour easier to reproduce reliably) and, as E5 shows, a real optimisation in its own
right once something else is trying to stay resident in the same cache.

---

## 3. Timing method: device-side, not host-side

`05-mfma-gemm/mfma_gemm.cpp` in this repo times kernels with `hipEventRecord`/`hipEventElapsedTime`
— a host-side measurement that brackets an entire kernel launch. That's fine for a single
short kernel, but it cannot see *inside* a persistent kernel (E9, E10) that runs multiple logical
stages without returning to the host between them. For those, and as good practice generally in
this module, use a **device-side timestamp** around just the region of interest:

```cpp
uint64_t t0 = device_timestamp();
// ... region of interest ...
uint64_t t1 = device_timestamp();
// stash (t1 - t0) somewhere a host-visible buffer can read it later
```

`device_timestamp()` in `common/gemv_common.hpp` wraps `__builtin_amdgcn_s_memrealtime()` — a
free-running counter at a fixed real-time frequency, not tied to the engine's current clock, which
matters when you're comparing durations across differently-clocked runs.

**Reporting standard for this whole module:** run at least 100 iterations, discard the first 10
(cold-start / clock-ramp effects), and report the **median** and **interquartile range (IQR)** of
what's left — never a single mean. `median_iqr()` in `common/gemv_common.hpp` implements this once.
A single mean hides bimodal behaviour (e.g. half your iterations hit MALL and half don't) that the
median-plus-IQR pair makes visible as a wide spread instead of an average that describes neither
mode.

---

## 4. The MBU definition this module uses, stated once

> **MBU** = (bytes that must be read) ÷ (elapsed time) ÷ (bandwidth this specific GPU partition
> actually delivers, as measured in E0)

Two things this definition deliberately does *not* use:

- **Not the datasheet number.** MI300X's spec sheet says 5.3 TB/s. On a shared, virtualised, or
  SR-IOV partition, the bandwidth a single tenant can actually pull is frequently lower — sometimes
  substantially. E0 exists to measure the real denominator on the machine you're actually running
  on, once, so every later MBU number in this module divides by a number you obtained yourself
  rather than one you looked up.
- **Not a number without a stated regime.** Every MBU you report from here on should be labeled
  cache-cold or cache-warm per Defence 1 above. An MBU without that label is not wrong, exactly —
  it's just missing the information needed to know what it means.

---

## 5. What "verify in the ISA" means, mechanically

Several guides ask you to check that the compiler emitted what you intended. The general recipe:

```bash
amdclang++ --offload-arch=gfx942 -O3 -save-temps -o exe source.hip
```

`-save-temps` leaves behind `.ll` (LLVM IR), `.bc` (bitcode), and other intermediate files next to
your binary. For the actual GPU instruction stream, disassemble the binary itself:

```bash
llvm-objdump -d --offloading exe > exe.isa.txt
```

Then `grep` for the instruction or modifier you expect (e.g. `dwordx4`, `v_dot2_f32_f16`,
`dpp`, `v_mfma`). Each exercise guide tells you exactly what to search for and what "it didn't
fire" looks like in that search. This is not an optional nicety — several exercises (E3, E9) are
specifically about the compiler declining to do what the source code asked for, and the only way
to know that happened is to look.

---

## 6. Checklist before you start E0

- [ ] I understand why a benchmark that reuses one weight-matrix pointer for 100 iterations is
      measuring the wrong thing after iteration 1.
- [ ] I know where `RotatingBuffers<T>`, `device_timestamp()`, and `median_iqr()` live and what
      each one is for.
- [ ] I will report cache-cold and cache-warm as two separate, labeled numbers, not one.
- [ ] I will report median + IQR over ≥ 100 iterations (first 10 discarded), not a mean.

Once these are checked, move to `e0-bandwidth-ceiling/README.md`.
