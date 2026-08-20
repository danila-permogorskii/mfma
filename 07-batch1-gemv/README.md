# Experiment 07: Batch-1 GEMV — Memory-Bandwidth-Bound Decode Kernels

## Overview

LLM autoregressive decoding at batch size 1 generates one token at a time, and every layer's
projection is a matrix-vector product (GEMV), not a matrix-matrix product (GEMM): a weight matrix
of shape `[rows × cols]` is read once, in full, to produce a single output vector. At this batch
size the arithmetic is trivial and the byte count is not — decode is bound by how fast you can
stream weights off HBM, not by how fast you can multiply. This module builds that understanding
from the ground up: a bad baseline, the fix, the reasoning for the fix, and then the harder
questions (register pressure, cache behaviour, chiplet topology, kernel launch overhead,
cross-block synchronisation) that a real batch-1 decode engine has to answer.

This module follows the format used elsewhere in this repo (see the top-level `README.md`), but
with ten linked exercises instead of one file — closer in shape to `06-xcd-awareness/`, extended.
**Type the kernel bodies and reductions yourself; the boilerplate around them is fine to paste.**
See "How to use these guides" below for exactly where that line falls in each exercise.

---

## Prerequisites

- Completed experiments 01–06 in this repo (HIP basics, wavefronts, LDS, MFMA, XCD awareness).
- Access to an MI300X (gfx942). Every exercise assumes gfx942; flag it if you're on a different
  CDNA3/CDNA2 part, since a few instructions (packed dot products, DPP modifiers) vary by
  generation.
- `amdclang++`, `rocprofv3` (or `rocprof`), and `llvm-objdump` on `PATH`. Every build command in
  this module invokes `amdclang++` directly rather than the `hipcc` wrapper script used in modules
  01–06 — same underlying compiler, no functional difference. Source files use the `.hip`
  extension (not `.cpp`, unlike modules 01–06) specifically so `amdclang++` auto-detects HIP
  language from the extension alone — no `-x hip` flag needed anywhere in this module.
- **Read [`00-measurement-methodology.md`](00-measurement-methodology.md) before starting E0.** It
  is not optional background reading — E0's acceptance criteria depend on it.

### Toolchain note for this machine

This module was written and verified against the ROCm **7.14** stack (`amdclang++` reports
`AMD clang version 23.0.0git`, `Found HIP installation: .../core-7.14, version 7.14.60850`),
installed alongside an older `rocm-core` 7.0.2 base. If `amdclang++ ...` fails with
`'hip/hip_runtime.h' file not found` even though the compiler itself is found, the `-dev` package
with 7.14's own headers is likely missing (only the runtime library was installed, not the
headers). Fix with:

```bash
apt-get install amdrocm-runtime-dev7.14
```

This pulls in `amdrocm-llvm-dev7.14` as well and installs cleanly alongside the existing 7.0.2
base (no removals or upgrades of anything already installed). Confirm it worked with:

```bash
amdclang++ --offload-arch=gfx942 -O3 -c -o /dev/null e0-bandwidth-ceiling/bw_ceiling.hip -I common
```

---

## Directory map

```
07-batch1-gemv/
├── README.md                       ← you are here
├── 00-measurement-methodology.md   ← REQUIRED READING before E0
├── FINDINGS_TEMPLATE.md            ← copy this, fill it in as you complete exercises
├── LAB_LOG_TEMPLATE.md             ← copy this, append one entry per work session
├── common/
│   └── gemv_common.hpp             ← HIP_CHECK, RotatingBuffers<T>, device_timestamp(), median_iqr()
├── e0-bandwidth-ceiling/           ← compulsory — establishes the denominator for every MBU number
├── e1-naive-gemv/                  ← compulsory — the bad baseline
├── e2-wave-reduction/              ← compulsory — the correct baseline structure
├── e3-packed-dot/                  ← recommended — v_dot2_f32_f16, and why it changes nothing here
├── e4-rows-per-wave/               ← compulsory — the register-pressure story
├── e5-non-temporal/                ← recommended — protecting co-resident data, not the weight stream
├── e6-xcd-placement/               ← optional, cut first if the week runs long
├── e7-fused-gate-up/               ← optional, cut first if the week runs long
├── e9-mfma-trap/                   ← compulsory — matrix cores are the wrong instrument here
└── e10-sentinel-sync/              ← compulsory — sentinel synchronisation vs a naive barrier
    (there is no e8/ — E8's idea folds directly into E10's guide, see that README)
```

Each `eN-*/` directory contains:

```
eN-topic/
├── README.md          ← the guide: objective, concept, step-by-step build, acceptance criteria
├── <name>.hip           ← stub: working main()/scaffolding, kernel body left as TODO(E<n>)
└── Makefile             ← all / run / profile / profile-v3 / debug / clean / help
```

---

## Priority

| Exercise | Priority | Depends on |
|---|---|---|
| E0 — bandwidth ceiling | **compulsory** | `00-measurement-methodology.md` |
| E1 — naive GEMV | **compulsory** | E0 |
| E2 — wave-per-row + reduction | **compulsory** | E1 |
| E3 — packed dot product | recommended | E2 |
| E4 — rows per wave | **compulsory** | E2 |
| E5 — non-temporal hints | recommended | E2, and conceptually E0 |
| E6 — XCD placement | optional (cut first) | E2, module 06 |
| E7 — fused gate/up projection | optional (cut first) | E2 |
| E9 — the MFMA trap | **compulsory** | E2, E4 |
| E10 — sentinel synchronisation | **compulsory** | E2 |

Work through them roughly in this order — E2 is the shared foundation almost everything after it
builds on.

---

## How to use these guides

Every guide labels each code block **[type this]** or **[paste this]**:

- **[type this]** — the load-bearing lines: the kernel body, the reduction, the synchronisation
  primitive. This is where the actual learning happens; typing it (and making the small mistakes
  that come with typing it) is the point.
- **[paste this]** — boilerplate that's identical in spirit to something you've already typed once
  in an earlier exercise: argument parsing, buffer setup, `main()` scaffolding. Re-typing this
  every time teaches you nothing new; paste it and spend the time on the interesting part instead.

Every guide follows the same seven-part shape: objective, a concept primer with an ASCII
schematic, the step-by-step build, acceptance criteria, failure modes, a checkpoint question, and
what you can now explain going into the next exercise. The checkpoint questions are there so you
notice, before moving on, whether you actually understood the mechanism or just got the numbers to
come out — answer them for yourself (in the lab log is a good place) before continuing.

---

## Quick start

```bash
cd ~/mfma/07-batch1-gemv

# Confirm your GPU and toolchain first
rocminfo | grep -E "Name:.*gfx"        # expect gfx942
amdclang++ --version

# Read the methodology doc, then start on E0
cd e0-bandwidth-ceiling
cat README.md
make && make run
```

---

## The two documents you produce

This module's deliverable isn't the code — the exercises are how you get there. It's two files,
started from the templates in this directory and filled in as you go:

- **`FINDINGS_TEMPLATE.md`** → becomes your results write-up: methodology first, one results
  table, one paragraph per exercise on what limited it, a limitations section, and open questions.
- **`LAB_LOG_TEMPLATE.md`** → a running, dated log of what you tried, kept separate from the
  findings write-up on purpose — the findings document is the polished conclusion, the lab log is
  the record of how you got there, warts included.

---

## References

- AMD Instinct MI300 ISA Manual — the authoritative reference for the instructions and scope bits
  used throughout this module (DPP, `v_dot2_f32_f16`, `v_mfma_f32_16x16x16_f16`, memory scope
  bits). Same document referenced by modules 04–06 of this repo.
- `00-measurement-methodology.md` (this directory) — MALL/Infinity Cache trap, timing method, MBU
  definition.
- `06-xcd-awareness/README.md` (this repo) — XCD/IOD topology background used again in E6.
