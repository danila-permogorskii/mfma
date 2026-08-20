# Batch-1 GEMV on MI300X — Findings

*Copy this file (e.g. to `FINDINGS.md`) and fill it in as you complete each exercise. Sections
follow the order this module's exercises produce evidence in — methodology first, then one shared
results table, then per-exercise narrative, then limitations and open questions.*

---

## 1. Methodology

*(Summarise in your own words — don't just paste `00-measurement-methodology.md`. What matters
here is demonstrating you understand why the two-regime, buffer-rotation, median+IQR discipline
exists, not reproducing the document.)*

- Cache-cold / cache-warm definitions used:
- Buffer rotation approach (count, total size):
- Timing method (device-side timestamp region, iteration count, discard count):
- MBU denominator (from E0), and how it compares to the datasheet figure:

## 2. Results table

| Exercise | Configuration | GB/s (cold) | GB/s (warm) | MBU (cold) | MBU (warm) | VGPRs | Occupancy |
|---|---|---|---|---|---|---|---|
| E0 | best sweep point | | | — | — | — | — |
| E1 | naive | | | | | — | — |
| E2 | DPP | | | | | | |
| E2 | LDS tree | | | | | | |
| E3 | dot2 | | | | | | |
| E4 | R=1 | | | | | | |
| E4 | R=2 | | | | | | |
| E4 | R=4 | | | | | | |
| E4 | R=8 | | | | | | |
| E4 | R=16 | | | | | | |
| E5 | weights alone, hinted | | | | | | |
| E5 | weights alone, unhinted | | | | | | |
| E5 | +resident, hinted | | | | | | |
| E5 | +resident, unhinted | | | | | | |
| E6 | default placement | | | | | — | — |
| E6 | XCD-matched placement | | | | | — | — |
| E7 | (a) two launches | | | — | — | — | — |
| E7 | (b) fused sequential | | | — | — | — | — |
| E7 | (c) fused interleaved | | | — | — | — | — |
| E9 | VALU (E2/E4 baseline) | | | | | | |
| E9 | MFMA padded | | | | | | |
| E10 | naive barrier (latency) | — | — | — | — | — | — |
| E10 | grid sync (latency) | — | — | — | — | — | — |
| E10 | NaN sentinel (latency) | — | — | — | — | — | — |

*(Add rows for anything this table doesn't fit — e.g. the E10 straggler experiment probably
deserves its own small table rather than cramming into this one.)*

## 3. Per-exercise notes

*(One paragraph each — what limited it, what the disassembly showed. Keep these short; the table
above carries the numbers, these carry the reasoning.)*

### E0 — bandwidth ceiling


### E1 — naive GEMV


### E2 — wave-per-row reduction


### E3 — packed dot product


### E4 — rows per wave


### E5 — non-temporal hints


### E6 — XCD placement *(if attempted)*


### E7 — fused gate/up *(if attempted)*


### E9 — MFMA trap


### E10 — sentinel synchronisation


## 4. Limitations

*(State these yourself, plainly — understatement reads better than polish here, and every
limitation named up front is one that can't be sprung on you later.)*

- Single operations tested in isolation, not a full decode pipeline.
- One architecture (gfx942/MI300X); results may not transfer to CDNA2 or RDNA.
- One work period; not re-run across multiple sessions or machines for variance.
- No multi-GPU / tensor-parallel measurements.
- *(add anything else genuinely true of your run)*

## 5. Open questions

*(End here, not on a conclusion — a one-week artefact reads better closing on questions than on
claims.)*

- The MALL/MBU denominator question: how much of a real decode kernel's weight stream is actually
  served from Infinity Cache rather than HBM, given that individual per-layer weight matrices can
  be smaller than the 256 MB cache? What's the right way to report MBU when that's true?
- The NaN-sentinel hazard from E10: which mitigation is actually used in a production
  implementation of this technique, and why that one over the alternatives?
- *(add your own — the exercises will surface more than these two)*
