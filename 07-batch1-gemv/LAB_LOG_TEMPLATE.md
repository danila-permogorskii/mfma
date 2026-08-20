# Lab log — Module 07 batch-1 GEMV

*Copy this file (e.g. to `LAB_LOG.md`) and append one entry per work session. This is the record of
how you got to the results in `FINDINGS.md` — warts, dead ends, and wrong turns included. It's
meant to look different from the findings document: rougher, chronological, honest about what
didn't work.*

---

## Entry format

Copy this block for each session:

```
## YYYY-MM-DD — <what you worked on>

**Started with:** <where you left off / what you're attempting>

**Did:**
- ...

**Measured:**
- ...

**Broke / didn't work:**
- ...

**Learned:**
- ...

**Next:**
- ...
```

---

## Example entry (delete once you have real ones)

```
## 2026-08-20 — E0 setup, first sweep attempt

**Started with:** fresh clone, no prior work on this module.

**Did:**
- Built bw_ceiling.hip with the dword variant from the guide, typed by hand.
- Wrote dwordx2/dwordx4 variants myself, following the same shape.
- Wrote the sweep driver for width x waves/CU x cached/NT.

**Measured:**
- dword: ~X GB/s cache-cold. dwordx4: ~Y GB/s cache-cold. (fill in real numbers)

**Broke / didn't work:**
- First sweep run showed >5.3 TB/s effective bandwidth — forgot to size
  RotatingBuffers past 512MB, was measuring MALL not HBM. Fixed by bumping
  buffer count.
- dwordx4 disassembly showed dwordx2 instructions instead — pointer wasn't
  16-byte aligned. Fixed with an explicit alignment attribute.

**Learned:**
- The "it ran suspiciously fast" instinct from 00-measurement-methodology.md
  is not theoretical — it happened on the first real run.

**Next:**
- Finish the waves/CU axis of the sweep, then move to E1.
```
