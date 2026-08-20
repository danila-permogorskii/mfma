# Kog Labs published articles — analysis and mapping to this module

*Standalone background document. Nothing in `README.md` or any `eN-*/README.md` links here or
assumes you've read it — the exercises stand on their own technical merits. This document exists
separately because you asked for an independent assessment of what a specific company (Kog Labs)
has actually published, and how it relates to the problems this module exercises, kept apart from
the teaching material itself.*

Researched via web search and direct fetch of the source pages on 2026-08-20. Quotes below are as
extracted from the live pages on that date; blog content can change after publication, so treat
this as a snapshot, not a permanent record of what those URLs say.

---

## 1. What they published

Source: ["Building a single-kernel, latency-optimized LLM inference engine on AMD MI300X
GPUs"](https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/),
Kog Labs blog (`blog.kog.ai`), plus a secondary check of ["Kog Reaches 3.5x Breakthrough Inference
Speed on AMD Instinct MI300X
GPUs"](https://blog.kog.ai/kog-reaches-3-5x-breakthrough-inference-speed-on-amd-instinct-mi300x-gpus/).

**Performance headline.** 3,000+ output tokens/s per request on 8× MI300X, FP16, batch size 1, for
a 2B-parameter model ("Laneformer" in the post's terminology) — no speculative decoding. Stated
comparison: 2,100 tokens/s on 8× H200 for the same setup; typical decode speed elsewhere cited as
100–300 tokens/s for 2B–8B models. The second post separately claims "up to 3.5× faster token
generation than vLLM and TensorRT-LLM" and "cross-GPU latency down to 4μs" — the latter number is
close to, but not confirmed identical to, the 4.5 µs kernel-launch figure below; the two posts
don't cross-reference precisely enough to say if it's the same measurement described two ways.

**Synchronisation.** Their sentinel-based scheme: **0.80–0.93 µs** (max end − max start across 256
CUs). A naive atomic-counter barrier over the same workload: **7.59–7.88 µs**. They state grid
synchronisation represents **~35% of total token generation time** for the naive approach — a
figure worth treating as approximate rather than load-bearing, since the post doesn't fully specify
what's held constant across that percentage claim.

**Kernel launch overhead.** **4.5 µs** for launch and cleanup on MI300X, plus a separately-stated
**~0.5 µs** HBM-latency cost when memory loads restart at the beginning of a new kernel.

**Hardware utilisation.** 256 of MI300X's 304 CUs used (`gridDim=(256,)`, `blockDim=(64,8)`).
Topology stated as 8 XCDs × 38 CUs, 4 IODs × 2 HBM stacks × 2 XCDs each — matches the standard
public MI300X topology description also used in this repo's `06-xcd-awareness/README.md`.

**Weight streaming.** Non-temporal loads via the "NT scope bit on the load instructions," stated
purpose "deprioritize storing weights" (i.e., don't let single-use weight reads evict things worth
keeping). W1/W3 (gate/up) FFN tensors are "pre-loaded before entering the FFN section" — a
prefetch-across-a-dependency pattern. They state ~80% of layer weights are dedicated to FFN in
their model, which is why prefetching that specific dependency matters more than others.

**Arithmetic path at batch 1.** They use `dot2` (scalar-ALU packed dot product) rather than matrix
cores for the batch-1 projections, with partial results "accumulated in FP32, reduced within the
wave using Data Parallel Primitive (DPP) operations." This is stated as their actual production
choice, not a comparison against an MFMA alternative — they don't report having measured MFMA at
batch 1 and rejected it; they simply built the `dot2`+DPP path directly.

**Recomputation over communication.** Their attention implementation duplicates certain tensors per
IOD rather than communicating them cross-IOD, to reduce interconnect traffic — a different
trade-off than the placement question E6 asks (E6 asks about placement of an existing single copy;
they describe avoiding the need for cross-IOD access at all via duplication).

---

## 2. Mapping table — their claim to this module's exercise

| Kog Labs claim | This module | Relationship |
|---|---|---|
| Sentinel sync 0.80–0.93 µs vs barrier 7.59–7.88 µs | E10 | E10 asks you to reproduce the *mechanism* and measure your own three numbers; comparing your numbers against theirs is optional and left to you (see E10's README note on why the guide itself doesn't hand you their figures as a target). |
| `dot2` + DPP reduction for batch-1 projections, no matrix cores | E2 (DPP reduction), E3 (`v_dot2_f32_f16`), E9 (MFMA-is-pointless finding) | Their production choice is exactly the structure E2/E3 build and E9 independently justifies by measurement rather than by citing their design. |
| NT-scope weight loads, "deprioritize storing weights" | E5 | Same mechanism (scope bits on weight loads); E5's framing — the hint protects co-resident data, not the weight stream itself — is a more precise claim than the post states, and worth checking your own E5 result against. |
| W1/W3 FFN prefetch during attention | E10's folded-in "E8" idea (prefetch across a dependency) | Same pattern: issue known-in-advance loads before the producing stage completes. Their reason (80% of weights are FFN) is a model-specific justification for *which* dependency is worth prefetching — a question E10 doesn't ask you to answer, since it depends on a real model's weight distribution, not a synthetic benchmark. |
| 8 XCD / 4 IOD topology, 256/304 CUs used | E6 | Same topology this repo's `06-xcd-awareness/README.md` already covers; their CU-count choice (256 of 304) isn't explained in the post and isn't something E6 asks you to reproduce — E6 asks about weight *placement*, not CU *count*. |
| 4.5 µs launch overhead | E7 | E7 asks you to measure your own reference figure for launch overhead on your own hardware rather than hand you theirs — useful as a sanity-check range once you have your own number, not as a target. |
| Duplicate-per-IOD instead of cross-IOD communication for attention tensors | (not directly exercised) | A different technique than E6 tests (E6: placement of one copy; this: avoiding the need for one shared copy at all). Worth noting as a technique this module doesn't currently have an exercise for. |

---

## 3. Where the posts are silent, or leave a real question open

- **No mention of Infinity Cache / MALL anywhere in either post checked.** Confirmed by direct
  fetch of both. This matters because `00-measurement-methodology.md` in this module spends real
  space on the MALL cache-cold/cache-warm distinction, and the posts give no visibility into
  whether their reported bandwidth/MBU-style figures account for it. Individual per-layer weight
  matrices in a model their size can be well under MI300X's 256 MB MALL — meaning some fraction of
  their "batch-1 is bandwidth-bound" story could, in principle, be partially served from cache
  rather than HBM on repeated layers/tokens, and the posts don't state whether their methodology
  controls for that. This is exactly the open question `FINDINGS_TEMPLATE.md` seeds under "MALL
  denominator question" — it isn't answerable from what's published, only from your own E0/E5
  measurements plus a genuine question to whoever built the system.
- **No stated MBU formula or denominator.** They report throughput (tokens/s) and latency
  components (µs), not a bandwidth-utilisation percentage with a stated denominator — so there's
  nothing to directly cross-check against this module's `00-measurement-methodology.md` §4
  definition. This isn't a criticism of the posts (tokens/s is arguably the more meaningful
  end-to-end metric); it just means "how does their MBU compare to mine" isn't a question their
  public writing answers.
- **No comparison of `dot2`+DPP against MFMA at batch 1.** They state their choice, not a rejected
  alternative with numbers — E9 in this module produces that comparison independently; it isn't
  something their posts hand you.
- **The "~35% of total token generation time" figure for grid-sync overhead** isn't accompanied by
  enough detail (model size, sequence length, what's held constant) to be directly reproducible —
  treat it as an order-of-magnitude claim, not a target to match.

---

## 4. Sources

- Kog Labs, "Building a single-kernel, latency-optimized LLM inference engine on AMD MI300X GPUs."
  https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/
- Kog Labs, "Kog Reaches 3.5x Breakthrough Inference Speed on AMD Instinct MI300X GPUs."
  https://blog.kog.ai/kog-reaches-3-5x-breakthrough-inference-speed-on-amd-instinct-mi300x-gpus/
- Kog Labs blog index (for any further posts not individually checked here): https://blog.kog.ai/
- Cross-posted/aggregated coverage found during search (not independently verified against the
  primary source, listed for completeness only): AMD's own blog republishing the "3.5x" post
  (amd.com/en/blogs/2025/kog-reaches-3-5x-breakthrough-inference-speed-on-amd-instinct-mi.html),
  and a DEV Community summary (dev.to/creeta/kog-hits-3k-ts-on-mi300x-no-kernel-switches-test-it-now-55dh).
