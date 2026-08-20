# E3 — `v_dot2_f32_f16` and the instruction-mix question

**Depends on:** E2
**Priority:** recommended

## 1. Objective

Halve the VALU instruction count on the multiply-accumulate by packing FP16 pairs and using the
packed dot-product-2 instruction, and use the result to answer a question you can only answer by
measuring: is VALU throughput anywhere close to a co-bottleneck at batch 1, or is it irrelevant?

## 2. Concept primer: two multiplies, one instruction

```
 Scalar path (what E2's loop compiles to, roughly):
   lane holds W[c], W[c+1] and x[c], x[c+1] as separate FP16 scalars
   acc += (float)W[c]   * (float)x[c]      ← v_fma  #1
   acc += (float)W[c+1] * (float)x[c+1]    ← v_fma  #2
   → 2 VALU instructions for 2 multiply-adds

 Packed path:
   lane holds W[c:c+1] and x[c:c+1] each as ONE packed FP16x2 register
   acc = v_dot2_f32_f16(W_packed, x_packed, acc)
   → 1 VALU instruction for the SAME 2 multiply-adds
```

Bytes moved from memory are **unchanged** — you're still reading the same W and x. Only the
arithmetic path changes. That's the point of this exercise: it isolates whether arithmetic
throughput was ever the limiter, independent of the memory-bandwidth question E0-E2 already
answered.

## 3. Step-by-step build

### Step 3.1 — files **[paste this]**

Copy `wave_gemv_dpp` from E2 into `dot2_gemv.hip` as your starting point (paste it — it's the
proven baseline, not the new material).

### Step 3.2 — pack and dot **[type this]**

```cpp
// Instead of reading two half scalars and doing two v_fma's, read one
// packed FP16x2 value from each of W and x, and use the packed dot product:
half2 w_pair = *reinterpret_cast<const half2*>(&W[(size_t)row * cols + c]);
half2 x_pair = *reinterpret_cast<const half2*>(&x_shared[c]);
acc = __builtin_amdgcn_fdot2(w_pair, x_pair, acc, false);
// (exact builtin name/signature varies by ROCm version — check what's
// available; some versions expose this via a HIP-level intrinsic instead
// of the raw __builtin_amdgcn_* name. If your compiler declines to form
// v_dot2_f32_f16 from ordinary scalar code, this explicit builtin is the
// fallback — see Step 3.4.)
```

`false` in the signature above is typically a "clamp" flag — check the actual signature for your
ROCm version rather than copying this blind; builtin signatures for packed dot products have
changed across ROCm releases.

### Step 3.3 — measure **[paste this]**

```bash
make run
```

Expect GB/s to be **unchanged or nearly unchanged** from E2's DPP variant, since you haven't
touched the memory access pattern at all.

### Step 3.4 — verify in the ISA — this is the actual exercise **[paste this, then read it]**

```bash
make isa
grep "v_dot2_f32_f16" dot2_gemv.isa.txt
```

**If that grep comes back empty, you have measured nothing.** Compilers frequently decline to form
packed dot products from scalar-looking source, even when you wrote `half2` types — the pattern
match required to recognise "these two multiplies are a dot product" is narrower than you'd expect.
If it didn't fire:
1. Try the explicit builtin from Step 3.2 instead of writing scalar-looking code and hoping the
   compiler recognises the pattern.
2. If the builtin also doesn't produce `v_dot2_f32_f16` in the disassembly, reach for inline asm.
3. Either way — **write down in your findings that you checked and what you found.** "I checked and
   the compiler had not done what I asked" is a more valuable sentence than a speedup number
   obtained without checking.

Also check instruction counts directly:

```bash
rocprof -i <(echo "pmc: SQ_INSTS_VALU") -o e2_valu.csv ./  # run against the E2 binary
rocprof -i <(echo "pmc: SQ_INSTS_VALU") -o e3_valu.csv ./  # run against this E3 binary
```

Expect `SQ_INSTS_VALU` for E3 to be roughly half of E2's.

## 4. Acceptance criteria

- [ ] `v_dot2_f32_f16` confirmed present in the disassembly (or an honest account of why it wasn't,
      and what you did about it).
- [ ] `SQ_INSTS_VALU` roughly halved versus E2.
- [ ] GB/s essentially unchanged versus E2.
- [ ] An explicit written conclusion: **at batch 1 the arithmetic path is not the constraint** —
      which is precisely why halving it changed the instruction count but not the wall-clock time.
      State this plainly; it's a stronger finding than a speedup would have been, because it rules
      out a whole category of future "optimisation" that wouldn't have helped.

## 5. Failure modes

- **Claiming a speedup that isn't there.** If your GB/s number *did* improve meaningfully, look for
  a confound before believing it — a change in cache behaviour between runs, a different launch
  configuration, or accidentally comparing against E1 instead of E2's DPP variant.
- **Not checking the ISA and reporting the "halved instructions" claim anyway.** This is the
  specific trap this exercise is built to catch.

## 6. Checkpoint question

If VALU isn't the constraint at batch 1, at what batch size *would* you expect it to start
mattering — and what would you measure to find out?

## 7. What you can now explain

You have the empirical version of the answer this module builds toward more fully in E9: the
observation that a resource can be halved without changing performance, and what that tells you
about which resource actually gates this kernel.

Next: [`../e4-rows-per-wave/README.md`](../e4-rows-per-wave/README.md)
