# E6 — XCD awareness (optional)

**Depends on:** E2, module 06 (`06-xcd-awareness/`)
**Priority:** optional — cut this one first if the week runs long.

## 1. Objective

Test whether the physical placement of weight rows relative to XCD assignment moves the GEMV
number. Module 06 in this repo already established how to identify which XCD a block landed on and
how swizzling can improve GEMM cache behaviour; this exercise asks the same question of a
bandwidth-bound GEMV instead of a compute-bound GEMM.

## 2. Concept primer: 8 XCDs, 4 IODs, and where the bytes actually live

```
 MI300X: 8 XCDs (compute), 4 IODs (memory controllers + HBM stacks)

  ┌─────────────────────────────────────────────────────────────┐
  │  IOD 0        │  IOD 1        │  IOD 2        │  IOD 3       │
  │  ┌────┐┌────┐ │  ┌────┐┌────┐ │  ┌────┐┌────┐ │  ┌────┐┌────┐│
  │  │XCD0││XCD1│ │  │XCD2││XCD3│ │  │XCD4││XCD5│ │  │XCD6││XCD7││
  │  └────┘└────┘ │  └────┘└────┘ │  └────┘└────┘ │  └────┘└────┘│
  │  2×HBM stacks │  2×HBM stacks │  2×HBM stacks │  2×HBM stacks│
  └─────────────────────────────────────────────────────────────┘

 Default: blocks distributed round-robin across XCDs. A block on XCD 0
 reading a weight row physically homed on IOD 3's HBM crosses the
 interconnect to get it — module 06's "NUMA problem," now asked of a
 bandwidth-bound kernel instead of a cache-reuse one.

 This exercise: partition the weight matrix so the rows a given XCD's
 blocks consume are placed in memory attached to that XCD's own IOD,
 and compare against default (unmanaged) placement.
```

## 3. Step-by-step build

### Step 3.1 — reuse module 06's XCD identification **[paste this]**

`06-xcd-awareness/06a-xcd-discovery/xcd_discovery.cpp` already establishes how to determine which
XCD a running block landed on. Reuse that mechanism rather than rederiving it.

### Step 3.2 — control placement **[type this]**

```cpp
// TODO(E6): allocate the weight matrix in row-partitioned chunks, one
// hipMalloc per XCD's share, and (NPS-mode dependent — see failure modes)
// bias the physical placement toward that XCD's IOD. The exact placement
// control available depends on the partition's NPS (NUMA-per-socket) mode;
// state which mode you're on before drawing conclusions.
```

### Step 3.3 — measure, default vs placed **[type this]**

Run the E2-style wave-per-row GEMV against both the default (round-robin, unmanaged) allocation and
the XCD-matched allocation. Report MBU for both.

## 4. Acceptance criteria

- [ ] MBU measured for both default and placed configurations.
- [ ] **Either** a real, explained effect, **or** a null result with an explanation. A null result
      here is a legitimate, publishable finding: it says that at this working-set size and access
      pattern, the interconnect was not the limiter — not that the experiment failed.
- [ ] The NPS mode of the partition stated explicitly, since it affects what "controlled placement"
      even means on this hardware.

## 5. Failure modes

- **Believing you controlled placement when you didn't.** The allocator and NPS mode both interfere
  with naive attempts to pin memory to a specific IOD — verify placement actually took effect
  (e.g. via `hipMemAdvise`/topology queries) rather than assuming your `hipMalloc` pattern achieved
  what you intended.
- **Drawing a strong conclusion from a working set much smaller than what would make cross-IOD
  traffic dominant.** Scale matters here — say what scale you tested at.

## 6. Checkpoint question

At what point (working-set size, or fraction of traffic crossing IODs) would you expect placement
to start mattering for a bandwidth-bound kernel, even if it doesn't at the scale you tested?

## 7. What you can now explain

Whether XCD/IOD placement matters for a batch-1 GEMV at the scale you tested, backed by a
measurement either way — and, if null, why a null result here is still informative rather than a
non-finding.

Next: [`../e7-fused-gate-up/README.md`](../e7-fused-gate-up/README.md) (optional), or back to the
compulsory exercises: [`../e9-mfma-trap/README.md`](../e9-mfma-trap/README.md) / [`../e10-sentinel-sync/README.md`](../e10-sentinel-sync/README.md)
