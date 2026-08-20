/**
 * =============================================================================
 * Module 07 — shared helpers for the batch-1 GEMV exercises
 * =============================================================================
 *
 * PURPOSE:
 *   Every exercise from E0 to E10 needs the same four things: a way to check
 *   HIP return codes, a way to defeat the Infinity Cache (MALL) so a benchmark
 *   loop measures HBM instead of a 256 MB cache, a way to read a device-side
 *   timestamp from inside a persistent kernel, and a way to turn a pile of
 *   iteration times into a median + IQR instead of a single misleading mean.
 *
 *   Modules 01–06 in this repo duplicate HIP_CHECK in every file, which is
 *   fine at 5 standalone demos. At 10 exercises sharing identical
 *   buffer-rotation and timing needs, duplicating it 10 times would just be
 *   10 places to fix the same bug — this header exists instead. It is this
 *   module's ONE deliberate exception to the "everything inline" convention
 *   used elsewhere in the repo.
 *
 *   Read 00-measurement-methodology.md before using this file — it explains
 *   *why* each helper below exists, not just what it does.
 *
 * WHAT IS DELIBERATELY NOT HERE:
 *   The GEMV kernel itself. Every exercise's kernel is the point of that
 *   exercise and is written by hand in that exercise's .hip file, following
 *   its README. This header is boilerplate only.
 *
 * BUILD:
 *   This is a header-only file. Exercises include it with:
 *     #include "../common/gemv_common.hpp"
 *   and compile with -I../common if they don't use a relative include.
 *
 * =============================================================================
 */

#pragma once

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

// -----------------------------------------------------------------------------
// HIP_CHECK — same shape as the copy in every other module (04, 05, 06...),
// just factored out so it's defined once for this module's 10 exercises.
// -----------------------------------------------------------------------------
#define HIP_CHECK(call)                                                                                              \
    do {                                                                                                             \
        hipError_t err = call;                                                                                       \
        if (err != hipSuccess) {                                                                                     \
            printf("HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__);                          \
            exit(1);                                                                                                 \
        }                                                                                                            \
    } while (0)

// -----------------------------------------------------------------------------
// device_timestamp() — a device-side cycle-ish counter you can call from
// inside a kernel, before and after a region of interest, without paying for
// a kernel launch or a host round-trip. See 00-measurement-methodology.md,
// "Timing method", for why host-side hipEvent_t timing (as used in modules
// 04/05) cannot see intra-kernel stages and why E9/E10 in particular need
// this instead.
//
// __builtin_amdgcn_s_memrealtime() maps to s_memrealtime, a free-running
// real-time counter (fixed frequency, independent of engine clock). Prefer
// it over s_memtime (engine-clock ticks) when comparing durations across
// kernels that might run at different clocks.
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint64_t device_timestamp() {
    return __builtin_amdgcn_s_memrealtime();
}

// -----------------------------------------------------------------------------
// RotatingBuffers<T> — the cache-cold defence from the methodology doc,
// implemented once. Allocates `count` device copies of a buffer of `elems`
// elements each, sized so the *total* comfortably exceeds the 256 MB MALL
// (aim for > 512 MB across all copies). Call next() once per iteration in
// your benchmark loop so iteration i never reads the bytes iteration i-1
// just made cache-resident.
//
// This does NOT initialise the buffers with meaningful data beyond a fixed
// pattern — for exercises that need specific weight/activation values,
// initialise each of the `count` copies identically after construction.
// -----------------------------------------------------------------------------
template <typename T>
class RotatingBuffers {
public:
    RotatingBuffers(size_t elems, int count) : elems_(elems), count_(count), idx_(0) {
        ptrs_.resize(count_);
        for (int i = 0; i < count_; ++i) {
            HIP_CHECK(hipMalloc(&ptrs_[i], elems_ * sizeof(T)));
        }
    }

    ~RotatingBuffers() {
        for (auto p : ptrs_) hipFree(p);
    }

    // Returns the next buffer in rotation and advances the cursor. Call this
    // exactly once per benchmark iteration, immediately before launching the
    // kernel that reads it.
    T *next() {
        T *p = ptrs_[idx_];
        idx_ = (idx_ + 1) % count_;
        return p;
    }

    T *at(int i) const { return ptrs_[i]; }
    int count() const { return count_; }
    size_t bytes_total() const { return elems_ * sizeof(T) * count_; }

private:
    size_t elems_;
    int count_;
    int idx_;
    std::vector<T *> ptrs_;
};

// -----------------------------------------------------------------------------
// median_iqr — "median of >= 100 iterations, first 10 discarded, plus IQR"
// is the reporting standard set in 00-measurement-methodology.md. Implemented
// once here instead of once per exercise. `samples` is modified (sorted) by
// this call.
// -----------------------------------------------------------------------------
struct MedianIqr {
    double median;
    double q1;
    double q3;
};

inline MedianIqr median_iqr(std::vector<double> &samples, int discard_first = 10) {
    if ((int) samples.size() > discard_first) {
        samples.erase(samples.begin(), samples.begin() + discard_first);
    }
    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();
    auto pick = [&](double frac) {
        size_t i = (size_t) (frac * (double) (n - 1));
        return samples[i];
    };
    MedianIqr r;
    r.median = pick(0.50);
    r.q1 = pick(0.25);
    r.q3 = pick(0.75);
    return r;
}
