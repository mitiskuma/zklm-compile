#include <metal_stdlib>
using namespace metal;

// Mersenne-31 field: p = 2^31 - 1
constant uint M31_P = 0x7FFFFFFF;

// M31 reduction: reduce a u64 product to [0, p)
inline uint m31_reduce(ulong x) {
    // x mod (2^31 - 1): split into high and low 31-bit chunks
    // x = hi * 2^31 + lo = hi + lo (mod p)
    uint lo = uint(x) & M31_P;
    uint hi = uint(x >> 31);
    uint sum = lo + hi;
    // If sum >= p, subtract p (same as clearing bit 31)
    return (sum >= M31_P) ? (sum - M31_P) : sum;
}

inline uint m31_add(uint a, uint b) {
    uint sum = a + b;
    return (sum >= M31_P) ? (sum - M31_P) : sum;
}

inline uint m31_sub(uint a, uint b) {
    return (a >= b) ? (a - b) : (M31_P - b + a);
}

inline uint m31_mul(uint a, uint b) {
    return m31_reduce(ulong(a) * ulong(b));
}

// ===== Sumcheck fold kernel =====
// f[j] = (1-r)*f[j] + r*f[j+half], for j in [0, half)
kernel void sumcheck_fold(
    device uint* f [[buffer(0)]],
    constant uint& half_n [[buffer(1)]],
    constant uint& r_val [[buffer(2)]],
    constant uint& one_minus_r [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < half_n) {
        uint lo = f[gid];
        uint hi = f[gid + half_n];
        f[gid] = m31_add(m31_mul(one_minus_r, lo), m31_mul(r_val, hi));
    }
}

// ===== Sumcheck reduce (product): compute partial sums of (s0, s1, s2) =====
// Each threadgroup reduces a chunk and writes 3 partial sums.
// s0 += f0*g0, s1 += f1*g1, s2 += (2*f1-f0)*(2*g1-g0)
kernel void sumcheck_reduce_product(
    device const uint* f [[buffer(0)]],
    device const uint* g [[buffer(1)]],
    device uint* partials [[buffer(2)]],
    constant uint& half_n [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup uint shared_s0[256];
    threadgroup uint shared_s1[256];
    threadgroup uint shared_s2[256];

    uint local_s0 = 0;
    uint local_s1 = 0;
    uint local_s2 = 0;

    if (gid < half_n) {
        uint f0 = f[gid];
        uint f1 = f[gid + half_n];
        uint g0 = g[gid];
        uint g1 = g[gid + half_n];

        local_s0 = m31_mul(f0, g0);
        local_s1 = m31_mul(f1, g1);

        // f2 = 2*f1 - f0, g2 = 2*g1 - g0
        uint f2 = m31_sub(m31_add(f1, f1), f0);
        uint g2 = m31_sub(m31_add(g1, g1), g0);
        local_s2 = m31_mul(f2, g2);
    }

    shared_s0[tid] = local_s0;
    shared_s1[tid] = local_s1;
    shared_s2[tid] = local_s2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_s0[tid] = m31_add(shared_s0[tid], shared_s0[tid + stride]);
            shared_s1[tid] = m31_add(shared_s1[tid], shared_s1[tid + stride]);
            shared_s2[tid] = m31_add(shared_s2[tid], shared_s2[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partials[tg_id * 3] = shared_s0[0];
        partials[tg_id * 3 + 1] = shared_s1[0];
        partials[tg_id * 3 + 2] = shared_s2[0];
    }
}

// ===== Sumcheck reduce (triple): compute partial sums of (s0, s1, s2, s3) =====
kernel void sumcheck_reduce_triple(
    device const uint* f [[buffer(0)]],
    device const uint* g [[buffer(1)]],
    device const uint* h [[buffer(2)]],
    device uint* partials [[buffer(3)]],
    constant uint& half_n [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup uint shared_s0[256];
    threadgroup uint shared_s1[256];
    threadgroup uint shared_s2[256];
    threadgroup uint shared_s3[256];

    uint local_s0 = 0;
    uint local_s1 = 0;
    uint local_s2 = 0;
    uint local_s3 = 0;

    if (gid < half_n) {
        uint f0 = f[gid];
        uint f1 = f[gid + half_n];
        uint g0 = g[gid];
        uint g1 = g[gid + half_n];
        uint h0 = h[gid];
        uint h1 = h[gid + half_n];

        local_s0 = m31_mul(m31_mul(f0, g0), h0);
        local_s1 = m31_mul(m31_mul(f1, g1), h1);

        uint f2 = m31_sub(m31_add(f1, f1), f0);
        uint g2 = m31_sub(m31_add(g1, g1), g0);
        uint h2 = m31_sub(m31_add(h1, h1), h0);
        local_s2 = m31_mul(m31_mul(f2, g2), h2);

        // t=3: 3*x1 - 2*x0
        uint three = 3;
        uint two = 2;
        uint f3 = m31_sub(m31_mul(three, f1), m31_mul(two, f0));
        uint g3 = m31_sub(m31_mul(three, g1), m31_mul(two, g0));
        uint h3 = m31_sub(m31_mul(three, h1), m31_mul(two, h0));
        local_s3 = m31_mul(m31_mul(f3, g3), h3);
    }

    shared_s0[tid] = local_s0;
    shared_s1[tid] = local_s1;
    shared_s2[tid] = local_s2;
    shared_s3[tid] = local_s3;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_s0[tid] = m31_add(shared_s0[tid], shared_s0[tid + stride]);
            shared_s1[tid] = m31_add(shared_s1[tid], shared_s1[tid + stride]);
            shared_s2[tid] = m31_add(shared_s2[tid], shared_s2[tid + stride]);
            shared_s3[tid] = m31_add(shared_s3[tid], shared_s3[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partials[tg_id * 4] = shared_s0[0];
        partials[tg_id * 4 + 1] = shared_s1[0];
        partials[tg_id * 4 + 2] = shared_s2[0];
        partials[tg_id * 4 + 3] = shared_s3[0];
    }
}

// ===== Batch M31 modular inverse: a^(p-2) via square-and-multiply =====
// p-2 = 2^31 - 3 = 0x7FFFFFFD
kernel void batch_m31_inverse(
    device uint* vals [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;

    uint a = vals[gid];
    if (a == 0) {
        vals[gid] = 0;
        return;
    }

    // Compute a^(p-2) where p-2 = 2^31 - 3
    // Binary of p-2: 0111...1101 (bit 1 is 0, bit 0 is 1, all others are 1)
    uint result = 1;
    uint base = a;

    // Bit 0: 1
    result = m31_mul(result, base);
    base = m31_mul(base, base);

    // Bit 1: 0 (skip multiply)
    base = m31_mul(base, base);

    // Bits 2..30: all 1
    for (int i = 2; i <= 30; i++) {
        result = m31_mul(result, base);
        base = m31_mul(base, base);
    }

    vals[gid] = result;
}

// ===== eq_evals butterfly: one level of the butterfly =====
// After this kernel, entries [0..2*populated) are filled.
// For j in [0..populated): evals[2j] = evals[j] * (1-ri), evals[2j+1] = evals[j] * ri
kernel void eq_evals_butterfly(
    device uint* evals [[buffer(0)]],
    constant uint& populated [[buffer(1)]],
    constant uint& ri [[buffer(2)]],
    constant uint& one_minus_ri [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= populated) return;

    // Process in reverse order to avoid overwriting — but since we dispatch
    // exactly `populated` threads and each reads evals[gid] and writes
    // evals[2*gid] and evals[2*gid+1], we need to process in reverse.
    // Use the trick: gid maps to j = populated - 1 - gid (reverse order).
    uint j = populated - 1 - gid;
    uint val = evals[j];
    evals[2 * j] = m31_mul(one_minus_ri, val);
    evals[2 * j + 1] = m31_mul(ri, val);
}
