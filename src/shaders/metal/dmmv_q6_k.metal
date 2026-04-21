#include <metal_stdlib>
using namespace metal;

static inline float f16le_to_f32_q6(uchar lo, uchar hi) {
    const ushort bits = ushort(lo) | (ushort(hi) << 8);
    return float(as_type<half>(bits));
}

static inline int unpack_q6(uchar lower, uchar upper) {
    return int(lower | (upper << 4u)) - 32;
}

kernel void dmmv_q6_k(device const uchar *weights [[buffer(0)]],
                      device const float *vector [[buffer(1)]],
                      device float *output [[buffer(2)]],
                      constant uint &row_stride_bytes [[buffer(3)]],
                      constant uint &cols [[buffer(4)]],
                      constant uint &rows [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= rows) {
        return;
    }

    const device uchar *row = weights + (ulong(gid) * ulong(row_stride_bytes));
    const uint block_count = cols / 256u;
    float sum = 0.0f;

    for (uint block_idx = 0; block_idx < block_count; ++block_idx) {
        const device uchar *block = row + ulong(block_idx) * 210ul;
        const device uchar *ql = block;
        const device uchar *qh = block + 128;
        const device uchar *scales = block + 192;
        const float d = f16le_to_f32_q6(block[208], block[209]);
        const uint vec_base = block_idx * 256u;

        for (uint group = 0; group < 2u; ++group) {
            const uint ql_base = group * 64u;
            const uint qh_base = group * 32u;
            const uint scale_base = group * 8u;
            const uint group_vec_base = vec_base + group * 128u;

            for (uint idx = 0; idx < 32u; ++idx) {
                const uint scale_offset = idx / 16u;
                const float scale0 = float(scales[scale_base + scale_offset + 0u]);
                const float scale1 = float(scales[scale_base + scale_offset + 2u]);
                const float scale2 = float(scales[scale_base + scale_offset + 4u]);
                const float scale3 = float(scales[scale_base + scale_offset + 6u]);
                const uchar high = qh[qh_base + idx];

                const int q0 = unpack_q6(ql[ql_base + idx] & 0x0Fu, (high >> 0u) & 0x03u);
                const int q1 = unpack_q6(ql[ql_base + idx + 32u] & 0x0Fu, (high >> 2u) & 0x03u);
                const int q2 = unpack_q6(ql[ql_base + idx] >> 4u, (high >> 4u) & 0x03u);
                const int q3 = unpack_q6(ql[ql_base + idx + 32u] >> 4u, (high >> 6u) & 0x03u);

                sum = fma(d * scale0 * float(q0), vector[group_vec_base + idx], sum);
                sum = fma(d * scale1 * float(q1), vector[group_vec_base + 32u + idx], sum);
                sum = fma(d * scale2 * float(q2), vector[group_vec_base + 64u + idx], sum);
                sum = fma(d * scale3 * float(q3), vector[group_vec_base + 96u + idx], sum);
            }
        }
    }

    output[gid] = sum;
}
