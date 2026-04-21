#include <metal_stdlib>
using namespace metal;

static inline float f16le_to_f32(uchar lo, uchar hi) {
    const ushort bits = ushort(lo) | (ushort(hi) << 8);
    return float(as_type<half>(bits));
}

static inline uchar scale_k4(const device uchar *scales, uint index) {
    if (index < 4u) {
        return scales[index] & 63u;
    }
    return (scales[index + 4u] & 0x0Fu) | ((scales[index - 4u] >> 6u) << 4u);
}

static inline uchar min_k4(const device uchar *scales, uint index) {
    if (index < 4u) {
        return scales[index + 4u] & 63u;
    }
    return (scales[index + 4u] >> 4u) | ((scales[index] >> 6u) << 4u);
}

kernel void dmmv_q4_k(device const uchar *weights [[buffer(0)]],
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
        const device uchar *block = row + ulong(block_idx) * 144ul;
        const float d = f16le_to_f32(block[0], block[1]);
        const float dmin = f16le_to_f32(block[2], block[3]);
        const device uchar *scales = block + 4;
        const device uchar *quants = block + 16;
        const uint vec_base = block_idx * 256u;

        for (uint chunk = 0; chunk < 4u; ++chunk) {
            const uint idx0 = chunk * 2u;
            const uint idx1 = idx0 + 1u;
            const float d0 = d * float(scale_k4(scales, idx0));
            const float m0 = dmin * float(min_k4(scales, idx0));
            const float d1 = d * float(scale_k4(scales, idx1));
            const float m1 = dmin * float(min_k4(scales, idx1));
            const uint q_base = chunk * 32u;
            const uint x_base = vec_base + chunk * 64u;

            for (uint q = 0; q < 32u; ++q) {
                const uchar quant_byte = quants[q_base + q];
                const float x0 = vector[x_base + q];
                const float x1 = vector[x_base + 32u + q];
                sum = fma(d0 * float(quant_byte & 0x0Fu) - m0, x0, sum);
                sum = fma(d1 * float(quant_byte >> 4u) - m1, x1, sum);
            }
        }
    }

    output[gid] = sum;
}
