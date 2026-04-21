#include <metal_stdlib>
using namespace metal;

kernel void dense_matvec_f32(device const float *matrix [[buffer(0)]],
                             device const float *vector [[buffer(1)]],
                             device float *output [[buffer(2)]],
                             constant uint &cols [[buffer(3)]],
                             constant uint &rows [[buffer(4)]],
                             uint local_index [[thread_index_in_threadgroup]],
                             uint3 group_id [[threadgroup_position_in_grid]],
                             uint simd_group_index [[simdgroup_index_in_threadgroup]],
                             uint simd_lane [[thread_index_in_simdgroup]]) {
    threadgroup float vector_tile[128];
    const uint row = group_id.x * 4u + simd_group_index;
    const bool row_active = row < rows;

    float acc = 0.0f;
    const uint row_base = row * cols;
    for (uint tile_base = 0; tile_base < cols; tile_base += 128u) {
        const uint global_col = tile_base + local_index;
        vector_tile[local_index] = global_col < cols ? vector[global_col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_active) {
            for (uint chunk = 0; chunk < 128u; chunk += 32u) {
                const uint tile_col = chunk + simd_lane;
                const uint col = tile_base + tile_col;
                if (col < cols) {
                    acc = fma(matrix[row_base + col], vector_tile[tile_col], acc);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    acc = simd_sum(acc);
    if (simd_lane == 0u && row_active) {
        output[row] = acc;
    }
}
