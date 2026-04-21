#include <metal_stdlib>
using namespace metal;

kernel void add_one(device float *values [[buffer(0)]],
                    uint gid [[thread_position_in_grid]]) {
    values[gid] = values[gid] + 1.0f;
}
