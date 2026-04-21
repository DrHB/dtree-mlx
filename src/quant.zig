const std = @import("std");

pub const qk_k: usize = 256;
pub const k_scale_size: usize = 12;

pub const q4_k_block_bytes: usize = 2 + 2 + k_scale_size + (qk_k / 2);
pub const q6_k_block_bytes: usize = 2 + (qk_k / 16) + (3 * qk_k / 4);

pub const TypeTraits = struct {
    name: []const u8,
    block_elements: usize,
    block_bytes: usize,
    quantized: bool,
};

pub fn typeTraits(ggml_type: i32) ?TypeTraits {
    return switch (ggml_type) {
        0 => .{
            .name = "f32",
            .block_elements = 1,
            .block_bytes = 4,
            .quantized = false,
        },
        12 => .{
            .name = "q4_K",
            .block_elements = qk_k,
            .block_bytes = q4_k_block_bytes,
            .quantized = true,
        },
        14 => .{
            .name = "q6_K",
            .block_elements = qk_k,
            .block_bytes = q6_k_block_bytes,
            .quantized = true,
        },
        else => null,
    };
}

pub fn rowBytes(ggml_type: i32, row_len: usize) !usize {
    const traits = typeTraits(ggml_type) orelse return error.UnsupportedTensorType;
    if (!traits.quantized) {
        return std.math.mul(usize, row_len, traits.block_bytes);
    }
    if (row_len % traits.block_elements != 0) {
        return error.InvalidRowLength;
    }
    return std.math.mul(usize, row_len / traits.block_elements, traits.block_bytes);
}

pub fn dequantizeRow(
    ggml_type: i32,
    row_data: []const u8,
    out: []f32,
) !void {
    switch (ggml_type) {
        0 => try dequantizeRowF32(row_data, out),
        12 => try dequantizeRowQ4K(row_data, out),
        14 => try dequantizeRowQ6K(row_data, out),
        else => return error.UnsupportedTensorType,
    }
}

pub fn dotRow(
    ggml_type: i32,
    row_data: []const u8,
    input: []const f32,
) !f32 {
    return switch (ggml_type) {
        0 => try dotRowF32(row_data, input),
        12 => try dotRowQ4K(row_data, input),
        14 => try dotRowQ6K(row_data, input),
        else => error.UnsupportedTensorType,
    };
}

fn dequantizeRowF32(row_data: []const u8, out: []f32) !void {
    if (row_data.len != out.len * @sizeOf(f32)) {
        return error.InvalidRowData;
    }

    for (out, 0..) |*value, idx| {
        const start = idx * @sizeOf(f32);
        const bits = std.mem.readInt(u32, row_data[start..][0..4], .little);
        value.* = @bitCast(bits);
    }
}

fn dotRowF32(row_data: []const u8, input: []const f32) !f32 {
    if (row_data.len != input.len * @sizeOf(f32)) {
        return error.InvalidRowData;
    }

    var sum: f32 = 0;
    for (input, 0..) |value, idx| {
        const start = idx * @sizeOf(f32);
        const bits = std.mem.readInt(u32, row_data[start..][0..4], .little);
        sum += @as(f32, @bitCast(bits)) * value;
    }
    return sum;
}

fn dequantizeRowQ4K(row_data: []const u8, out: []f32) !void {
    if (out.len % qk_k != 0) {
        return error.InvalidRowLength;
    }

    const blocks = out.len / qk_k;
    if (row_data.len != blocks * q4_k_block_bytes) {
        return error.InvalidRowData;
    }

    for (0..blocks) |block_idx| {
        const row_start = block_idx * qk_k;
        const data_start = block_idx * q4_k_block_bytes;
        try dequantizeBlockQ4K(
            row_data[data_start..][0..q4_k_block_bytes],
            out[row_start..][0..qk_k],
        );
    }
}

fn dotRowQ4K(row_data: []const u8, input: []const f32) !f32 {
    if (input.len % qk_k != 0) {
        return error.InvalidRowLength;
    }

    const blocks = input.len / qk_k;
    if (row_data.len != blocks * q4_k_block_bytes) {
        return error.InvalidRowData;
    }

    var sum: f32 = 0;
    for (0..blocks) |block_idx| {
        const input_start = block_idx * qk_k;
        const data_start = block_idx * q4_k_block_bytes;
        sum += try dotBlockQ4K(
            row_data[data_start..][0..q4_k_block_bytes],
            input[input_start..][0..qk_k],
        );
    }
    return sum;
}

fn dequantizeRowQ6K(row_data: []const u8, out: []f32) !void {
    if (out.len % qk_k != 0) {
        return error.InvalidRowLength;
    }

    const blocks = out.len / qk_k;
    if (row_data.len != blocks * q6_k_block_bytes) {
        return error.InvalidRowData;
    }

    for (0..blocks) |block_idx| {
        const row_start = block_idx * qk_k;
        const data_start = block_idx * q6_k_block_bytes;
        try dequantizeBlockQ6K(
            row_data[data_start..][0..q6_k_block_bytes],
            out[row_start..][0..qk_k],
        );
    }
}

fn dotRowQ6K(row_data: []const u8, input: []const f32) !f32 {
    if (input.len % qk_k != 0) {
        return error.InvalidRowLength;
    }

    const blocks = input.len / qk_k;
    if (row_data.len != blocks * q6_k_block_bytes) {
        return error.InvalidRowData;
    }

    var sum: f32 = 0;
    for (0..blocks) |block_idx| {
        const input_start = block_idx * qk_k;
        const data_start = block_idx * q6_k_block_bytes;
        sum += try dotBlockQ6K(
            row_data[data_start..][0..q6_k_block_bytes],
            input[input_start..][0..qk_k],
        );
    }
    return sum;
}

fn dequantizeBlockQ4K(block: []const u8, out: []f32) !void {
    if (block.len != q4_k_block_bytes or out.len != qk_k) {
        return error.InvalidBlockData;
    }

    const d = readF16Le(block[0..2]);
    const dmin = readF16Le(block[2..4]);
    const scales = block[4..16];
    const quants = block[16..];

    var out_index: usize = 0;
    var scale_index: usize = 0;
    var quant_index: usize = 0;

    while (quant_index < quants.len) : ({
        quant_index += 32;
        scale_index += 2;
        out_index += 64;
    }) {
        const scale_min_1 = getScaleMinK4(scale_index + 0, scales);
        const scale_min_2 = getScaleMinK4(scale_index + 1, scales);
        const d1 = d * toF32(scale_min_1.scale);
        const m1 = dmin * toF32(scale_min_1.min);
        const d2 = d * toF32(scale_min_2.scale);
        const m2 = dmin * toF32(scale_min_2.min);

        for (0..32) |idx| {
            const quant_byte = quants[quant_index + idx];
            out[out_index + idx] = d1 * toF32(quant_byte & 0x0F) - m1;
            out[out_index + 32 + idx] = d2 * toF32(quant_byte >> 4) - m2;
        }
    }
}

fn dotBlockQ4K(block: []const u8, input: []const f32) !f32 {
    if (block.len != q4_k_block_bytes or input.len != qk_k) {
        return error.InvalidBlockData;
    }

    const d = readF16Le(block[0..2]);
    const dmin = readF16Le(block[2..4]);
    const scales = block[4..16];
    const quants = block[16..];

    var input_index: usize = 0;
    var scale_index: usize = 0;
    var quant_index: usize = 0;
    var sum: f32 = 0;

    while (quant_index < quants.len) : ({
        quant_index += 32;
        scale_index += 2;
        input_index += 64;
    }) {
        const scale_min_1 = getScaleMinK4(scale_index + 0, scales);
        const scale_min_2 = getScaleMinK4(scale_index + 1, scales);
        const d1 = d * toF32(scale_min_1.scale);
        const m1 = dmin * toF32(scale_min_1.min);
        const d2 = d * toF32(scale_min_2.scale);
        const m2 = dmin * toF32(scale_min_2.min);

        for (0..32) |idx| {
            const quant_byte = quants[quant_index + idx];
            const x1 = input[input_index + idx];
            const x2 = input[input_index + 32 + idx];
            sum += (d1 * toF32(quant_byte & 0x0F) - m1) * x1;
            sum += (d2 * toF32(quant_byte >> 4) - m2) * x2;
        }
    }

    return sum;
}

fn dequantizeBlockQ6K(block: []const u8, out: []f32) !void {
    if (block.len != q6_k_block_bytes or out.len != qk_k) {
        return error.InvalidBlockData;
    }

    const ql = block[0..128];
    const qh = block[128..192];
    const scales = block[192..208];
    const d = readF16Le(block[208..210]);

    var out_index: usize = 0;
    var ql_index: usize = 0;
    var qh_index: usize = 0;
    var scale_index: usize = 0;

    while (ql_index < ql.len) : ({
        out_index += 128;
        ql_index += 64;
        qh_index += 32;
        scale_index += 8;
    }) {
        for (0..32) |idx| {
            const scale_offset = idx / 16;
            const scale_1 = toF32Signed(scales[scale_index + scale_offset + 0]);
            const scale_2 = toF32Signed(scales[scale_index + scale_offset + 2]);
            const scale_3 = toF32Signed(scales[scale_index + scale_offset + 4]);
            const scale_4 = toF32Signed(scales[scale_index + scale_offset + 6]);
            const high = qh[qh_index + idx];

            const q1 = unpackQ6(ql[ql_index + idx + 0] & 0x0F, (high >> 0) & 0x03);
            const q2 = unpackQ6(ql[ql_index + idx + 32] & 0x0F, (high >> 2) & 0x03);
            const q3 = unpackQ6(ql[ql_index + idx + 0] >> 4, (high >> 4) & 0x03);
            const q4 = unpackQ6(ql[ql_index + idx + 32] >> 4, (high >> 6) & 0x03);

            out[out_index + idx + 0] = d * scale_1 * toF32Signed(q1);
            out[out_index + idx + 32] = d * scale_2 * toF32Signed(q2);
            out[out_index + idx + 64] = d * scale_3 * toF32Signed(q3);
            out[out_index + idx + 96] = d * scale_4 * toF32Signed(q4);
        }
    }
}

fn dotBlockQ6K(block: []const u8, input: []const f32) !f32 {
    if (block.len != q6_k_block_bytes or input.len != qk_k) {
        return error.InvalidBlockData;
    }

    const ql = block[0..128];
    const qh = block[128..192];
    const scales = block[192..208];
    const d = readF16Le(block[208..210]);

    var input_index: usize = 0;
    var ql_index: usize = 0;
    var qh_index: usize = 0;
    var scale_index: usize = 0;
    var sum: f32 = 0;

    while (ql_index < ql.len) : ({
        input_index += 128;
        ql_index += 64;
        qh_index += 32;
        scale_index += 8;
    }) {
        for (0..32) |idx| {
            const scale_offset = idx / 16;
            const scale_1 = toF32Signed(scales[scale_index + scale_offset + 0]);
            const scale_2 = toF32Signed(scales[scale_index + scale_offset + 2]);
            const scale_3 = toF32Signed(scales[scale_index + scale_offset + 4]);
            const scale_4 = toF32Signed(scales[scale_index + scale_offset + 6]);
            const high = qh[qh_index + idx];

            const q1 = unpackQ6(ql[ql_index + idx + 0] & 0x0F, (high >> 0) & 0x03);
            const q2 = unpackQ6(ql[ql_index + idx + 32] & 0x0F, (high >> 2) & 0x03);
            const q3 = unpackQ6(ql[ql_index + idx + 0] >> 4, (high >> 4) & 0x03);
            const q4 = unpackQ6(ql[ql_index + idx + 32] >> 4, (high >> 6) & 0x03);

            sum += d * scale_1 * toF32Signed(q1) * input[input_index + idx + 0];
            sum += d * scale_2 * toF32Signed(q2) * input[input_index + idx + 32];
            sum += d * scale_3 * toF32Signed(q3) * input[input_index + idx + 64];
            sum += d * scale_4 * toF32Signed(q4) * input[input_index + idx + 96];
        }
    }

    return sum;
}

const ScaleMin = struct {
    scale: u8,
    min: u8,
};

fn getScaleMinK4(index: usize, scales: []const u8) ScaleMin {
    std.debug.assert(scales.len == k_scale_size);

    if (index < 4) {
        return .{
            .scale = scales[index] & 63,
            .min = scales[index + 4] & 63,
        };
    }

    return .{
        .scale = (scales[index + 4] & 0x0F) | ((scales[index - 4] >> 6) << 4),
        .min = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
    };
}

fn unpackQ6(lower: u8, upper: u8) i8 {
    const value = (@as(i32, lower) | (@as(i32, upper) << 4)) - 32;
    return @intCast(value);
}

fn readF16Le(bytes: []const u8) f32 {
    const bits = std.mem.readInt(u16, bytes[0..2], .little);
    const value = @as(f16, @bitCast(bits));
    return @as(f32, @floatCast(value));
}

fn toF32(value: anytype) f32 {
    return @as(f32, @floatFromInt(value));
}

fn toF32Signed(value: anytype) f32 {
    return @as(f32, @floatFromInt(value));
}

test "row byte counts match ggml layouts" {
    try std.testing.expectEqual(@as(usize, 1024), try rowBytes(0, 256));
    try std.testing.expectEqual(@as(usize, q4_k_block_bytes), try rowBytes(12, 256));
    try std.testing.expectEqual(@as(usize, q6_k_block_bytes), try rowBytes(14, 256));
    try std.testing.expectEqual(@as(usize, 1152), try rowBytes(12, 2048));
    try std.testing.expectEqual(@as(usize, 1680), try rowBytes(14, 2048));
}

test "dequantize q4_K block" {
    var block = [_]u8{0} ** q4_k_block_bytes;
    block[0] = 0x00;
    block[1] = 0x38; // 0.5
    block[2] = 0x00;
    block[3] = 0x34; // 0.25

    for (0..4) |idx| {
        block[4 + idx] = 2;
        block[8 + idx] = 1;
        block[12 + idx] = 0x12;
    }
    for (16..block.len) |idx| {
        block[idx] = 0x73;
    }

    var out: [qk_k]f32 = undefined;
    try dequantizeRow(12, &block, &out);

    try std.testing.expectApproxEqAbs(@as(f32, 2.75), out[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.75), out[31], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.75), out[32], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.75), out[63], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.75), out[64], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.75), out[255], 0.0001);
}

test "dequantize q6_K block" {
    var block = [_]u8{0} ** q6_k_block_bytes;
    for (192..208) |idx| {
        block[idx] = 1;
    }
    block[208] = 0x00;
    block[209] = 0x38; // 0.5

    var out: [qk_k]f32 = undefined;
    try dequantizeRow(14, &block, &out);

    for (out) |value| {
        try std.testing.expectApproxEqAbs(@as(f32, -16.0), value, 0.0001);
    }
}

test "dot q4_K row matches dequantized reference" {
    var block = [_]u8{0} ** q4_k_block_bytes;
    block[0] = 0x00;
    block[1] = 0x38; // 0.5
    block[2] = 0x00;
    block[3] = 0x34; // 0.25

    for (0..4) |idx| {
        block[4 + idx] = 2;
        block[8 + idx] = 1;
        block[12 + idx] = 0x12;
    }
    for (16..block.len) |idx| {
        block[idx] = 0x73;
    }

    var input: [qk_k]f32 = undefined;
    fillTestInput(&input);

    var ref: [qk_k]f32 = undefined;
    try dequantizeRow(12, &block, &ref);

    const got = try dotRow(12, &block, &input);
    const expected = dotReference(&ref, &input);
    try std.testing.expectApproxEqAbs(expected, got, 0.01);
}

test "dot q6_K row matches dequantized reference" {
    var block = [_]u8{0} ** q6_k_block_bytes;
    for (0..128) |idx| {
        block[idx] = @intCast(idx % 16);
    }
    for (128..192) |idx| {
        block[idx] = @intCast(idx % 4);
    }
    for (192..208) |idx| {
        const scale = @as(i32, @intCast(idx - 192)) - 8;
        block[idx] = @bitCast(@as(i8, @intCast(scale)));
    }
    block[208] = 0x00;
    block[209] = 0x38; // 0.5

    var input: [qk_k]f32 = undefined;
    fillTestInput(&input);

    var ref: [qk_k]f32 = undefined;
    try dequantizeRow(14, &block, &ref);

    const got = try dotRow(14, &block, &input);
    const expected = dotReference(&ref, &input);
    try std.testing.expectApproxEqAbs(expected, got, 0.05);
}

fn fillTestInput(out: []f32) void {
    for (out, 0..) |*value, idx| {
        const lane = @as(i32, @intCast(idx % 19)) - 9;
        value.* = @as(f32, @floatFromInt(lane)) / 4.0;
    }
}

fn dotReference(lhs: []const f32, rhs: []const f32) f32 {
    std.debug.assert(lhs.len == rhs.len);
    var sum: f32 = 0;
    for (lhs, rhs) |a, b| {
        sum += a * b;
    }
    return sum;
}
