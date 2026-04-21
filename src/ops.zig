const std = @import("std");

pub fn rmsNorm(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    epsilon: f32,
) !void {
    if (out.len != input.len or input.len != weight.len) {
        return error.LengthMismatch;
    }

    var sum_sq: f64 = 0;
    for (input) |value| {
        sum_sq += @as(f64, value) * @as(f64, value);
    }

    const mean_sq = sum_sq / @as(f64, @floatFromInt(input.len));
    const inv_rms = @as(f32, @floatCast(1.0 / std.math.sqrt(mean_sq + epsilon)));

    for (out, input, weight) |*dst, x, w| {
        dst.* = x * inv_rms * w;
    }
}

test "rms norm applies scale" {
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var out: [4]f32 = undefined;

    try rmsNorm(&out, &input, &weight, 1e-6);

    var sum_sq: f64 = 0;
    for (input) |v| sum_sq += v * v;
    const inv_rms = @as(f32, @floatCast(1.0 / std.math.sqrt(sum_sq / 4.0 + 1e-6)));

    for (out, input) |got, x| {
        try std.testing.expectApproxEqAbs(x * inv_rms, got, 0.0001);
    }
}
