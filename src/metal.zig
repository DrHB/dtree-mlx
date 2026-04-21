const std = @import("std");
const metal = @import("metal/runtime.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const report = try metal.runBootstrap(allocator);
    defer {
        var owned = report;
        owned.deinit(allocator);
    }

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &stdout_buffer);

    try stdout.interface.writeAll("metal bootstrap ok\n");
    try stdout.interface.print("device: {s}\n", .{report.device_name});
    try stdout.interface.print("thread_execution_width: {d}\n", .{report.thread_execution_width});
    try stdout.interface.print("max_threads_per_threadgroup: {d}\n", .{report.max_total_threads_per_threadgroup});
    try stdout.interface.writeAll("input: ");
    try printSlice(&stdout.interface, report.input[0..]);
    try stdout.interface.writeAll("output: ");
    try printSlice(&stdout.interface, report.output[0..]);
    try stdout.interface.flush();
}

fn printSlice(writer: *std.Io.Writer, values: []const f32) !void {
    try writer.writeByte('[');
    for (values, 0..) |value, idx| {
        if (idx != 0) try writer.writeAll(", ");
        try writer.print("{d:.3}", .{value});
    }
    try writer.writeAll("]\n");
}
