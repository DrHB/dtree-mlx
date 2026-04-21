const std = @import("std");
const metal = @import("metal/runtime.zig");

const BenchKind = enum {
    add_one,
    matvec,
};

const Args = struct {
    bench: bool = false,
    bench_kind: BenchKind = .add_one,
    bench_iters: usize = 20,
    bench_warmup: usize = 5,
    elements: usize = 1 << 20,
    rows: usize = 8192,
    cols: usize = 2048,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &stdout_buffer);

    var arg_it = try init.minimal.args.iterateAllocator(allocator);
    defer arg_it.deinit();
    const args = try parseArgs(&arg_it);

    if (args.bench) {
        switch (args.bench_kind) {
            .add_one => {
                const result = try metal.runBenchmark(allocator, args.elements, args.bench_warmup, args.bench_iters);
                defer {
                    var owned = result;
                    owned.deinit(allocator);
                }
                try stdout.interface.writeAll("Metal add-one benchmark\n");
                try stdout.interface.print("device: {s}\n", .{result.device_name});
                try stdout.interface.print("thread_execution_width: {d}\n", .{result.thread_execution_width});
                try stdout.interface.print("max_threads_per_threadgroup: {d}\n", .{result.max_total_threads_per_threadgroup});
                try stdout.interface.print("elements: {d}\n", .{result.elements});
                try stdout.interface.print("bench_warmup: {d}\n", .{result.bench_warmup});
                try stdout.interface.print("bench_iters: {d}\n", .{result.bench_iters});
                try stdout.interface.print("elapsed_s: {d}\n", .{result.elapsed_s});
                try stdout.interface.print("metal_dispatches_per_s: {d}\n", .{result.dispatches_per_s});
                try stdout.interface.print("metal_elements_per_s: {d}\n", .{result.elements_per_s});
                try stdout.interface.print("checksum: {d}\n", .{result.checksum});
                try stdout.interface.writeAll("note: bootstrap bandwidth check only, not inference-shaped work.\n");
            },
            .matvec => {
                const result = try metal.runMatVecBenchmark(allocator, args.rows, args.cols, args.bench_warmup, args.bench_iters);
                defer {
                    var owned = result;
                    owned.deinit(allocator);
                }
                try stdout.interface.writeAll("Metal dense matvec benchmark\n");
                try stdout.interface.print("device: {s}\n", .{result.device_name});
                try stdout.interface.print("thread_execution_width: {d}\n", .{result.thread_execution_width});
                try stdout.interface.print("max_threads_per_threadgroup: {d}\n", .{result.max_total_threads_per_threadgroup});
                try stdout.interface.print("rows: {d}\n", .{result.rows});
                try stdout.interface.print("cols: {d}\n", .{result.cols});
                try stdout.interface.print("bench_warmup: {d}\n", .{result.bench_warmup});
                try stdout.interface.print("bench_iters: {d}\n", .{result.bench_iters});
                try stdout.interface.print("elapsed_s: {d}\n", .{result.elapsed_s});
                try stdout.interface.print("metal_projection_passes_per_s: {d}\n", .{result.projection_passes_per_s});
                try stdout.interface.print("metal_matvec_output_rows_per_s: {d}\n", .{result.output_rows_per_s});
                try stdout.interface.print("metal_matvec_gflops: {d}\n", .{result.gflops});
                try stdout.interface.print("checksum: {d}\n", .{result.checksum});
                try stdout.interface.writeAll("note: this is the first inference-shaped native Zig Metal projection baseline.\n");
            },
        }
    } else {
        const report = try metal.runBootstrap(allocator);
        defer {
            var owned = report;
            owned.deinit(allocator);
        }

        try stdout.interface.writeAll("metal bootstrap ok\n");
        try stdout.interface.print("device: {s}\n", .{report.device_name});
        try stdout.interface.print("thread_execution_width: {d}\n", .{report.thread_execution_width});
        try stdout.interface.print("max_threads_per_threadgroup: {d}\n", .{report.max_total_threads_per_threadgroup});
        try stdout.interface.writeAll("input: ");
        try printSlice(&stdout.interface, report.input[0..]);
        try stdout.interface.writeAll("output: ");
        try printSlice(&stdout.interface, report.output[0..]);
    }

    try stdout.interface.flush();
}

fn parseArgs(arg_it: *std.process.Args.Iterator) !Args {
    var out = Args{};

    _ = arg_it.next();
    while (arg_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "--bench")) {
            out.bench = true;
        } else if (std.mem.eql(u8, arg, "--bench-kind")) {
            const value = arg_it.next() orelse return error.MissingValue;
            if (std.mem.eql(u8, value, "add-one")) {
                out.bench_kind = .add_one;
            } else if (std.mem.eql(u8, value, "matvec")) {
                out.bench_kind = .matvec;
            } else {
                std.debug.print("unknown bench kind: {s}\n", .{value});
                printUsage();
                return error.UnknownArgument;
            }
        } else if (std.mem.eql(u8, arg, "--bench-iters")) {
            out.bench_iters = try std.fmt.parseInt(usize, arg_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--bench-warmup")) {
            out.bench_warmup = try std.fmt.parseInt(usize, arg_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--elements")) {
            out.elements = try std.fmt.parseInt(usize, arg_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--rows")) {
            out.rows = try std.fmt.parseInt(usize, arg_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--cols")) {
            out.cols = try std.fmt.parseInt(usize, arg_it.next() orelse return error.MissingValue, 10);
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            printUsage();
            return error.UnknownArgument;
        }
    }

    return out;
}

fn printSlice(writer: *std.Io.Writer, values: []const f32) !void {
    try writer.writeByte('[');
    for (values, 0..) |value, idx| {
        if (idx != 0) try writer.writeAll(", ");
        try writer.print("{d:.3}", .{value});
    }
    try writer.writeAll("]\n");
}

fn printUsage() void {
    std.debug.print(
        \\Usage: dtree-mlx-metal-bootstrap [options]
        \\
        \\Options:
        \\  --bench              Run a native Metal benchmark.
        \\  --bench-kind KIND    `add-one` or `matvec`. Default: add-one
        \\  --bench-iters N      Timed benchmark dispatches. Default: 20
        \\  --bench-warmup N     Warmup dispatches. Default: 5
        \\  --elements N         Number of f32 elements processed per dispatch. Default: 1048576
        \\  --rows N             Matvec output rows. Default: 8192
        \\  --cols N             Matvec input columns. Default: 2048
        \\  --help               Print this help text.
        \\
    , .{});
}
