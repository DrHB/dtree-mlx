const std = @import("std");

const buffer_mod = @import("buffer.zig");
const command = @import("command.zig");
const common = @import("common.zig");
const device = @import("device.zig");
const pipeline_mod = @import("pipeline.zig");

pub const Error = common.Error;

const DemoValues = [_]f32{ 0.0, 1.0, 2.0, 3.0 };

pub const Report = struct {
    device_name: []u8,
    thread_execution_width: usize,
    max_total_threads_per_threadgroup: usize,
    input: [DemoValues.len]f32,
    output: [DemoValues.len]f32,

    pub fn deinit(self: *Report, allocator: std.mem.Allocator) void {
        allocator.free(self.device_name);
    }
};

pub const BenchmarkResult = struct {
    device_name: []u8,
    thread_execution_width: usize,
    max_total_threads_per_threadgroup: usize,
    elements: usize,
    bench_iters: usize,
    bench_warmup: usize,
    elapsed_s: f64,
    dispatches_per_s: f64,
    elements_per_s: f64,
    checksum: f64,

    pub fn deinit(self: *BenchmarkResult, allocator: std.mem.Allocator) void {
        allocator.free(self.device_name);
    }
};

pub const MatVecBenchmarkResult = struct {
    device_name: []u8,
    thread_execution_width: usize,
    max_total_threads_per_threadgroup: usize,
    rows: usize,
    cols: usize,
    bench_iters: usize,
    bench_warmup: usize,
    elapsed_s: f64,
    projection_passes_per_s: f64,
    output_rows_per_s: f64,
    gflops: f64,
    checksum: f64,

    pub fn deinit(self: *MatVecBenchmarkResult, allocator: std.mem.Allocator) void {
        allocator.free(self.device_name);
    }
};

pub const DenseBuffer = buffer_mod.DenseBuffer;
pub const RawBuffer = buffer_mod.RawBuffer;

pub const DenseContext = struct {
    session: device.Session,
    dense_pipeline: pipeline_mod.Pipeline,
    q4_k_pipeline: pipeline_mod.Pipeline,
    q6_k_pipeline: pipeline_mod.Pipeline,

    pub fn init() Error!DenseContext {
        var session = try device.Session.init();
        errdefer session.deinit();

        var dense_pipeline = try pipeline_mod.create(&session, "dense_matvec_f32");
        errdefer dense_pipeline.deinit();
        var q4_k_pipeline = try pipeline_mod.create(&session, "dmmv_q4_k");
        errdefer q4_k_pipeline.deinit();
        var q6_k_pipeline = try pipeline_mod.create(&session, "dmmv_q6_k");
        errdefer q6_k_pipeline.deinit();

        return .{
            .session = session,
            .dense_pipeline = dense_pipeline,
            .q4_k_pipeline = q4_k_pipeline,
            .q6_k_pipeline = q6_k_pipeline,
        };
    }

    pub fn deinit(self: *DenseContext) void {
        self.q6_k_pipeline.deinit();
        self.q4_k_pipeline.deinit();
        self.dense_pipeline.deinit();
        self.session.deinit();
        self.* = undefined;
    }

    pub fn allocBuffer(self: *DenseContext, element_count: usize) Error!DenseBuffer {
        return buffer_mod.allocateSharedBuffer(&self.session, element_count);
    }

    pub fn releaseBuffer(self: *DenseContext, buffer: *DenseBuffer) void {
        _ = self;
        buffer_mod.releaseBuffer(buffer);
    }

    pub fn wrapBytesNoCopy(self: *DenseContext, bytes: []const u8) Error!RawBuffer {
        return buffer_mod.wrapBytesNoCopy(&self.session, bytes);
    }

    pub fn releaseRawBuffer(self: *DenseContext, buffer: *RawBuffer) void {
        _ = self;
        buffer_mod.releaseRawBuffer(buffer);
    }

    pub fn matvec(
        self: *DenseContext,
        matrix: *const DenseBuffer,
        vector: *const DenseBuffer,
        output: *const DenseBuffer,
        rows: usize,
        cols: usize,
    ) Error!void {
        const matrix_elements = std.math.mul(usize, rows, cols) catch return error.InvalidInputBuffer;
        if (matrix.len < matrix_elements) return error.InvalidInputBuffer;
        if (vector.len < cols) return error.InvalidInputBuffer;
        if (output.len < rows) return error.OutputBufferTooSmall;

        try command.dispatchDenseMatVec(
            &self.session,
            &self.dense_pipeline,
            matrix,
            vector,
            output,
            rows,
            cols,
        );
    }

    pub fn matvecQ4K(
        self: *DenseContext,
        weights: *const RawBuffer,
        vector: *const DenseBuffer,
        output: *const DenseBuffer,
        rows: usize,
        cols: usize,
        row_bytes: usize,
    ) Error!void {
        try command.dispatchQ4KMatVec(
            &self.session,
            &self.q4_k_pipeline,
            weights,
            vector,
            output,
            rows,
            cols,
            row_bytes,
        );
    }

    pub fn matvecQ6K(
        self: *DenseContext,
        weights: *const RawBuffer,
        vector: *const DenseBuffer,
        output: *const DenseBuffer,
        rows: usize,
        cols: usize,
        row_bytes: usize,
    ) Error!void {
        try command.dispatchQ6KMatVec(
            &self.session,
            &self.q6_k_pipeline,
            weights,
            vector,
            output,
            rows,
            cols,
            row_bytes,
        );
    }
};

pub fn runBootstrap(allocator: std.mem.Allocator) Error!Report {
    var session = try device.Session.init();
    defer session.deinit();

    var pipeline = try pipeline_mod.create(&session, "add_one");
    defer pipeline.deinit();

    var buffer = try buffer_mod.allocateSharedBuffer(&session, DemoValues.len);
    defer buffer_mod.releaseBuffer(&buffer);

    @memcpy(buffer.floats[0..DemoValues.len], DemoValues[0..]);
    try command.dispatchAddOne(&session, &pipeline, &buffer, DemoValues.len);

    var output: [DemoValues.len]f32 = undefined;
    @memcpy(output[0..], buffer.floats[0..DemoValues.len]);
    for (output, 0..) |value, idx| {
        const expected = DemoValues[idx] + 1.0;
        if (@abs(value - expected) > 0.0001) return error.ValidationFailed;
    }

    return .{
        .device_name = try session.copyDeviceName(allocator),
        .thread_execution_width = pipeline.thread_execution_width,
        .max_total_threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup,
        .input = DemoValues,
        .output = output,
    };
}

pub fn runBenchmark(
    allocator: std.mem.Allocator,
    element_count: usize,
    bench_warmup: usize,
    bench_iters: usize,
) Error!BenchmarkResult {
    if (element_count == 0) return error.InvalidElementCount;
    if (bench_iters == 0) return error.InvalidBenchIters;

    var session = try device.Session.init();
    defer session.deinit();

    var pipeline = try pipeline_mod.create(&session, "add_one");
    defer pipeline.deinit();

    var buffer = try buffer_mod.allocateSharedBuffer(&session, element_count);
    defer buffer_mod.releaseBuffer(&buffer);

    fillPattern(buffer.floats, element_count);

    for (0..bench_warmup) |_| {
        try command.dispatchAddOne(&session, &pipeline, &buffer, element_count);
    }

    const start_ns = try monotonicNowNs();
    for (0..bench_iters) |_| {
        try command.dispatchAddOne(&session, &pipeline, &buffer, element_count);
    }
    const end_ns = try monotonicNowNs();
    const elapsed_s = @as(f64, @floatFromInt(end_ns - start_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s));

    const expected_add = @as(f32, @floatFromInt(bench_warmup + bench_iters));
    var checksum: f64 = 0;
    for (0..element_count) |idx| {
        const expected = initialValue(idx) + expected_add;
        const actual = buffer.floats[idx];
        if (@abs(actual - expected) > 0.0001) return error.ValidationFailed;
        checksum += actual;
    }

    return .{
        .device_name = try session.copyDeviceName(allocator),
        .thread_execution_width = pipeline.thread_execution_width,
        .max_total_threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup,
        .elements = element_count,
        .bench_iters = bench_iters,
        .bench_warmup = bench_warmup,
        .elapsed_s = elapsed_s,
        .dispatches_per_s = @as(f64, @floatFromInt(bench_iters)) / elapsed_s,
        .elements_per_s = (@as(f64, @floatFromInt(element_count)) * @as(f64, @floatFromInt(bench_iters))) / elapsed_s,
        .checksum = checksum,
    };
}

pub fn runMatVecBenchmark(
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    bench_warmup: usize,
    bench_iters: usize,
) Error!MatVecBenchmarkResult {
    if (rows == 0) return error.InvalidRowCount;
    if (cols == 0) return error.InvalidColCount;
    if (bench_iters == 0) return error.InvalidBenchIters;

    const matrix_count = std.math.mul(usize, rows, cols) catch return error.InvalidInputBuffer;

    var session = try device.Session.init();
    defer session.deinit();

    var pipeline = try pipeline_mod.create(&session, "dense_matvec_f32");
    defer pipeline.deinit();

    var matrix = try buffer_mod.allocateSharedBuffer(&session, matrix_count);
    defer buffer_mod.releaseBuffer(&matrix);
    var vector = try buffer_mod.allocateSharedBuffer(&session, cols);
    defer buffer_mod.releaseBuffer(&vector);
    var output = try buffer_mod.allocateSharedBuffer(&session, rows);
    defer buffer_mod.releaseBuffer(&output);

    fillMatVecMatrix(matrix.floats, matrix_count);
    fillMatVecVector(vector.floats, cols);
    @memset(output.floats[0..rows], 0);

    for (0..bench_warmup) |_| {
        try command.dispatchDenseMatVec(&session, &pipeline, &matrix, &vector, &output, rows, cols);
    }

    const start_ns = try monotonicNowNs();
    for (0..bench_iters) |_| {
        try command.dispatchDenseMatVec(&session, &pipeline, &matrix, &vector, &output, rows, cols);
    }
    const end_ns = try monotonicNowNs();
    const elapsed_s = @as(f64, @floatFromInt(end_ns - start_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s));
    const checksum = try validateMatVec(matrix.floats, vector.floats, output.floats, rows, cols);
    const operations_per_pass = @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(cols)) * 2.0;

    return .{
        .device_name = try session.copyDeviceName(allocator),
        .thread_execution_width = pipeline.thread_execution_width,
        .max_total_threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup,
        .rows = rows,
        .cols = cols,
        .bench_iters = bench_iters,
        .bench_warmup = bench_warmup,
        .elapsed_s = elapsed_s,
        .projection_passes_per_s = @as(f64, @floatFromInt(bench_iters)) / elapsed_s,
        .output_rows_per_s = (@as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(bench_iters))) / elapsed_s,
        .gflops = (operations_per_pass * @as(f64, @floatFromInt(bench_iters))) / elapsed_s / 1_000_000_000.0,
        .checksum = checksum,
    };
}

fn fillPattern(values: [*]f32, element_count: usize) void {
    for (0..element_count) |idx| {
        values[idx] = initialValue(idx);
    }
}

fn fillMatVecMatrix(values: [*]f32, element_count: usize) void {
    for (0..element_count) |idx| {
        values[idx] = matVecMatrixValue(idx);
    }
}

fn fillMatVecVector(values: [*]f32, element_count: usize) void {
    for (0..element_count) |idx| {
        values[idx] = matVecVectorValue(idx);
    }
}

fn initialValue(idx: usize) f32 {
    return @as(f32, @floatFromInt(idx % 251));
}

fn matVecMatrixValue(idx: usize) f32 {
    const raw: i32 = @as(i32, @intCast(idx % 97)) - 48;
    return @as(f32, @floatFromInt(raw)) * 0.03125;
}

fn matVecVectorValue(idx: usize) f32 {
    const raw: i32 = @as(i32, @intCast(idx % 29)) - 14;
    return @as(f32, @floatFromInt(raw)) * 0.0625;
}

fn validateMatVec(matrix: [*]const f32, vector: [*]const f32, output: [*]const f32, rows: usize, cols: usize) Error!f64 {
    var checksum: f64 = 0;
    for (0..rows) |row| {
        const base = row * cols;
        var expected: f64 = 0;
        for (0..cols) |col| {
            expected += @as(f64, matrix[base + col]) * @as(f64, vector[col]);
        }
        const expected_f32: f32 = @floatCast(expected);
        const actual = output[row];
        const tolerance = 0.05 + 0.001 * @abs(expected_f32);
        if (@abs(actual - expected_f32) > tolerance) return error.ValidationFailed;
        checksum += actual;
    }
    return checksum;
}

fn monotonicNowNs() Error!u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts) != 0) {
        return error.ClockGetTimeFailed;
    }

    const total_ns = @as(i128, ts.sec) * std.time.ns_per_s + @as(i128, ts.nsec);
    return std.math.cast(u64, total_ns) orelse error.ClockOutOfRange;
}
