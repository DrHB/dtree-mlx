const std = @import("std");

pub const Error = error{
    OutOfMemory,
    UnsupportedPlatform,
    NoMetalDevice,
    StringCreationFailed,
    ShaderSourceDirectoryNotFound,
    ShaderSourceLoadFailed,
    ShaderCompilationFailed,
    FunctionLookupFailed,
    PipelineCreationFailed,
    CommandQueueCreationFailed,
    CommandBufferCreationFailed,
    ComputeEncoderCreationFailed,
    CommandExecutionFailed,
    BufferCreationFailed,
    BufferMapFailed,
    ValidationFailed,
    InvalidBenchIters,
    InvalidElementCount,
    InvalidRowCount,
    InvalidColCount,
    InvalidInputBuffer,
    OutputBufferTooSmall,
    ClockGetTimeFailed,
    ClockOutOfRange,
};

pub const Report = struct {
    device_name: []u8,
    thread_execution_width: usize,
    max_total_threads_per_threadgroup: usize,
    input: [4]f32,
    output: [4]f32,

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

pub const DenseBuffer = struct {
    object: ?*anyopaque,
    floats: [*]f32,
    len: usize,
};

pub const RawBuffer = struct {
    object: ?*anyopaque,
    len_bytes: usize,
};

pub const DenseContext = struct {
    pub fn init() Error!DenseContext {
        return error.UnsupportedPlatform;
    }

    pub fn deinit(self: *DenseContext) void {
        self.* = undefined;
    }

    pub fn allocBuffer(self: *DenseContext, element_count: usize) Error!DenseBuffer {
        _ = self;
        _ = element_count;
        return error.UnsupportedPlatform;
    }

    pub fn releaseBuffer(self: *DenseContext, buffer: *DenseBuffer) void {
        _ = self;
        buffer.* = undefined;
    }

    pub fn wrapBytesNoCopy(self: *DenseContext, bytes: []const u8) Error!RawBuffer {
        _ = self;
        _ = bytes;
        return error.UnsupportedPlatform;
    }

    pub fn releaseRawBuffer(self: *DenseContext, buffer: *RawBuffer) void {
        _ = self;
        buffer.* = undefined;
    }

    pub fn matvec(
        self: *DenseContext,
        matrix: *const DenseBuffer,
        vector: *const DenseBuffer,
        output: *const DenseBuffer,
        rows: usize,
        cols: usize,
    ) Error!void {
        _ = self;
        _ = matrix;
        _ = vector;
        _ = output;
        _ = rows;
        _ = cols;
        return error.UnsupportedPlatform;
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
        _ = self;
        _ = weights;
        _ = vector;
        _ = output;
        _ = rows;
        _ = cols;
        _ = row_bytes;
        return error.UnsupportedPlatform;
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
        _ = self;
        _ = weights;
        _ = vector;
        _ = output;
        _ = rows;
        _ = cols;
        _ = row_bytes;
        return error.UnsupportedPlatform;
    }
};

pub fn runBootstrap(allocator: std.mem.Allocator) Error!Report {
    _ = allocator;
    return error.UnsupportedPlatform;
}

pub fn runBenchmark(
    allocator: std.mem.Allocator,
    element_count: usize,
    bench_warmup: usize,
    bench_iters: usize,
) Error!BenchmarkResult {
    _ = allocator;
    _ = element_count;
    _ = bench_warmup;
    _ = bench_iters;
    return error.UnsupportedPlatform;
}

pub fn runMatVecBenchmark(
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    bench_warmup: usize,
    bench_iters: usize,
) Error!MatVecBenchmarkResult {
    _ = allocator;
    _ = rows;
    _ = cols;
    _ = bench_warmup;
    _ = bench_iters;
    return error.UnsupportedPlatform;
}
