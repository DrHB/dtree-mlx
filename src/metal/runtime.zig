const builtin = @import("builtin");
const std = @import("std");

const Id = ?*anyopaque;
const Sel = ?*anyopaque;

const metal_path: []const u8 = "/System/Library/Frameworks/Metal.framework/Versions/Current/Metal";
const foundation_path: []const u8 = "/System/Library/Frameworks/Foundation.framework/Versions/Current/Foundation";
const libobjc_path: []const u8 = "/usr/lib/libobjc.A.dylib";

const max_matvec_threadgroup_width: usize = 256;

const MetalSource =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\kernel void add_one(device float *values [[buffer(0)]],
    \\                    uint gid [[thread_position_in_grid]]) {
    \\    values[gid] = values[gid] + 1.0f;
    \\}
    \\
    \\kernel void dense_matvec_f32(device const float *matrix [[buffer(0)]],
    \\                             device const float *vector [[buffer(1)]],
    \\                             device float *output [[buffer(2)]],
    \\                             constant uint &cols [[buffer(3)]],
    \\                             uint2 gid [[thread_position_in_grid]],
    \\                             uint lane [[thread_index_in_threadgroup]],
    \\                             uint2 threads_per_group [[threads_per_threadgroup]]) {
    \\    const uint row = gid.y;
    \\    const uint group_width = threads_per_group.x;
    \\    threadgroup float partials[256];
    \\
    \\    float acc = 0.0f;
    \\    const uint base = row * cols;
    \\    for (uint col = gid.x; col < cols; col += group_width) {
    \\        acc = fma(matrix[base + col], vector[col], acc);
    \\    }
    \\
    \\    partials[lane] = acc;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (uint stride = group_width >> 1; stride > 0; stride >>= 1) {
    \\        if (lane < stride) {
    \\            partials[lane] += partials[lane + stride];
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    if (lane == 0) {
    \\        output[row] = partials[0];
    \\    }
    \\}
;

const DemoValues = [_]f32{ 0.0, 1.0, 2.0, 3.0 };

const MTLResourceStorageModeShared: usize = 0;

const MTLSize = extern struct {
    width: usize,
    height: usize,
    depth: usize,
};

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

pub const Error = std.DynLib.Error || error{
    MissingSymbol,
    UnsupportedPlatform,
    NoMetalDevice,
    StringCreationFailed,
    ShaderCompilationFailed,
    FunctionLookupFailed,
    PipelineCreationFailed,
    CommandQueueCreationFailed,
    CommandBufferCreationFailed,
    ComputeEncoderCreationFailed,
    BufferCreationFailed,
    BufferMapFailed,
    ValidationFailed,
    InvalidBenchIters,
    InvalidElementCount,
    InvalidRowCount,
    InvalidColCount,
    ClockGetTimeFailed,
    ClockOutOfRange,
};

const Symbols = struct {
    create_default_device: *const fn () callconv(.c) Id,
    objc_getClass: *const fn ([*:0]const u8) callconv(.c) Id,
    sel_registerName: *const fn ([*:0]const u8) callconv(.c) Sel,
    objc_release: *const fn (Id) callconv(.c) void,
    msg_send_id_0: *const fn (Id, Sel) callconv(.c) Id,
    msg_send_id_cstr: *const fn (Id, Sel, [*:0]const u8) callconv(.c) Id,
    msg_send_id_id: *const fn (Id, Sel, Id) callconv(.c) Id,
    msg_send_id_id_ptrid: *const fn (Id, Sel, Id, *Id) callconv(.c) Id,
    msg_send_id_id_id_ptrid: *const fn (Id, Sel, Id, Id, *Id) callconv(.c) Id,
    msg_send_id_usize_usize: *const fn (Id, Sel, usize, usize) callconv(.c) Id,
    msg_send_void_0: *const fn (Id, Sel) callconv(.c) void,
    msg_send_void_id: *const fn (Id, Sel, Id) callconv(.c) void,
    msg_send_void_id_usize_usize: *const fn (Id, Sel, Id, usize, usize) callconv(.c) void,
    msg_send_void_ptr_usize_usize: *const fn (Id, Sel, ?*const anyopaque, usize, usize) callconv(.c) void,
    msg_send_void_size_size: *const fn (Id, Sel, MTLSize, MTLSize) callconv(.c) void,
    msg_send_ptr_0: *const fn (Id, Sel) callconv(.c) ?*anyopaque,
    msg_send_cstr_0: *const fn (Id, Sel) callconv(.c) ?[*:0]const u8,
    msg_send_usize_0: *const fn (Id, Sel) callconv(.c) usize,
};

const Session = struct {
    libobjc: std.DynLib,
    foundation: std.DynLib,
    metal: std.DynLib,
    symbols: Symbols,
    device: Id,
    queue: Id,
    library: Id,

    fn deinit(self: *Session) void {
        release(&self.symbols, self.library);
        release(&self.symbols, self.queue);
        release(&self.symbols, self.device);
        self.metal.close();
        self.foundation.close();
        self.libobjc.close();
    }
};

const Pipeline = struct {
    symbols: *const Symbols,
    function: Id,
    object: Id,
    thread_execution_width: usize,
    max_total_threads_per_threadgroup: usize,

    fn deinit(self: *Pipeline) void {
        release(self.symbols, self.object);
        release(self.symbols, self.function);
    }
};

pub fn runBootstrap(allocator: std.mem.Allocator) Error!Report {
    if (builtin.os.tag != .macos) return error.UnsupportedPlatform;

    var session = try createSession();
    defer session.deinit();
    var pipeline = try createPipeline(&session, "add_one");
    defer pipeline.deinit();

    const buffer = try allocateSharedBuffer(&session, DemoValues.len);
    defer release(&session.symbols, buffer.object);
    @memcpy(buffer.floats[0..DemoValues.len], DemoValues[0..]);

    try dispatchAddOne(&session, &pipeline, buffer.object, DemoValues.len);

    var output: [DemoValues.len]f32 = undefined;
    @memcpy(output[0..], buffer.floats[0..DemoValues.len]);
    for (output, 0..) |value, idx| {
        const expected = DemoValues[idx] + 1.0;
        if (@abs(value - expected) > 0.0001) return error.ValidationFailed;
    }

    return .{
        .device_name = try copyNSString(
            allocator,
            &session.symbols,
            session.symbols.msg_send_id_0(session.device, sel(&session.symbols, "name")) orelse return error.StringCreationFailed,
        ),
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
    if (builtin.os.tag != .macos) return error.UnsupportedPlatform;
    if (element_count == 0) return error.InvalidElementCount;
    if (bench_iters == 0) return error.InvalidBenchIters;

    var session = try createSession();
    defer session.deinit();
    var pipeline = try createPipeline(&session, "add_one");
    defer pipeline.deinit();

    const buffer = try allocateSharedBuffer(&session, element_count);
    defer release(&session.symbols, buffer.object);
    fillPattern(buffer.floats, element_count);

    for (0..bench_warmup) |_| {
        try dispatchAddOne(&session, &pipeline, buffer.object, element_count);
    }

    const start_ns = try monotonicNowNs();
    for (0..bench_iters) |_| {
        try dispatchAddOne(&session, &pipeline, buffer.object, element_count);
    }
    const end_ns = try monotonicNowNs();
    const elapsed_s = @as(f64, @floatFromInt(end_ns - start_ns)) / @as(f64, std.time.ns_per_s);

    const expected_add = @as(f32, @floatFromInt(bench_warmup + bench_iters));
    var checksum: f64 = 0;
    for (0..element_count) |idx| {
        const expected = initialValue(idx) + expected_add;
        const actual = buffer.floats[idx];
        if (@abs(actual - expected) > 0.0001) return error.ValidationFailed;
        checksum += actual;
    }

    return .{
        .device_name = try copyNSString(
            allocator,
            &session.symbols,
            session.symbols.msg_send_id_0(session.device, sel(&session.symbols, "name")) orelse return error.StringCreationFailed,
        ),
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
    if (builtin.os.tag != .macos) return error.UnsupportedPlatform;
    if (rows == 0) return error.InvalidRowCount;
    if (cols == 0) return error.InvalidColCount;
    if (bench_iters == 0) return error.InvalidBenchIters;

    const matrix_count = rows * cols;
    var session = try createSession();
    defer session.deinit();
    var pipeline = try createPipeline(&session, "dense_matvec_f32");
    defer pipeline.deinit();

    const matrix = try allocateSharedBuffer(&session, matrix_count);
    defer release(&session.symbols, matrix.object);
    const vector = try allocateSharedBuffer(&session, cols);
    defer release(&session.symbols, vector.object);
    const output = try allocateSharedBuffer(&session, rows);
    defer release(&session.symbols, output.object);

    fillMatVecMatrix(matrix.floats, matrix_count);
    fillMatVecVector(vector.floats, cols);
    @memset(output.floats[0..rows], 0);

    for (0..bench_warmup) |_| {
        try dispatchDenseMatVec(&session, &pipeline, matrix.object, vector.object, output.object, rows, cols);
    }

    const start_ns = try monotonicNowNs();
    for (0..bench_iters) |_| {
        try dispatchDenseMatVec(&session, &pipeline, matrix.object, vector.object, output.object, rows, cols);
    }
    const end_ns = try monotonicNowNs();
    const elapsed_s = @as(f64, @floatFromInt(end_ns - start_ns)) / @as(f64, std.time.ns_per_s);
    const checksum = try validateMatVec(matrix.floats, vector.floats, output.floats, rows, cols);
    const operations_per_pass = @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(cols)) * 2.0;

    return .{
        .device_name = try copyNSString(
            allocator,
            &session.symbols,
            session.symbols.msg_send_id_0(session.device, sel(&session.symbols, "name")) orelse return error.StringCreationFailed,
        ),
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

const SharedBuffer = struct {
    object: Id,
    floats: [*]f32,
};

fn createSession() Error!Session {
    var libobjc = try std.DynLib.open(libobjc_path);
    errdefer libobjc.close();
    var foundation = try std.DynLib.open(foundation_path);
    errdefer foundation.close();
    var metal = try std.DynLib.open(metal_path);
    errdefer metal.close();

    const symbols = try loadSymbols(&libobjc, &metal);
    const device = symbols.create_default_device() orelse return error.NoMetalDevice;
    errdefer release(&symbols, device);

    const queue = symbols.msg_send_id_0(device, sel(&symbols, "newCommandQueue")) orelse return error.CommandQueueCreationFailed;
    errdefer release(&symbols, queue);

    const source = try nsString(&symbols, MetalSource);

    var compile_error: Id = null;
    const library = symbols.msg_send_id_id_id_ptrid(
        device,
        sel(&symbols, "newLibraryWithSource:options:error:"),
        source,
        null,
        &compile_error,
    ) orelse {
        if (compile_error) |err| try printNSError(err);
        return error.ShaderCompilationFailed;
    };
    errdefer release(&symbols, library);

    return .{
        .libobjc = libobjc,
        .foundation = foundation,
        .metal = metal,
        .symbols = symbols,
        .device = device,
        .queue = queue,
        .library = library,
    };
}

fn createPipeline(session: *Session, function_name: [:0]const u8) Error!Pipeline {
    const name = try nsString(&session.symbols, function_name);
    var pipeline_error: Id = null;
    const function = session.symbols.msg_send_id_id(session.library, sel(&session.symbols, "newFunctionWithName:"), name) orelse return error.FunctionLookupFailed;
    errdefer release(&session.symbols, function);

    const pipeline = session.symbols.msg_send_id_id_ptrid(
        session.device,
        sel(&session.symbols, "newComputePipelineStateWithFunction:error:"),
        function,
        &pipeline_error,
    ) orelse {
        if (pipeline_error) |err| try printNSError(err);
        return error.PipelineCreationFailed;
    };
    errdefer release(&session.symbols, pipeline);

    return .{
        .symbols = &session.symbols,
        .function = function,
        .object = pipeline,
        .thread_execution_width = session.symbols.msg_send_usize_0(pipeline, sel(&session.symbols, "threadExecutionWidth")),
        .max_total_threads_per_threadgroup = session.symbols.msg_send_usize_0(pipeline, sel(&session.symbols, "maxTotalThreadsPerThreadgroup")),
    };
}

fn allocateSharedBuffer(session: *Session, element_count: usize) Error!SharedBuffer {
    const byte_len = element_count * @sizeOf(f32);
    const buffer = session.symbols.msg_send_id_usize_usize(
        session.device,
        sel(&session.symbols, "newBufferWithLength:options:"),
        byte_len,
        MTLResourceStorageModeShared,
    ) orelse return error.BufferCreationFailed;
    const raw_contents = session.symbols.msg_send_ptr_0(buffer, sel(&session.symbols, "contents")) orelse return error.BufferMapFailed;
    return .{
        .object = buffer,
        .floats = @ptrCast(@alignCast(raw_contents)),
    };
}

fn dispatchAddOne(session: *Session, pipeline: *const Pipeline, buffer: Id, element_count: usize) Error!void {
    const command_buffer = session.symbols.msg_send_id_0(session.queue, sel(&session.symbols, "commandBuffer")) orelse return error.CommandBufferCreationFailed;
    defer release(&session.symbols, command_buffer);
    const encoder = session.symbols.msg_send_id_0(command_buffer, sel(&session.symbols, "computeCommandEncoder")) orelse return error.ComputeEncoderCreationFailed;
    defer release(&session.symbols, encoder);

    session.symbols.msg_send_void_id(encoder, sel(&session.symbols, "setComputePipelineState:"), pipeline.object);
    session.symbols.msg_send_void_id_usize_usize(encoder, sel(&session.symbols, "setBuffer:offset:atIndex:"), buffer, 0, 0);

    const threads_per_group_width = chooseThreadgroupWidth(
        pipeline.thread_execution_width,
        pipeline.max_total_threads_per_threadgroup,
        element_count,
    );
    session.symbols.msg_send_void_size_size(
        encoder,
        sel(&session.symbols, "dispatchThreads:threadsPerThreadgroup:"),
        .{
            .width = element_count,
            .height = 1,
            .depth = 1,
        },
        .{
            .width = threads_per_group_width,
            .height = 1,
            .depth = 1,
        },
    );
    session.symbols.msg_send_void_0(encoder, sel(&session.symbols, "endEncoding"));
    session.symbols.msg_send_void_0(command_buffer, sel(&session.symbols, "commit"));
    session.symbols.msg_send_void_0(command_buffer, sel(&session.symbols, "waitUntilCompleted"));
}

fn dispatchDenseMatVec(
    session: *Session,
    pipeline: *const Pipeline,
    matrix: Id,
    vector: Id,
    output: Id,
    rows: usize,
    cols: usize,
) Error!void {
    const cols_u32 = std.math.cast(u32, cols) orelse return error.InvalidColCount;
    const threadgroup_width = chooseMatVecThreadgroupWidth(
        pipeline.thread_execution_width,
        pipeline.max_total_threads_per_threadgroup,
        cols,
    );

    const command_buffer = session.symbols.msg_send_id_0(session.queue, sel(&session.symbols, "commandBuffer")) orelse return error.CommandBufferCreationFailed;
    defer release(&session.symbols, command_buffer);
    const encoder = session.symbols.msg_send_id_0(command_buffer, sel(&session.symbols, "computeCommandEncoder")) orelse return error.ComputeEncoderCreationFailed;
    defer release(&session.symbols, encoder);

    session.symbols.msg_send_void_id(encoder, sel(&session.symbols, "setComputePipelineState:"), pipeline.object);
    session.symbols.msg_send_void_id_usize_usize(encoder, sel(&session.symbols, "setBuffer:offset:atIndex:"), matrix, 0, 0);
    session.symbols.msg_send_void_id_usize_usize(encoder, sel(&session.symbols, "setBuffer:offset:atIndex:"), vector, 0, 1);
    session.symbols.msg_send_void_id_usize_usize(encoder, sel(&session.symbols, "setBuffer:offset:atIndex:"), output, 0, 2);
    session.symbols.msg_send_void_ptr_usize_usize(
        encoder,
        sel(&session.symbols, "setBytes:length:atIndex:"),
        @ptrCast(&cols_u32),
        @sizeOf(u32),
        3,
    );
    session.symbols.msg_send_void_size_size(
        encoder,
        sel(&session.symbols, "dispatchThreads:threadsPerThreadgroup:"),
        .{
            .width = threadgroup_width,
            .height = rows,
            .depth = 1,
        },
        .{
            .width = threadgroup_width,
            .height = 1,
            .depth = 1,
        },
    );
    session.symbols.msg_send_void_0(encoder, sel(&session.symbols, "endEncoding"));
    session.symbols.msg_send_void_0(command_buffer, sel(&session.symbols, "commit"));
    session.symbols.msg_send_void_0(command_buffer, sel(&session.symbols, "waitUntilCompleted"));
}

fn chooseThreadgroupWidth(thread_execution_width: usize, max_total_threads: usize, element_count: usize) usize {
    const preferred = @min(@max(thread_execution_width * 8, thread_execution_width), max_total_threads);
    const capped = @min(preferred, element_count);
    const rounded = @max(thread_execution_width, (capped / thread_execution_width) * thread_execution_width);
    return @min(rounded, element_count);
}

fn chooseMatVecThreadgroupWidth(thread_execution_width: usize, max_total_threads: usize, cols: usize) usize {
    const capped = @min(@min(max_total_threads, max_matvec_threadgroup_width), cols);
    if (capped <= 1) return 1;

    var candidate = highestPowerOfTwo(capped);
    if (candidate < thread_execution_width and thread_execution_width <= capped) {
        candidate = thread_execution_width;
    }
    return @max(candidate, 1);
}

fn highestPowerOfTwo(limit: usize) usize {
    var value = limit;
    var result: usize = 1;
    while (value > 1) : (value >>= 1) {
        result <<= 1;
    }
    return result;
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

fn loadSymbols(libobjc: *std.DynLib, metal: *std.DynLib) Error!Symbols {
    return .{
        .create_default_device = try lookupRequired(*const fn () callconv(.c) Id, metal, "MTLCreateSystemDefaultDevice"),
        .objc_getClass = try lookupRequired(*const fn ([*:0]const u8) callconv(.c) Id, libobjc, "objc_getClass"),
        .sel_registerName = try lookupRequired(*const fn ([*:0]const u8) callconv(.c) Sel, libobjc, "sel_registerName"),
        .objc_release = try lookupRequired(*const fn (Id) callconv(.c) void, libobjc, "objc_release"),
        .msg_send_id_0 = try lookupRequired(*const fn (Id, Sel) callconv(.c) Id, libobjc, "objc_msgSend"),
        .msg_send_id_cstr = try lookupRequired(*const fn (Id, Sel, [*:0]const u8) callconv(.c) Id, libobjc, "objc_msgSend"),
        .msg_send_id_id = try lookupRequired(*const fn (Id, Sel, Id) callconv(.c) Id, libobjc, "objc_msgSend"),
        .msg_send_id_id_ptrid = try lookupRequired(*const fn (Id, Sel, Id, *Id) callconv(.c) Id, libobjc, "objc_msgSend"),
        .msg_send_id_id_id_ptrid = try lookupRequired(*const fn (Id, Sel, Id, Id, *Id) callconv(.c) Id, libobjc, "objc_msgSend"),
        .msg_send_id_usize_usize = try lookupRequired(*const fn (Id, Sel, usize, usize) callconv(.c) Id, libobjc, "objc_msgSend"),
        .msg_send_void_0 = try lookupRequired(*const fn (Id, Sel) callconv(.c) void, libobjc, "objc_msgSend"),
        .msg_send_void_id = try lookupRequired(*const fn (Id, Sel, Id) callconv(.c) void, libobjc, "objc_msgSend"),
        .msg_send_void_id_usize_usize = try lookupRequired(*const fn (Id, Sel, Id, usize, usize) callconv(.c) void, libobjc, "objc_msgSend"),
        .msg_send_void_ptr_usize_usize = try lookupRequired(*const fn (Id, Sel, ?*const anyopaque, usize, usize) callconv(.c) void, libobjc, "objc_msgSend"),
        .msg_send_void_size_size = try lookupRequired(*const fn (Id, Sel, MTLSize, MTLSize) callconv(.c) void, libobjc, "objc_msgSend"),
        .msg_send_ptr_0 = try lookupRequired(*const fn (Id, Sel) callconv(.c) ?*anyopaque, libobjc, "objc_msgSend"),
        .msg_send_cstr_0 = try lookupRequired(*const fn (Id, Sel) callconv(.c) ?[*:0]const u8, libobjc, "objc_msgSend"),
        .msg_send_usize_0 = try lookupRequired(*const fn (Id, Sel) callconv(.c) usize, libobjc, "objc_msgSend"),
    };
}

fn lookupRequired(comptime T: type, lib: *std.DynLib, name: [:0]const u8) Error!T {
    return lib.lookup(T, name) orelse error.MissingSymbol;
}

fn sel(symbols: *const Symbols, name: [:0]const u8) Sel {
    return symbols.sel_registerName(name);
}

fn release(symbols: *const Symbols, object: Id) void {
    if (object != null) symbols.objc_release(object);
}

fn nsString(symbols: *const Symbols, value: [:0]const u8) Error!Id {
    const ns_string = symbols.objc_getClass("NSString") orelse return error.StringCreationFailed;
    return symbols.msg_send_id_cstr(ns_string, sel(symbols, "stringWithUTF8String:"), value) orelse return error.StringCreationFailed;
}

fn copyNSString(allocator: std.mem.Allocator, symbols: *const Symbols, string: Id) Error![]u8 {
    const raw = symbols.msg_send_cstr_0(string, sel(symbols, "UTF8String")) orelse return error.StringCreationFailed;
    return allocator.dupe(u8, std.mem.span(raw));
}

fn printNSError(err: Id) Error!void {
    const local_symbols = struct {
        fn helper(object: Id) ?[*:0]const u8 {
            var libobjc = std.DynLib.open(libobjc_path) catch return null;
            defer libobjc.close();
            const select = libobjc.lookup(*const fn ([*:0]const u8) callconv(.c) Sel, "sel_registerName") orelse return null;
            const msg_send_id = libobjc.lookup(*const fn (Id, Sel) callconv(.c) Id, "objc_msgSend") orelse return null;
            const msg_send_cstr = libobjc.lookup(*const fn (Id, Sel) callconv(.c) ?[*:0]const u8, "objc_msgSend") orelse return null;
            const desc = msg_send_id(object, select("localizedDescription")) orelse return null;
            return msg_send_cstr(desc, select("UTF8String"));
        }
    };
    if (local_symbols.helper(err)) |message| {
        std.debug.print("metal error: {s}\n", .{std.mem.span(message)});
    }
}

fn monotonicNowNs() !u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts) != 0) {
        return error.ClockGetTimeFailed;
    }
    const total_ns = @as(i128, ts.sec) * std.time.ns_per_s + @as(i128, ts.nsec);
    return std.math.cast(u64, total_ns) orelse error.ClockOutOfRange;
}
