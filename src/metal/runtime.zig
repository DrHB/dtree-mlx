const builtin = @import("builtin");
const std = @import("std");

const Id = ?*anyopaque;
const Sel = ?*anyopaque;

const metal_path: []const u8 = "/System/Library/Frameworks/Metal.framework/Versions/Current/Metal";
const foundation_path: []const u8 = "/System/Library/Frameworks/Foundation.framework/Versions/Current/Foundation";
const libobjc_path: []const u8 = "/usr/lib/libobjc.A.dylib";

const MetalSource =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\kernel void add_one(device float *values [[buffer(0)]],
    \\                    uint gid [[thread_position_in_grid]]) {
    \\    values[gid] = values[gid] + 1.0f;
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
    msg_send_void_size_size: *const fn (Id, Sel, MTLSize, MTLSize) callconv(.c) void,
    msg_send_ptr_0: *const fn (Id, Sel) callconv(.c) ?*anyopaque,
    msg_send_cstr_0: *const fn (Id, Sel) callconv(.c) ?[*:0]const u8,
    msg_send_usize_0: *const fn (Id, Sel) callconv(.c) usize,
};

pub fn runBootstrap(allocator: std.mem.Allocator) Error!Report {
    if (builtin.os.tag != .macos) return error.UnsupportedPlatform;

    var libobjc = try std.DynLib.open(libobjc_path);
    defer libobjc.close();
    var foundation = try std.DynLib.open(foundation_path);
    defer foundation.close();
    var metal = try std.DynLib.open(metal_path);
    defer metal.close();
    _ = &foundation;

    const symbols = try loadSymbols(&libobjc, &metal);

    const device = symbols.create_default_device() orelse return error.NoMetalDevice;
    defer release(&symbols, device);

    const queue = symbols.msg_send_id_0(device, sel(&symbols, "newCommandQueue")) orelse return error.CommandQueueCreationFailed;
    defer release(&symbols, queue);

    const source = try nsString(&symbols, MetalSource);
    const function_name = try nsString(&symbols, "add_one");

    var compile_error: Id = null;
    const library = symbols.msg_send_id_id_id_ptrid(
        device,
        sel(&symbols, "newLibraryWithSource:options:error:"),
        source,
        null,
        &compile_error,
    ) orelse {
        if (compile_error) |err| {
            defer release(&symbols, err);
            try printNSError(err);
        }
        return error.ShaderCompilationFailed;
    };
    defer release(&symbols, library);
    if (compile_error) |_| {}

    const function = symbols.msg_send_id_id(library, sel(&symbols, "newFunctionWithName:"), function_name) orelse return error.FunctionLookupFailed;
    defer release(&symbols, function);

    var pipeline_error: Id = null;
    const pipeline = symbols.msg_send_id_id_ptrid(
        device,
        sel(&symbols, "newComputePipelineStateWithFunction:error:"),
        function,
        &pipeline_error,
    ) orelse {
        if (pipeline_error) |err| {
            defer release(&symbols, err);
            try printNSError(err);
        }
        return error.PipelineCreationFailed;
    };
    defer release(&symbols, pipeline);
    if (pipeline_error) |_| {}

    const byte_len = DemoValues.len * @sizeOf(f32);
    const buffer = symbols.msg_send_id_usize_usize(
        device,
        sel(&symbols, "newBufferWithLength:options:"),
        byte_len,
        MTLResourceStorageModeShared,
    ) orelse return error.BufferCreationFailed;
    defer release(&symbols, buffer);

    const raw_contents = symbols.msg_send_ptr_0(buffer, sel(&symbols, "contents")) orelse return error.BufferMapFailed;
    const floats: [*]f32 = @ptrCast(@alignCast(raw_contents));
    @memcpy(floats[0..DemoValues.len], DemoValues[0..]);

    const command_buffer = symbols.msg_send_id_0(queue, sel(&symbols, "commandBuffer")) orelse return error.CommandBufferCreationFailed;

    const encoder = symbols.msg_send_id_0(command_buffer, sel(&symbols, "computeCommandEncoder")) orelse return error.ComputeEncoderCreationFailed;

    symbols.msg_send_void_id(encoder, sel(&symbols, "setComputePipelineState:"), pipeline);
    symbols.msg_send_void_id_usize_usize(encoder, sel(&symbols, "setBuffer:offset:atIndex:"), buffer, 0, 0);

    const threads_per_group = MTLSize{
        .width = DemoValues.len,
        .height = 1,
        .depth = 1,
    };
    const threadgroups = MTLSize{
        .width = 1,
        .height = 1,
        .depth = 1,
    };
    symbols.msg_send_void_size_size(
        encoder,
        sel(&symbols, "dispatchThreadgroups:threadsPerThreadgroup:"),
        threadgroups,
        threads_per_group,
    );
    symbols.msg_send_void_0(encoder, sel(&symbols, "endEncoding"));
    symbols.msg_send_void_0(command_buffer, sel(&symbols, "commit"));
    symbols.msg_send_void_0(command_buffer, sel(&symbols, "waitUntilCompleted"));

    var output: [DemoValues.len]f32 = undefined;
    @memcpy(output[0..], floats[0..DemoValues.len]);
    for (output, 0..) |value, idx| {
        const expected = DemoValues[idx] + 1.0;
        if (@abs(value - expected) > 0.0001) return error.ValidationFailed;
    }

    const device_name = try copyNSString(allocator, &symbols, symbols.msg_send_id_0(device, sel(&symbols, "name")) orelse return error.StringCreationFailed);
    const thread_execution_width = symbols.msg_send_usize_0(pipeline, sel(&symbols, "threadExecutionWidth"));
    const max_total_threads = symbols.msg_send_usize_0(pipeline, sel(&symbols, "maxTotalThreadsPerThreadgroup"));

    return .{
        .device_name = device_name,
        .thread_execution_width = thread_execution_width,
        .max_total_threads_per_threadgroup = max_total_threads,
        .input = DemoValues,
        .output = output,
    };
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
