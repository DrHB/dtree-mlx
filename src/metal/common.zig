const std = @import("std");

pub const Id = ?*anyopaque;

pub const c = @cImport({
    @cInclude("shim.h");
});

pub const matvec_threadgroup_width: usize = 128;
pub const matvec_rows_per_group: usize = 4;

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

pub fn checkStatus(status: c.dt_metal_status, error_message: [*c]u8) Error!void {
    if (status == c.DT_METAL_STATUS_OK) {
        freeMessage(error_message);
        return;
    }

    defer logAndFreeMessage(error_message);

    return switch (status) {
        c.DT_METAL_STATUS_OUT_OF_MEMORY => error.OutOfMemory,
        c.DT_METAL_STATUS_NO_DEVICE => error.NoMetalDevice,
        c.DT_METAL_STATUS_STRING_CREATION_FAILED => error.StringCreationFailed,
        c.DT_METAL_STATUS_SHADER_COMPILATION_FAILED => error.ShaderCompilationFailed,
        c.DT_METAL_STATUS_FUNCTION_LOOKUP_FAILED => error.FunctionLookupFailed,
        c.DT_METAL_STATUS_PIPELINE_CREATION_FAILED => error.PipelineCreationFailed,
        c.DT_METAL_STATUS_COMMAND_QUEUE_CREATION_FAILED => error.CommandQueueCreationFailed,
        c.DT_METAL_STATUS_COMMAND_BUFFER_CREATION_FAILED => error.CommandBufferCreationFailed,
        c.DT_METAL_STATUS_COMPUTE_ENCODER_CREATION_FAILED => error.ComputeEncoderCreationFailed,
        c.DT_METAL_STATUS_COMMAND_EXECUTION_FAILED => error.CommandExecutionFailed,
        c.DT_METAL_STATUS_BUFFER_CREATION_FAILED => error.BufferCreationFailed,
        c.DT_METAL_STATUS_BUFFER_MAP_FAILED => error.BufferMapFailed,
        else => error.ShaderSourceLoadFailed,
    };
}

pub fn freeMessage(error_message: [*c]u8) void {
    if (error_message != null) {
        c.dt_metal_string_free(error_message);
    }
}

fn logAndFreeMessage(error_message: [*c]u8) void {
    if (error_message != null) {
        const span = std.mem.span(@as([*:0]const u8, @ptrCast(error_message)));
        std.debug.print("metal error: {s}\n", .{span});
        c.dt_metal_string_free(error_message);
    }
}
