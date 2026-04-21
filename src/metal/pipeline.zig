const common = @import("common.zig");
const device = @import("device.zig");

const Error = common.Error;

pub const Pipeline = struct {
    handle: *common.c.dt_metal_pipeline,
    thread_execution_width: usize,
    max_total_threads_per_threadgroup: usize,

    pub fn deinit(self: *Pipeline) void {
        common.c.dt_metal_pipeline_destroy(self.handle);
        self.* = undefined;
    }
};

pub fn create(session: *const device.Session, function_name: [:0]const u8) Error!Pipeline {
    var info: common.c.dt_metal_pipeline_info = .{
        .handle = null,
        .thread_execution_width = 0,
        .max_total_threads_per_threadgroup = 0,
    };
    var error_message: [*c]u8 = null;
    try common.checkStatus(
        common.c.dt_metal_pipeline_create(session.handle, function_name.ptr, &info, &error_message),
        error_message,
    );

    return .{
        .handle = info.handle orelse return error.PipelineCreationFailed,
        .thread_execution_width = info.thread_execution_width,
        .max_total_threads_per_threadgroup = info.max_total_threads_per_threadgroup,
    };
}
