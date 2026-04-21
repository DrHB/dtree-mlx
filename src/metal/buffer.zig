const common = @import("common.zig");
const device = @import("device.zig");

const Error = common.Error;

pub const DenseBuffer = struct {
    object: common.Id,
    floats: [*]f32,
    len: usize,
};

pub fn allocateSharedBuffer(session: *const device.Session, element_count: usize) Error!DenseBuffer {
    var info: common.c.dt_metal_buffer_info = .{
        .handle = null,
        .floats = null,
    };
    var error_message: [*c]u8 = null;
    try common.checkStatus(
        common.c.dt_metal_buffer_create(session.handle, element_count, &info, &error_message),
        error_message,
    );

    const buffer_handle = info.handle orelse return error.BufferCreationFailed;
    const floats = info.floats orelse return error.BufferMapFailed;
    return .{
        .object = @ptrCast(buffer_handle),
        .floats = @ptrCast(floats),
        .len = element_count,
    };
}

pub fn releaseBuffer(buffer: *DenseBuffer) void {
    if (buffer.object) |object| {
        common.c.dt_metal_buffer_destroy(@ptrCast(@alignCast(object)));
    }
    buffer.* = undefined;
}

pub fn handle(buffer: *const DenseBuffer) *common.c.dt_metal_buffer {
    return @ptrCast(@alignCast(buffer.object orelse unreachable));
}
