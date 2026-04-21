const common = @import("common.zig");
const device = @import("device.zig");

const Error = common.Error;

pub const DenseBuffer = struct {
    object: common.Id,
    floats: [*]f32,
    len: usize,
};

pub const RawBuffer = struct {
    object: common.Id,
    len_bytes: usize,
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

pub fn wrapBytesNoCopy(session: *const device.Session, bytes: []const u8) Error!RawBuffer {
    var buffer_handle: ?*common.c.dt_metal_buffer = null;
    var error_message: [*c]u8 = null;
    try common.checkStatus(
        common.c.dt_metal_buffer_wrap_bytes_no_copy(
            session.handle,
            bytes.ptr,
            bytes.len,
            &buffer_handle,
            &error_message,
        ),
        error_message,
    );

    return .{
        .object = @ptrCast(buffer_handle orelse return error.BufferCreationFailed),
        .len_bytes = bytes.len,
    };
}

pub fn releaseRawBuffer(buffer: *RawBuffer) void {
    if (buffer.object) |object| {
        common.c.dt_metal_buffer_destroy(@ptrCast(@alignCast(object)));
    }
    buffer.* = undefined;
}

pub fn handleRaw(buffer: *const RawBuffer) *common.c.dt_metal_buffer {
    return @ptrCast(@alignCast(buffer.object orelse unreachable));
}
