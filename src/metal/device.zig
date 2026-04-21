const std = @import("std");

const build_options = @import("metal_build_options");
const common = @import("common.zig");

const Error = common.Error;

const shader_files = [_][]const u8{
    "add_one.metal",
    "dense_matvec_f32.metal",
    "dmmv_q4_k.metal",
    "dmmv_q6_k.metal",
};

const shader_dir_env_var = "DTREE_MLX_METAL_SHADER_DIR";

pub const Session = struct {
    handle: *common.c.dt_metal_session,

    pub fn init() Error!Session {
        const allocator = std.heap.page_allocator;
        const source = try loadShaderSource(allocator);
        defer allocator.free(source);

        var handle: ?*common.c.dt_metal_session = null;
        var error_message: [*c]u8 = null;
        try common.checkStatus(
            common.c.dt_metal_session_create(source.ptr, &handle, &error_message),
            error_message,
        );

        return .{
            .handle = handle orelse return error.ShaderCompilationFailed,
        };
    }

    pub fn deinit(self: *Session) void {
        common.c.dt_metal_session_destroy(self.handle);
        self.* = undefined;
    }

    pub fn copyDeviceName(self: *const Session, allocator: std.mem.Allocator) Error![]u8 {
        var raw_name: [*c]u8 = null;
        var error_message: [*c]u8 = null;
        try common.checkStatus(
            common.c.dt_metal_session_copy_device_name(self.handle, &raw_name, &error_message),
            error_message,
        );

        if (raw_name == null) return error.StringCreationFailed;
        const owned_name = raw_name;
        defer common.c.dt_metal_string_free(owned_name);

        const span = std.mem.span(@as([*:0]const u8, @ptrCast(owned_name)));
        return allocator.dupe(u8, span);
    }
};

fn loadShaderSource(allocator: std.mem.Allocator) Error![:0]u8 {
    if (std.c.getenv(shader_dir_env_var)) |raw_dir| {
        return loadShaderSourceFromDir(allocator, std.mem.span(raw_dir));
    }

    if (try tryExecutableShaderDir(allocator)) |dir_path| {
        defer allocator.free(dir_path);
        if (loadShaderSourceFromDir(allocator, dir_path)) |source| {
            return source;
        } else |err| switch (err) {
            error.ShaderSourceDirectoryNotFound => {},
            else => return err,
        }
    }

    return loadShaderSourceFromDir(allocator, build_options.project_shader_dir);
}

fn tryExecutableShaderDir(allocator: std.mem.Allocator) Error!?[]u8 {
    var raw_dir: [*c]u8 = null;
    var error_message: [*c]u8 = null;
    const status = common.c.dt_metal_copy_executable_dir(&raw_dir, &error_message);
    if (status != common.c.DT_METAL_STATUS_OK) {
        common.freeMessage(error_message);
        return null;
    }

    if (raw_dir == null) return null;
    const owned_dir = raw_dir;
    defer common.c.dt_metal_string_free(owned_dir);

    const exe_dir = std.mem.span(@as([*:0]const u8, @ptrCast(owned_dir)));
    return try std.fs.path.join(allocator, &.{ exe_dir, "shaders", "metal" });
}

fn loadShaderSourceFromDir(allocator: std.mem.Allocator, dir_path: []const u8) Error![:0]u8 {
    var combined: std.ArrayList(u8) = .empty;
    defer combined.deinit(allocator);

    for (shader_files, 0..) |file_name, idx| {
        const file_path = try std.fs.path.join(allocator, &.{ dir_path, file_name });
        defer allocator.free(file_path);

        const contents = readFileAlloc(allocator, file_path) catch |err| switch (err) {
            error.ShaderSourceDirectoryNotFound => {
                return if (idx == 0) error.ShaderSourceDirectoryNotFound else error.ShaderSourceLoadFailed;
            },
            else => return err,
        };
        defer allocator.free(contents);

        try combined.appendSlice(allocator, contents);
        if (contents.len == 0 or contents[contents.len - 1] != '\n') {
            try combined.append(allocator, '\n');
        }
    }

    return combined.toOwnedSliceSentinel(allocator, 0);
}

fn readFileAlloc(allocator: std.mem.Allocator, path: []const u8) Error![]u8 {
    const fd = std.posix.openat(std.c.AT.FDCWD, path, .{ .ACCMODE = .RDONLY }, 0) catch |err| switch (err) {
        error.FileNotFound, error.NotDir => return error.ShaderSourceDirectoryNotFound,
        else => return error.ShaderSourceLoadFailed,
    };
    defer _ = std.c.close(fd);

    var file_bytes: std.ArrayList(u8) = .empty;
    defer file_bytes.deinit(allocator);

    var buf: [4096]u8 = undefined;
    while (true) {
        const read_len = std.posix.read(fd, &buf) catch return error.ShaderSourceLoadFailed;
        if (read_len == 0) break;
        try file_bytes.appendSlice(allocator, buf[0..read_len]);
    }

    return file_bytes.toOwnedSlice(allocator);
}
