const std = @import("std");
const gguf = @import("gguf.zig");
const quant = @import("quant.zig");

pub const Store = struct {
    parsed: gguf.File,
    mapping: []align(std.heap.page_size_min) const u8,

    pub fn open(
        allocator: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
    ) !Store {
        var file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);

        const stat = try file.stat(io);
        const size = std.math.cast(usize, stat.size) orelse return error.FileTooBig;
        const mapping = try std.posix.mmap(
            null,
            std.mem.alignForward(usize, size, std.heap.pageSize()),
            .{ .READ = true },
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );
        errdefer std.posix.munmap(mapping);

        var parsed = try gguf.loadFromPath(allocator, io, path);
        errdefer parsed.deinit(allocator);

        return .{
            .parsed = parsed,
            .mapping = mapping,
        };
    }

    pub fn deinit(self: *Store, allocator: std.mem.Allocator) void {
        std.posix.munmap(self.mapping);
        self.parsed.deinit(allocator);
        self.* = undefined;
    }

    pub fn tensor(self: *const Store, name: []const u8) !TensorView {
        for (self.parsed.tensors) |*info| {
            if (std.mem.eql(u8, info.name, name)) {
                return try self.tensorView(info);
            }
        }
        return error.TensorNotFound;
    }

    fn tensorView(self: *const Store, info: *const gguf.TensorInfo) !TensorView {
        if (info.dimensions.len == 0) {
            return error.InvalidTensorShape;
        }

        const row_len = try castUsize(info.dimensions[0]);
        var row_count: usize = 1;
        for (info.dimensions[1..]) |dimension| {
            row_count = try std.math.mul(usize, row_count, try castUsize(dimension));
        }

        const row_bytes = try quant.rowBytes(info.ggml_type, row_len);
        const byte_len = try std.math.mul(usize, row_bytes, row_count);
        const data_start_u64 = try std.math.add(u64, self.parsed.data_offset, info.offset);
        const data_start = try castUsize(data_start_u64);
        const data_end = try std.math.add(usize, data_start, byte_len);
        if (data_end > self.mapping.len) {
            return error.TensorOutOfBounds;
        }

        return .{
            .info = info,
            .data = self.mapping[data_start..data_end],
            .row_len = row_len,
            .row_count = row_count,
            .row_bytes = row_bytes,
            .byte_len = byte_len,
        };
    }
};

pub const TensorView = struct {
    info: *const gguf.TensorInfo,
    data: []const u8,
    row_len: usize,
    row_count: usize,
    row_bytes: usize,
    byte_len: usize,

    pub fn row(self: *const TensorView, row_index: usize) ![]const u8 {
        if (row_index >= self.row_count) {
            return error.RowOutOfRange;
        }

        const start = try std.math.mul(usize, row_index, self.row_bytes);
        const end = try std.math.add(usize, start, self.row_bytes);
        return self.data[start..end];
    }

    pub fn dequantizeRow(
        self: *const TensorView,
        row_index: usize,
        out: []f32,
    ) !void {
        if (out.len != self.row_len) {
            return error.InvalidOutputBuffer;
        }

        const row_bytes = try self.row(row_index);
        try quant.dequantizeRow(self.info.ggml_type, row_bytes, out);
    }

    pub fn dotRow(
        self: *const TensorView,
        row_index: usize,
        input: []const f32,
    ) !f32 {
        if (input.len != self.row_len) {
            return error.InvalidInputBuffer;
        }

        const row_bytes = try self.row(row_index);
        return quant.dotRow(self.info.ggml_type, row_bytes, input);
    }

};

fn castUsize(value: u64) !usize {
    return std.math.cast(usize, value) orelse error.ValueTooLarge;
}
