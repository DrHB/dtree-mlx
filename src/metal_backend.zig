const builtin = @import("builtin");
const std = @import("std");
const gguf_store = @import("gguf_store.zig");
const metal_runtime = @import("metal/runtime.zig");

pub const default_max_cache_bytes: usize = 4 * 1024 * 1024 * 1024;

pub const Backend = struct {
    allocator: std.mem.Allocator,
    enabled: bool,
    max_cache_bytes: usize,
    cache_bytes: usize,
    ctx: ?metal_runtime.DenseContext,
    input_buffer: ?metal_runtime.DenseBuffer,
    output_buffer: ?metal_runtime.DenseBuffer,
    caches: std.ArrayList(CacheEntry),

    pub fn init(
        allocator: std.mem.Allocator,
        max_cache_bytes: usize,
    ) !Backend {
        if (builtin.os.tag != .macos) {
            return .{
                .allocator = allocator,
                .enabled = false,
                .max_cache_bytes = max_cache_bytes,
                .cache_bytes = 0,
                .ctx = null,
                .input_buffer = null,
                .output_buffer = null,
                .caches = .empty,
            };
        }

        return .{
            .allocator = allocator,
            .enabled = true,
            .max_cache_bytes = max_cache_bytes,
            .cache_bytes = 0,
            .ctx = try metal_runtime.DenseContext.init(),
            .input_buffer = null,
            .output_buffer = null,
            .caches = .empty,
        };
    }

    pub fn deinit(self: *Backend) void {
        for (self.caches.items) |*entry| {
            if (self.ctx) |*ctx| {
                ctx.releaseBuffer(&entry.buffer);
            }
            self.allocator.free(entry.name);
        }
        self.caches.deinit(self.allocator);

        if (self.ctx) |*ctx| {
            if (self.input_buffer) |*buffer| ctx.releaseBuffer(buffer);
            if (self.output_buffer) |*buffer| ctx.releaseBuffer(buffer);
            ctx.deinit();
        }

        self.* = undefined;
    }

    pub fn projectAllRows(
        self: *Backend,
        tensor: gguf_store.TensorView,
        input: []const f32,
        out: []f32,
    ) !bool {
        if (!self.enabled or self.ctx == null) return false;
        const entry = (try self.getOrCreateCache(tensor)) orelse return false;
        if (input.len != entry.cols) return error.InvalidInputBuffer;
        if (out.len < entry.rows) return error.OutputBufferTooSmall;

        try self.ensureBuffers(entry.cols, entry.rows);

        const input_buffer = &self.input_buffer.?;
        const output_buffer = &self.output_buffer.?;
        @memcpy(input_buffer.floats[0..entry.cols], input);

        try self.ctx.?.matvec(&entry.buffer, input_buffer, output_buffer, entry.rows, entry.cols);
        @memcpy(out[0..entry.rows], output_buffer.floats[0..entry.rows]);
        return true;
    }

    fn ensureBuffers(self: *Backend, input_len: usize, output_len: usize) !void {
        const ctx = &(self.ctx orelse return error.UnsupportedPlatform);

        if (self.input_buffer) |*buffer| {
            if (buffer.len < input_len) {
                ctx.releaseBuffer(buffer);
                self.input_buffer = null;
            }
        }
        if (self.input_buffer == null) {
            self.input_buffer = try ctx.allocBuffer(input_len);
        }

        if (self.output_buffer) |*buffer| {
            if (buffer.len < output_len) {
                ctx.releaseBuffer(buffer);
                self.output_buffer = null;
            }
        }
        if (self.output_buffer == null) {
            self.output_buffer = try ctx.allocBuffer(output_len);
        }
    }

    fn getOrCreateCache(
        self: *Backend,
        tensor: gguf_store.TensorView,
    ) !?*CacheEntry {
        for (self.caches.items) |*entry| {
            if (std.mem.eql(u8, entry.name, tensor.info.name)) return entry;
        }

        if (!shouldCacheTensor(tensor)) return null;

        const dense_bytes = try std.math.mul(usize, try std.math.mul(usize, tensor.row_count, tensor.row_len), @sizeOf(f32));
        if (dense_bytes > self.max_cache_bytes) return null;
        if (self.cache_bytes + dense_bytes > self.max_cache_bytes and !std.mem.eql(u8, tensor.info.name, "output.weight")) {
            return null;
        }

        const ctx = &(self.ctx orelse return error.UnsupportedPlatform);
        var buffer = try ctx.allocBuffer(tensor.row_count * tensor.row_len);
        errdefer ctx.releaseBuffer(&buffer);

        for (0..tensor.row_count) |row_idx| {
            const start = row_idx * tensor.row_len;
            try tensor.dequantizeRow(row_idx, buffer.floats[start .. start + tensor.row_len]);
        }

        const name = try self.allocator.dupe(u8, tensor.info.name);
        try self.caches.append(self.allocator, .{
            .name = name,
            .buffer = buffer,
            .rows = tensor.row_count,
            .cols = tensor.row_len,
            .dense_bytes = dense_bytes,
        });
        self.cache_bytes += dense_bytes;
        return &self.caches.items[self.caches.items.len - 1];
    }
};

const CacheEntry = struct {
    name: []u8,
    buffer: metal_runtime.DenseBuffer,
    rows: usize,
    cols: usize,
    dense_bytes: usize,
};

fn shouldCacheTensor(tensor: gguf_store.TensorView) bool {
    if (tensor.info.dimensions.len != 2) return false;
    if (tensor.row_count == 0 or tensor.row_len == 0) return false;

    const name = tensor.info.name;
    return std.mem.eql(u8, name, "output.weight");
}
