const builtin = @import("builtin");
const std = @import("std");
const gguf_store = @import("gguf_store.zig");
const metal_runtime = @import("metal/runtime.zig");

pub const default_max_cache_bytes: usize = 4 * 1024 * 1024 * 1024;

pub const Stats = struct {
    metal_project_calls: usize = 0,
    metal_project_hits: usize = 0,
    metal_rowrange_calls: usize = 0,
    metal_rowrange_hits: usize = 0,
    cpu_fallback_calls: usize = 0,
    metal_cache_entries: usize = 0,
    metal_cache_bytes: usize = 0,
};

pub const Backend = struct {
    allocator: std.mem.Allocator,
    enabled: bool,
    max_cache_bytes: usize,
    cache_bytes: usize,
    device_name: []u8,
    stats: Stats,
    ctx: ?metal_runtime.DenseContext,
    matrix_buffer: ?metal_runtime.DenseBuffer,
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
                .device_name = try allocator.dupe(u8, "unavailable"),
                .stats = .{},
                .ctx = null,
                .matrix_buffer = null,
                .input_buffer = null,
                .output_buffer = null,
                .caches = .empty,
            };
        }

        var ctx = try metal_runtime.DenseContext.init();
        errdefer ctx.deinit();
        const device_name = try ctx.session.copyDeviceName(allocator);
        errdefer allocator.free(device_name);

        return .{
            .allocator = allocator,
            .enabled = true,
            .max_cache_bytes = max_cache_bytes,
            .cache_bytes = 0,
            .device_name = device_name,
            .stats = .{},
            .ctx = ctx,
            .matrix_buffer = null,
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
            if (self.matrix_buffer) |*buffer| ctx.releaseBuffer(buffer);
            if (self.input_buffer) |*buffer| ctx.releaseBuffer(buffer);
            if (self.output_buffer) |*buffer| ctx.releaseBuffer(buffer);
            ctx.deinit();
        }
        self.allocator.free(self.device_name);

        self.* = undefined;
    }

    pub fn deviceName(self: *const Backend) []const u8 {
        return self.device_name;
    }

    pub fn resetStats(self: *Backend) void {
        self.stats = .{
            .metal_cache_entries = self.caches.items.len,
            .metal_cache_bytes = self.cache_bytes,
        };
    }

    pub fn snapshotStats(self: *const Backend) Stats {
        var out = self.stats;
        out.metal_cache_entries = self.caches.items.len;
        out.metal_cache_bytes = self.cache_bytes;
        return out;
    }

    pub fn projectAllRows(
        self: *Backend,
        tensor: gguf_store.TensorView,
        input: []const f32,
        out: []f32,
    ) !bool {
        if (!self.enabled or self.ctx == null) return false;
        self.stats.metal_project_calls += 1;
        const entry = (try self.getOrCreateCache(tensor)) orelse {
            self.stats.cpu_fallback_calls += 1;
            return false;
        };
        if (input.len != entry.cols) return error.InvalidInputBuffer;
        if (out.len < entry.rows) return error.OutputBufferTooSmall;

        try self.ensureBuffers(entry.cols, entry.rows);

        const input_buffer = &self.input_buffer.?;
        const output_buffer = &self.output_buffer.?;
        @memcpy(input_buffer.floats[0..entry.cols], input);

        try self.ctx.?.matvec(&entry.buffer, input_buffer, output_buffer, entry.rows, entry.cols);
        @memcpy(out[0..entry.rows], output_buffer.floats[0..entry.rows]);
        self.stats.metal_project_hits += 1;
        return true;
    }

    pub fn projectRowRange(
        self: *Backend,
        tensor: gguf_store.TensorView,
        input: []const f32,
        row_start: usize,
        row_count: usize,
        out: []f32,
    ) !bool {
        if (!self.enabled or self.ctx == null) return false;
        self.stats.metal_rowrange_calls += 1;
        if (row_start > tensor.row_count or row_count > tensor.row_count - row_start) {
            return error.RowIndexOutOfRange;
        }
        if (input.len != tensor.row_len) return error.InvalidInputBuffer;
        if (out.len < row_count) return error.OutputBufferTooSmall;

        if (!shouldCacheTensor(tensor)) {
            self.stats.cpu_fallback_calls += 1;
            return false;
        }

        try self.ensureBuffers(tensor.row_len, row_count);
        try self.ensureMatrixBuffer(row_count * tensor.row_len);

        const matrix_buffer = &self.matrix_buffer.?;
        for (0..row_count) |row_offset| {
            const start = row_offset * tensor.row_len;
            try tensor.dequantizeRow(
                row_start + row_offset,
                matrix_buffer.floats[start .. start + tensor.row_len],
            );
        }

        const input_buffer = &self.input_buffer.?;
        const output_buffer = &self.output_buffer.?;
        @memcpy(input_buffer.floats[0..tensor.row_len], input);
        try self.ctx.?.matvec(
            matrix_buffer,
            input_buffer,
            output_buffer,
            row_count,
            tensor.row_len,
        );
        @memcpy(out[0..row_count], output_buffer.floats[0..row_count]);
        self.stats.metal_rowrange_hits += 1;
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

    fn ensureMatrixBuffer(self: *Backend, element_count: usize) !void {
        const ctx = &(self.ctx orelse return error.UnsupportedPlatform);

        if (self.matrix_buffer) |*buffer| {
            if (buffer.len < element_count) {
                ctx.releaseBuffer(buffer);
                self.matrix_buffer = null;
            }
        }
        if (self.matrix_buffer == null) {
            self.matrix_buffer = try ctx.allocBuffer(element_count);
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
    if (std.mem.eql(u8, name, "output.weight")) return true;

    return std.mem.endsWith(u8, name, ".attn_qkv.weight") or
        std.mem.endsWith(u8, name, ".attn_gate.weight") or
        std.mem.endsWith(u8, name, ".ssm_alpha.weight") or
        std.mem.endsWith(u8, name, ".ssm_beta.weight") or
        std.mem.endsWith(u8, name, ".ssm_out.weight") or
        std.mem.endsWith(u8, name, ".attn_q.weight") or
        std.mem.endsWith(u8, name, ".attn_k.weight") or
        std.mem.endsWith(u8, name, ".attn_v.weight") or
        std.mem.endsWith(u8, name, ".attn_output.weight") or
        std.mem.endsWith(u8, name, ".ffn_gate_inp.weight") or
        std.mem.endsWith(u8, name, ".ffn_up_shexp.weight") or
        std.mem.endsWith(u8, name, ".ffn_gate_shexp.weight") or
        std.mem.endsWith(u8, name, ".ffn_down_shexp.weight");
}
