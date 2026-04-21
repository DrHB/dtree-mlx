const std = @import("std");
const gguf_store = @import("gguf_store.zig");

pub const TensorDescriptor = struct {
    name: []const u8,
    ggml_type: i32,
    dimensions: []const u64,
    row_len: usize,
    row_count: usize,
    row_bytes: usize,
    byte_len: usize,
    data_offset: usize,
};

pub const Loader = struct {
    allocator: std.mem.Allocator,
    store: *const gguf_store.Store,
    descriptors: []TensorDescriptor,

    pub fn init(
        allocator: std.mem.Allocator,
        store: *const gguf_store.Store,
    ) !Loader {
        const descriptors = try allocator.alloc(TensorDescriptor, store.parsed.tensors.len);
        errdefer allocator.free(descriptors);

        for (store.parsed.tensors, 0..) |info, idx| {
            const view = try store.tensor(info.name);
            descriptors[idx] = .{
                .name = info.name,
                .ggml_type = info.ggml_type,
                .dimensions = info.dimensions,
                .row_len = view.row_len,
                .row_count = view.row_count,
                .row_bytes = view.row_bytes,
                .byte_len = view.byte_len,
                .data_offset = @intFromPtr(view.data.ptr) - @intFromPtr(store.mapping.ptr),
            };
        }

        return .{
            .allocator = allocator,
            .store = store,
            .descriptors = descriptors,
        };
    }

    pub fn deinit(self: *Loader) void {
        self.allocator.free(self.descriptors);
        self.* = undefined;
    }

    pub fn descriptor(self: *const Loader, name: []const u8) !*const TensorDescriptor {
        for (self.descriptors) |*entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry;
        }
        return error.TensorNotFound;
    }

    pub fn tensor(self: *const Loader, name: []const u8) !gguf_store.TensorView {
        return self.store.tensor(name);
    }
};
