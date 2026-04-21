const std = @import("std");

pub const ValueType = enum(i32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

pub const ArrayPreview = union(enum) {
    none,
    strings: []const []const u8,
    uints: []const u64,
    ints: []const i64,
    floats: []const f64,
    bools: []const bool,
};

pub const ArrayValue = struct {
    element_type: ValueType,
    len: u64,
    preview: ArrayPreview,
};

pub const Value = union(enum) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool: bool,
    string: []const u8,
    array: ArrayValue,
    uint64: u64,
    int64: i64,
    float64: f64,
};

pub const KvEntry = struct {
    key: []const u8,
    value: Value,
};

pub const TensorInfo = struct {
    name: []const u8,
    dimensions: []const u64,
    ggml_type: i32,
    offset: u64,
};

pub const File = struct {
    path: []const u8,
    version: u32,
    kv_count: u64,
    tensor_count: u64,
    alignment: u32,
    data_offset: u64,
    kv_entries: []const KvEntry,
    tensors: []const TensorInfo,

    pub fn findKey(self: *const File, key: []const u8) ?*const KvEntry {
        for (self.kv_entries) |*entry| {
            if (std.mem.eql(u8, entry.key, key)) {
                return entry;
            }
        }
        return null;
    }

    pub fn deinit(self: *File, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        for (self.kv_entries) |entry| {
            allocator.free(entry.key);
            freeValue(allocator, entry.value);
        }
        allocator.free(self.kv_entries);

        for (self.tensors) |tensor| {
            allocator.free(tensor.name);
            allocator.free(tensor.dimensions);
        }
        allocator.free(self.tensors);
        self.* = undefined;
    }
};

pub fn valueTypeName(value_type: ValueType) []const u8 {
    return switch (value_type) {
        .uint8 => "uint8",
        .int8 => "int8",
        .uint16 => "uint16",
        .int16 => "int16",
        .uint32 => "uint32",
        .int32 => "int32",
        .float32 => "float32",
        .bool => "bool",
        .string => "string",
        .array => "array",
        .uint64 => "uint64",
        .int64 => "int64",
        .float64 => "float64",
    };
}

pub fn ggmlTypeName(ggml_type: i32) []const u8 {
    return switch (ggml_type) {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        9 => "q8_1",
        10 => "q2_K",
        11 => "q3_K",
        12 => "q4_K",
        13 => "q5_K",
        14 => "q6_K",
        15 => "q8_K",
        16 => "iq2_xxs",
        17 => "iq2_xs",
        18 => "iq3_xxs",
        19 => "iq1_s",
        20 => "iq4_nl",
        21 => "iq3_s",
        22 => "iq2_s",
        23 => "iq4_xs",
        24 => "i8",
        25 => "i16",
        26 => "i32",
        27 => "i64",
        28 => "f64",
        29 => "iq1_m",
        30 => "bf16",
        34 => "tq1_0",
        35 => "tq2_0",
        39 => "mxfp4",
        40 => "nvfp4",
        41 => "q1_0",
        else => "unknown",
    };
}

pub fn formatValue(writer: *std.Io.Writer, value: Value) !void {
    switch (value) {
        .uint8 => |v| try writer.print("{d}", .{v}),
        .int8 => |v| try writer.print("{d}", .{v}),
        .uint16 => |v| try writer.print("{d}", .{v}),
        .int16 => |v| try writer.print("{d}", .{v}),
        .uint32 => |v| try writer.print("{d}", .{v}),
        .int32 => |v| try writer.print("{d}", .{v}),
        .float32 => |v| try writer.print("{d}", .{v}),
        .bool => |v| try writer.writeAll(if (v) "true" else "false"),
        .string => |v| try printString(writer, v),
        .uint64 => |v| try writer.print("{d}", .{v}),
        .int64 => |v| try writer.print("{d}", .{v}),
        .float64 => |v| try writer.print("{d}", .{v}),
        .array => |array| {
            try writer.print(
                "array<{s}>({d})",
                .{ valueTypeName(array.element_type), array.len },
            );
            try formatArrayPreview(writer, array);
        },
    }
}

pub fn loadFromPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !File {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    var reader_buffer: [64 * 1024]u8 = undefined;
    var file_reader = file.reader(io, &reader_buffer);
    var parser = Parser.init(allocator, &file_reader.interface, path);
    return parser.parse();
}

fn printString(writer: *std.Io.Writer, value: []const u8) !void {
    const max_len = 120;
    if (value.len <= max_len) {
        try writer.print("\"{s}\"", .{value});
    } else {
        try writer.print("\"{s}...\"", .{value[0..max_len]});
    }
}

fn formatArrayPreview(writer: *std.Io.Writer, array: ArrayValue) !void {
    switch (array.preview) {
        .none => return,
        .strings => |items| {
            try writer.writeAll(" [");
            for (items, 0..) |item, idx| {
                if (idx != 0) try writer.writeAll(", ");
                try printString(writer, item);
            }
            if (array.len > items.len) try writer.writeAll(", ...");
            try writer.writeAll("]");
        },
        .uints => |items| {
            try writer.writeAll(" [");
            for (items, 0..) |item, idx| {
                if (idx != 0) try writer.writeAll(", ");
                try writer.print("{d}", .{item});
            }
            if (array.len > items.len) try writer.writeAll(", ...");
            try writer.writeAll("]");
        },
        .ints => |items| {
            try writer.writeAll(" [");
            for (items, 0..) |item, idx| {
                if (idx != 0) try writer.writeAll(", ");
                try writer.print("{d}", .{item});
            }
            if (array.len > items.len) try writer.writeAll(", ...");
            try writer.writeAll("]");
        },
        .floats => |items| {
            try writer.writeAll(" [");
            for (items, 0..) |item, idx| {
                if (idx != 0) try writer.writeAll(", ");
                try writer.print("{d}", .{item});
            }
            if (array.len > items.len) try writer.writeAll(", ...");
            try writer.writeAll("]");
        },
        .bools => |items| {
            try writer.writeAll(" [");
            for (items, 0..) |item, idx| {
                if (idx != 0) try writer.writeAll(", ");
                try writer.writeAll(if (item) "true" else "false");
            }
            if (array.len > items.len) try writer.writeAll(", ...");
            try writer.writeAll("]");
        },
    }
}

fn freeValue(allocator: std.mem.Allocator, value: Value) void {
    switch (value) {
        .string => |text| allocator.free(text),
        .array => |array| switch (array.preview) {
            .none => {},
            .strings => |items| {
                for (items) |item| allocator.free(item);
                allocator.free(items);
            },
            .uints => |items| allocator.free(items),
            .ints => |items| allocator.free(items),
            .floats => |items| allocator.free(items),
            .bools => |items| allocator.free(items),
        },
        else => {},
    }
}

const Parser = struct {
    allocator: std.mem.Allocator,
    reader: *std.Io.Reader,
    path: []const u8,
    offset: u64 = 0,

    const max_numeric_preview = 8;
    const max_string_preview = 4;

    fn init(
        allocator: std.mem.Allocator,
        reader: *std.Io.Reader,
        path: []const u8,
    ) Parser {
        return .{
            .allocator = allocator,
            .reader = reader,
            .path = path,
        };
    }

    fn parse(self: *Parser) !File {
        const magic = try self.readOwnedBytes(4);
        defer self.allocator.free(magic);
        if (!std.mem.eql(u8, magic, "GGUF")) {
            return error.InvalidMagic;
        }

        const version = try self.takeInt(u32);
        if (version != 3) {
            return error.UnsupportedVersion;
        }

        const tensor_count = try self.takeNonNegativeCount();
        const kv_count = try self.takeNonNegativeCount();

        var kv_entries: std.ArrayList(KvEntry) = .empty;
        defer kv_entries.deinit(self.allocator);

        var alignment: u32 = 32;

        for (0..kv_count) |_| {
            const key = try self.readString();
            const value_type = try self.takeValueType();
            const value = try self.readValue(value_type);
            if (std.mem.eql(u8, key, "general.alignment") and value == .uint32) {
                alignment = value.uint32;
            }
            try kv_entries.append(self.allocator, .{
                .key = key,
                .value = value,
            });
        }

        var tensors: std.ArrayList(TensorInfo) = .empty;
        defer tensors.deinit(self.allocator);

        for (0..tensor_count) |_| {
            const name = try self.readString();
            const n_dimensions = try self.takeInt(u32);
            const dimensions = try self.allocator.alloc(u64, n_dimensions);

            for (dimensions) |*dimension| {
                const raw = try self.takeInt(i64);
                if (raw < 0) return error.InvalidDimension;
                dimension.* = @intCast(raw);
            }

            const ggml_type = try self.takeInt(i32);
            const data_tensor_offset = try self.takeInt(u64);
            try tensors.append(self.allocator, .{
                .name = name,
                .dimensions = dimensions,
                .ggml_type = ggml_type,
                .offset = data_tensor_offset,
            });
        }

        return .{
            .path = try self.allocator.dupe(u8, self.path),
            .version = version,
            .kv_count = kv_count,
            .tensor_count = tensor_count,
            .alignment = alignment,
            .data_offset = alignForward(self.offset, alignment),
            .kv_entries = try kv_entries.toOwnedSlice(self.allocator),
            .tensors = try tensors.toOwnedSlice(self.allocator),
        };
    }

    fn readValue(self: *Parser, value_type: ValueType) !Value {
        return switch (value_type) {
            .uint8 => .{ .uint8 = try self.takeInt(u8) },
            .int8 => .{ .int8 = try self.takeInt(i8) },
            .uint16 => .{ .uint16 = try self.takeInt(u16) },
            .int16 => .{ .int16 = try self.takeInt(i16) },
            .uint32 => .{ .uint32 = try self.takeInt(u32) },
            .int32 => .{ .int32 = try self.takeInt(i32) },
            .float32 => .{ .float32 = @bitCast(try self.takeInt(u32)) },
            .bool => .{ .bool = (try self.takeInt(u8)) != 0 },
            .string => .{ .string = try self.readString() },
            .array => .{ .array = try self.readArrayValue() },
            .uint64 => .{ .uint64 = try self.takeInt(u64) },
            .int64 => .{ .int64 = try self.takeInt(i64) },
            .float64 => .{ .float64 = @bitCast(try self.takeInt(u64)) },
        };
    }

    fn readArrayValue(self: *Parser) !ArrayValue {
        const element_type = try self.takeValueType();
        if (element_type == .array) {
            return error.NestedArraysUnsupported;
        }
        const len = try self.takeInt(u64);

        return switch (element_type) {
            .string => .{
                .element_type = element_type,
                .len = len,
                .preview = .{
                    .strings = try self.readStringArrayPreview(len),
                },
            },
            .bool => .{
                .element_type = element_type,
                .len = len,
                .preview = .{
                    .bools = try self.readBoolArrayPreview(len),
                },
            },
            .uint8, .uint16, .uint32, .uint64 => .{
                .element_type = element_type,
                .len = len,
                .preview = .{
                    .uints = try self.readUnsignedArrayPreview(element_type, len),
                },
            },
            .int8, .int16, .int32, .int64 => .{
                .element_type = element_type,
                .len = len,
                .preview = .{
                    .ints = try self.readSignedArrayPreview(element_type, len),
                },
            },
            .float32, .float64 => .{
                .element_type = element_type,
                .len = len,
                .preview = .{
                    .floats = try self.readFloatArrayPreview(element_type, len),
                },
            },
            .array => unreachable,
        };
    }

    fn readStringArrayPreview(self: *Parser, len: u64) ![]const []const u8 {
        const preview_len = @min(len, max_string_preview);
        const preview = try self.allocator.alloc([]const u8, preview_len);
        for (0..len) |idx| {
            if (idx < preview_len) {
                preview[idx] = try self.readString();
            } else {
                try self.skipString();
            }
        }
        return preview;
    }

    fn readBoolArrayPreview(self: *Parser, len: u64) ![]const bool {
        const preview_len = @min(len, max_numeric_preview);
        const preview = try self.allocator.alloc(bool, preview_len);
        for (0..preview_len) |idx| {
            preview[idx] = (try self.takeInt(u8)) != 0;
        }
        try self.skipBytes(len - preview_len);
        return preview;
    }

    fn readUnsignedArrayPreview(
        self: *Parser,
        element_type: ValueType,
        len: u64,
    ) ![]const u64 {
        const preview_len = @min(len, max_numeric_preview);
        const preview = try self.allocator.alloc(u64, preview_len);
        for (0..preview_len) |idx| {
            preview[idx] = switch (element_type) {
                .uint8 => try self.takeInt(u8),
                .uint16 => try self.takeInt(u16),
                .uint32 => try self.takeInt(u32),
                .uint64 => try self.takeInt(u64),
                else => unreachable,
            };
        }
        try self.skipBytes((len - preview_len) * byteLenForValueType(element_type));
        return preview;
    }

    fn readSignedArrayPreview(
        self: *Parser,
        element_type: ValueType,
        len: u64,
    ) ![]const i64 {
        const preview_len = @min(len, max_numeric_preview);
        const preview = try self.allocator.alloc(i64, preview_len);
        for (0..preview_len) |idx| {
            preview[idx] = switch (element_type) {
                .int8 => try self.takeInt(i8),
                .int16 => try self.takeInt(i16),
                .int32 => try self.takeInt(i32),
                .int64 => try self.takeInt(i64),
                else => unreachable,
            };
        }
        try self.skipBytes((len - preview_len) * byteLenForValueType(element_type));
        return preview;
    }

    fn readFloatArrayPreview(
        self: *Parser,
        element_type: ValueType,
        len: u64,
    ) ![]const f64 {
        const preview_len = @min(len, max_numeric_preview);
        const preview = try self.allocator.alloc(f64, preview_len);
        for (0..preview_len) |idx| {
            preview[idx] = switch (element_type) {
                .float32 => @as(f64, @floatCast(@as(f32, @bitCast(try self.takeInt(u32))))),
                .float64 => @bitCast(try self.takeInt(u64)),
                else => unreachable,
            };
        }
        try self.skipBytes((len - preview_len) * byteLenForValueType(element_type));
        return preview;
    }

    fn takeValueType(self: *Parser) !ValueType {
        const raw = try self.takeInt(i32);
        return std.enums.fromInt(ValueType, raw) orelse error.UnsupportedValueType;
    }

    fn readString(self: *Parser) ![]const u8 {
        const len = try self.takeInt(u64);
        const len_usize = try castUsize(len);
        return self.readOwnedBytes(len_usize);
    }

    fn skipString(self: *Parser) !void {
        const len = try self.takeInt(u64);
        try self.skipBytes(len);
    }

    fn takeNonNegativeCount(self: *Parser) !u64 {
        const raw = try self.takeInt(i64);
        if (raw < 0) return error.InvalidCount;
        return @intCast(raw);
    }

    fn takeInt(self: *Parser, comptime T: type) !T {
        const value = try self.reader.takeInt(T, .little);
        self.offset += @sizeOf(T);
        return value;
    }

    fn skipBytes(self: *Parser, len: u64) !void {
        var remaining = len;
        while (remaining > 0) {
            const chunk_u64 = @min(remaining, @as(u64, 1 << 20));
            const chunk = try castUsize(chunk_u64);
            const discarded = try self.reader.discard(.limited(chunk));
            if (discarded != chunk) return error.EndOfStream;
            self.offset += chunk_u64;
            remaining -= chunk_u64;
        }
    }

    fn readOwnedBytes(self: *Parser, len: usize) ![]u8 {
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(self.allocator);
        try self.reader.appendExact(self.allocator, &bytes, len);
        self.offset += len;
        return bytes.toOwnedSlice(self.allocator);
    }
};

fn byteLenForValueType(value_type: ValueType) u64 {
    return switch (value_type) {
        .uint8, .int8, .bool => 1,
        .uint16, .int16 => 2,
        .uint32, .int32, .float32 => 4,
        .uint64, .int64, .float64 => 8,
        .string, .array => unreachable,
    };
}

fn castUsize(value: u64) !usize {
    return std.math.cast(usize, value) orelse error.ValueTooLarge;
}

fn alignForward(value: u64, alignment: u32) u64 {
    if (alignment == 0) return value;
    const a = @as(u64, alignment);
    return ((value + a - 1) / a) * a;
}

test "parse minimal gguf metadata" {
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(std.testing.allocator);

    try appendSlice(&bytes, "GGUF");
    try appendInt(&bytes, u32, 3);
    try appendInt(&bytes, i64, 1);
    try appendInt(&bytes, i64, 2);

    try appendString(&bytes, "general.architecture");
    try appendInt(&bytes, i32, @intFromEnum(ValueType.string));
    try appendString(&bytes, "toy");

    try appendString(&bytes, "general.alignment");
    try appendInt(&bytes, i32, @intFromEnum(ValueType.uint32));
    try appendInt(&bytes, u32, 64);

    try appendString(&bytes, "weight");
    try appendInt(&bytes, u32, 2);
    try appendInt(&bytes, i64, 8);
    try appendInt(&bytes, i64, 16);
    try appendInt(&bytes, i32, 0);
    try appendInt(&bytes, u64, 0);

    var reader = std.Io.Reader.fixed(bytes.items);
    var parser = Parser.init(std.testing.allocator, &reader, "toy.gguf");
    var parsed = try parser.parse();
    defer parsed.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 3), parsed.version);
    try std.testing.expectEqual(@as(u64, 2), parsed.kv_count);
    try std.testing.expectEqual(@as(u64, 1), parsed.tensor_count);
    try std.testing.expectEqual(@as(u32, 64), parsed.alignment);
    try std.testing.expectEqualStrings("toy", parsed.findKey("general.architecture").?.value.string);
    try std.testing.expectEqualStrings("weight", parsed.tensors[0].name);
    try std.testing.expectEqual(@as(u64, 8), parsed.tensors[0].dimensions[0]);
    try std.testing.expectEqual(@as(u64, 16), parsed.tensors[0].dimensions[1]);
}

fn appendSlice(list: *std.ArrayList(u8), bytes: []const u8) !void {
    try list.appendSlice(std.testing.allocator, bytes);
}

fn appendInt(list: *std.ArrayList(u8), comptime T: type, value: T) !void {
    var buffer: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &buffer, value, .little);
    try appendSlice(list, &buffer);
}

fn appendString(list: *std.ArrayList(u8), text: []const u8) !void {
    try appendInt(list, u64, text.len);
    try appendSlice(list, text);
}
