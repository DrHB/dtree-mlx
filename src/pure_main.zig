const std = @import("std");
const gguf = @import("gguf.zig");
const gguf_store = @import("gguf_store.zig");
const metal_backend = @import("metal_backend.zig");
const ops = @import("ops.zig");
const parallel_rows = @import("parallel_rows.zig");
const single_token = @import("single_token.zig");
const cached_decode = @import("cached_decode.zig");

const Args = struct {
    model: []const u8 = "models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf",
    tensor_limit: usize = 12,
    kv_limit: usize = 24,
    all_kv: bool = false,
    all_tensors: bool = false,
    tensor_name: ?[]const u8 = null,
    row_index: usize = 0,
    value_limit: usize = 16,
    bench: bool = false,
    bench_rows: usize = 4096,
    bench_iters: usize = 10,
    bench_warmup: usize = 2,
    matvec: bool = false,
    token_id: ?usize = null,
    embed_tensor: []const u8 = "token_embd.weight",
    norm_tensor: ?[]const u8 = null,
    project_tensor: ?[]const u8 = null,
    full_token_pass: bool = false,
    cached_decode: bool = false,
    decode_steps: usize = 8,
    metal_decode: bool = false,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    var arg_it = try init.minimal.args.iterateAllocator(allocator);
    defer arg_it.deinit();

    const args = try parseArgs(allocator, &arg_it);

    var stdout_buffer: [8192]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &stdout_buffer);

    if (args.tensor_name != null) {
        var store = try gguf_store.Store.open(allocator, init.io, args.model);
        defer store.deinit(allocator);

        try printSummary(&stdout.interface, &store.parsed);
        if (args.bench and args.matvec) {
            try printMatVecBenchmark(allocator, &stdout.interface, &store, args);
        } else if (args.bench) {
            try printTensorBenchmark(allocator, &stdout.interface, &store, args);
        } else if (args.matvec) {
            try printMatVecDetail(allocator, &stdout.interface, &store, args);
        } else {
            try printMetadata(&stdout.interface, &store.parsed, args);
            try printTensorPreview(&stdout.interface, &store.parsed, args);
            try printTensorDetail(allocator, &stdout.interface, &store, args);
        }
        try stdout.interface.flush();
        return;
    }

    if (args.token_id != null and args.project_tensor != null) {
        var store = try gguf_store.Store.open(allocator, init.io, args.model);
        defer store.deinit(allocator);

        try printSummary(&stdout.interface, &store.parsed);
        if (args.bench) {
            try printTokenProjectionBenchmark(allocator, &stdout.interface, &store, args);
        } else {
            try printTokenProjectionDetail(allocator, &stdout.interface, &store, args);
        }
        try stdout.interface.flush();
        return;
    }

    if (args.full_token_pass and args.token_id != null) {
        var store = try gguf_store.Store.open(allocator, init.io, args.model);
        defer store.deinit(allocator);

        try printSummary(&stdout.interface, &store.parsed);
        if (args.bench) {
            try printSingleTokenBenchmark(allocator, &stdout.interface, &store, args);
        } else {
            try printSingleTokenDetail(allocator, &stdout.interface, &store, args);
        }
        try stdout.interface.flush();
        return;
    }

    if (args.cached_decode and args.token_id != null) {
        var store = try gguf_store.Store.open(allocator, init.io, args.model);
        defer store.deinit(allocator);

        try printSummary(&stdout.interface, &store.parsed);
        if (args.bench) {
            try printCachedDecodeBenchmark(allocator, &stdout.interface, &store, args);
        } else {
            try printCachedDecodeDetail(allocator, &stdout.interface, &store, args);
        }
        try stdout.interface.flush();
        return;
    }

    var model = try gguf.loadFromPath(allocator, init.io, args.model);
    defer model.deinit(allocator);

    try printSummary(&stdout.interface, &model);
    try printMetadata(&stdout.interface, &model, args);
    try stdout.interface.writeAll("\nTensor preview\n");
    try printTensorPreview(&stdout.interface, &model, args);
    try stdout.interface.flush();
}

fn printSummary(writer: *std.Io.Writer, model: *const gguf.File) !void {
    try writer.print("GGUF summary\n", .{});
    try writer.print("path: {s}\n", .{model.path});
    try writer.print("version: {d}\n", .{model.version});
    try writer.print("alignment: {d}\n", .{model.alignment});
    try writer.print("data_offset: {d}\n", .{model.data_offset});
    try writer.print("kv_count: {d}\n", .{model.kv_count});
    try writer.print("tensor_count: {d}\n", .{model.tensor_count});
    try writer.writeAll("\n");
}

fn printMetadata(writer: *std.Io.Writer, model: *const gguf.File, args: Args) !void {
    try writer.writeAll("Selected metadata\n");
    try printSelectedMetadata(writer, model);
    if (args.all_kv) {
        try writer.writeAll("\nAll metadata\n");
        try printAllMetadata(writer, model);
    } else {
        try writer.print(
            "\nUse --all-kv to print all {d} metadata entries.\n",
            .{model.kv_entries.len},
        );
    }
}

fn parseArgs(
    allocator: std.mem.Allocator,
    arg_it: *std.process.Args.Iterator,
) !Args {
    const defaults = Args{};
    var out = Args{
        .model = try allocator.dupe(u8, defaults.model),
    };

    _ = arg_it.next();
    while (arg_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "--model")) {
            out.model = try allocator.dupe(u8, arg_it.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--tensor-limit")) {
            out.tensor_limit = try std.fmt.parseInt(usize, arg_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--kv-limit")) {
            out.kv_limit = try std.fmt.parseInt(usize, arg_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--all-kv")) {
            out.all_kv = true;
        } else if (std.mem.eql(u8, arg, "--all-tensors")) {
            out.all_tensors = true;
        } else if (std.mem.eql(u8, arg, "--tensor")) {
            out.tensor_name = try allocator.dupe(u8, arg_it.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--row-index")) {
            out.row_index = try std.fmt.parseInt(usize, arg_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--value-limit")) {
            out.value_limit = try std.fmt.parseInt(usize, arg_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--bench")) {
            out.bench = true;
        } else if (std.mem.eql(u8, arg, "--bench-rows")) {
            out.bench_rows = try std.fmt.parseInt(usize, arg_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--bench-iters")) {
            out.bench_iters = try std.fmt.parseInt(usize, arg_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--bench-warmup")) {
            out.bench_warmup = try std.fmt.parseInt(usize, arg_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--matvec")) {
            out.matvec = true;
        } else if (std.mem.eql(u8, arg, "--token-id")) {
            out.token_id = try std.fmt.parseInt(usize, arg_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--embed-tensor")) {
            out.embed_tensor = try allocator.dupe(u8, arg_it.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--norm-tensor")) {
            out.norm_tensor = try allocator.dupe(u8, arg_it.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--project-tensor")) {
            out.project_tensor = try allocator.dupe(u8, arg_it.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--full-token-pass")) {
            out.full_token_pass = true;
        } else if (std.mem.eql(u8, arg, "--cached-decode")) {
            out.cached_decode = true;
        } else if (std.mem.eql(u8, arg, "--decode-steps")) {
            out.decode_steps = try std.fmt.parseInt(usize, arg_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--metal-decode")) {
            out.metal_decode = true;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            printUsage();
            return error.UnknownArgument;
        }
    }

    return out;
}

fn printSelectedMetadata(writer: *std.Io.Writer, model: *const gguf.File) !void {
    const selected = [_][]const u8{
        "general.architecture",
        "general.name",
        "general.basename",
        "general.size_label",
        "general.description",
        "general.file_type",
        "general.quantization_version",
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "qwen35moe.block_count",
        "qwen35moe.context_length",
        "qwen35moe.embedding_length",
        "qwen35moe.attention.head_count",
        "qwen35moe.attention.head_count_kv",
        "qwen35moe.rope.dimension_sections",
        "qwen35moe.expert_count",
        "qwen35moe.expert_used_count",
        "qwen35moe.full_attention_interval",
    };

    for (selected) |key| {
        if (model.findKey(key)) |entry| {
            try writer.print("{s} = ", .{entry.key});
            try gguf.formatValue(writer, entry.value);
            try writer.writeAll("\n");
        }
    }
}

fn printAllMetadata(writer: *std.Io.Writer, model: *const gguf.File) !void {
    for (model.kv_entries) |entry| {
        try writer.print("{s} = ", .{entry.key});
        try gguf.formatValue(writer, entry.value);
        try writer.writeAll("\n");
    }
}

fn printTensorPreview(
    writer: *std.Io.Writer,
    model: *const gguf.File,
    args: Args,
) !void {
    try writer.writeAll("\nTensor preview\n");
    const limit = if (args.all_tensors) model.tensors.len else @min(model.tensors.len, args.tensor_limit);
    for (model.tensors[0..limit], 0..) |tensor, idx| {
        try writer.print(
            "{d}. {s} type={s} dims=",
            .{ idx, tensor.name, gguf.ggmlTypeName(tensor.ggml_type) },
        );
        try printDimensions(writer, tensor.dimensions);
        try writer.print(" offset={d}\n", .{tensor.offset});
    }
    if (!args.all_tensors and model.tensors.len > limit) {
        try writer.print("... {d} more tensors omitted\n", .{model.tensors.len - limit});
    }
}

fn printTensorDetail(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    store: *const gguf_store.Store,
    args: Args,
) !void {
    const tensor_name = args.tensor_name orelse return;
    const tensor = try store.tensor(tensor_name);

    try writer.writeAll("\nTensor detail\n");
    try writer.print("name: {s}\n", .{tensor.info.name});
    try writer.print("type: {s}\n", .{gguf.ggmlTypeName(tensor.info.ggml_type)});
    try writer.writeAll("dims: ");
    try printDimensions(writer, tensor.info.dimensions);
    try writer.writeAll("\n");
    try writer.print("row_len: {d}\n", .{tensor.row_len});
    try writer.print("row_count: {d}\n", .{tensor.row_count});
    try writer.print("row_bytes: {d}\n", .{tensor.row_bytes});
    try writer.print("tensor_bytes: {d}\n", .{tensor.byte_len});

    const row = try allocator.alloc(f32, tensor.row_len);
    defer allocator.free(row);
    try tensor.dequantizeRow(args.row_index, row);

    const stats = rowStats(row);
    try writer.print("row_index: {d}\n", .{args.row_index});
    try writer.print("row_min: {d}\n", .{stats.min});
    try writer.print("row_max: {d}\n", .{stats.max});
    try writer.print("row_mean: {d}\n", .{stats.mean});
    try writer.writeAll("row_preview: [");

    const limit = @min(args.value_limit, row.len);
    for (row[0..limit], 0..) |value, idx| {
        if (idx != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{value});
    }
    if (row.len > limit) {
        try writer.writeAll(", ...");
    }
    try writer.writeAll("]\n");
}

fn printTensorBenchmark(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    store: *const gguf_store.Store,
    args: Args,
) !void {
    const tensor_name = args.tensor_name orelse return error.MissingTensorName;
    if (args.bench_rows == 0) return error.InvalidBenchRows;
    if (args.bench_iters == 0) return error.InvalidBenchIters;

    const tensor = try store.tensor(tensor_name);
    const row = try allocator.alloc(f32, tensor.row_len);
    defer allocator.free(row);

    var checksum: f64 = 0;
    for (0..args.bench_warmup) |iter_idx| {
        const start_row = try std.math.mul(usize, iter_idx, args.bench_rows);
        try runTensorBenchPass(tensor, start_row, args.bench_rows, row, &checksum);
    }

    const start_ns = try monotonicNowNs();
    for (0..args.bench_iters) |iter_idx| {
        const start_row = try std.math.mul(usize, iter_idx, args.bench_rows);
        try runTensorBenchPass(tensor, start_row, args.bench_rows, row, &checksum);
    }
    const end_ns = try monotonicNowNs();
    const elapsed_ns = end_ns - start_ns;
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, std.time.ns_per_s);
    const total_rows = try std.math.mul(usize, args.bench_rows, args.bench_iters);
    const total_elements = try std.math.mul(usize, total_rows, tensor.row_len);
    const total_input_bytes = try std.math.mul(usize, total_rows, tensor.row_bytes);

    try writer.writeAll("Tensor benchmark\n");
    try writer.print("name: {s}\n", .{tensor.info.name});
    try writer.print("type: {s}\n", .{gguf.ggmlTypeName(tensor.info.ggml_type)});
    try writer.writeAll("dims: ");
    try printDimensions(writer, tensor.info.dimensions);
    try writer.writeAll("\n");
    try writer.print("row_len: {d}\n", .{tensor.row_len});
    try writer.print("row_count: {d}\n", .{tensor.row_count});
    try writer.print("row_bytes: {d}\n", .{tensor.row_bytes});
    try writer.print("bench_rows: {d}\n", .{args.bench_rows});
    try writer.print("bench_iters: {d}\n", .{args.bench_iters});
    try writer.print("bench_warmup: {d}\n", .{args.bench_warmup});
    try writer.print("elapsed_s: {d}\n", .{elapsed_s});
    try writer.print("rows_per_s: {d}\n", .{@as(f64, @floatFromInt(total_rows)) / elapsed_s});
    try writer.print("elements_per_s: {d}\n", .{@as(f64, @floatFromInt(total_elements)) / elapsed_s});
    try writer.print("input_mib_per_s: {d}\n", .{bytesToMib(total_input_bytes) / elapsed_s});
    try writer.print(
        "full_tensor_sweeps_per_s: {d}\n",
        .{@as(f64, @floatFromInt(total_rows)) / elapsed_s / @as(f64, @floatFromInt(tensor.row_count))},
    );
    try writer.print("checksum: {d}\n", .{checksum});
    try writer.writeAll(
        "note: this is pure-Zig row decode throughput, not end-to-end generation tok/s yet.\n",
    );
}

fn printMatVecDetail(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    store: *const gguf_store.Store,
    args: Args,
) !void {
    const tensor_name = args.tensor_name orelse return error.MissingTensorName;
    const tensor = try store.tensor(tensor_name);
    const input = try allocator.alloc(f32, tensor.row_len);
    defer allocator.free(input);
    fillSyntheticInput(input);

    const rows_to_compute = @min(args.value_limit, tensor.row_count);
    const output = try allocator.alloc(f32, rows_to_compute);
    defer allocator.free(output);

    try parallel_rows.matvecRows(tensor, input, 0, rows_to_compute, output);

    try writer.writeAll("Tensor matvec\n");
    try writer.print("name: {s}\n", .{tensor.info.name});
    try writer.print("type: {s}\n", .{gguf.ggmlTypeName(tensor.info.ggml_type)});
    try writer.writeAll("dims: ");
    try printDimensions(writer, tensor.info.dimensions);
    try writer.writeAll("\n");
    try writer.print("row_len: {d}\n", .{tensor.row_len});
    try writer.print("rows_computed: {d}\n", .{rows_to_compute});
    try writer.writeAll("output_preview: [");
    for (output, 0..) |value, idx| {
        if (idx != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{value});
    }
    try writer.writeAll("]\n");
}

fn printMatVecBenchmark(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    store: *const gguf_store.Store,
    args: Args,
) !void {
    const tensor_name = args.tensor_name orelse return error.MissingTensorName;
    if (args.bench_rows == 0) return error.InvalidBenchRows;
    if (args.bench_iters == 0) return error.InvalidBenchIters;

    const tensor = try store.tensor(tensor_name);
    const input = try allocator.alloc(f32, tensor.row_len);
    defer allocator.free(input);
    fillSyntheticInput(input);

    const output = try allocator.alloc(f32, args.bench_rows);
    defer allocator.free(output);

    var checksum: f64 = 0;
    for (0..args.bench_warmup) |iter_idx| {
        const start_row = try std.math.mul(usize, iter_idx, args.bench_rows);
        try runMatVecBenchPass(tensor, start_row, args.bench_rows, input, output, &checksum);
    }

    const start_ns = try monotonicNowNs();
    for (0..args.bench_iters) |iter_idx| {
        const start_row = try std.math.mul(usize, iter_idx, args.bench_rows);
        try runMatVecBenchPass(tensor, start_row, args.bench_rows, input, output, &checksum);
    }
    const end_ns = try monotonicNowNs();
    const elapsed_ns = end_ns - start_ns;
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, std.time.ns_per_s);
    const total_rows = try std.math.mul(usize, args.bench_rows, args.bench_iters);
    const total_elements = try std.math.mul(usize, total_rows, tensor.row_len);
    const total_input_bytes = try std.math.mul(usize, total_rows, tensor.row_bytes);

    try writer.writeAll("Tensor matvec benchmark\n");
    try writer.print("name: {s}\n", .{tensor.info.name});
    try writer.print("type: {s}\n", .{gguf.ggmlTypeName(tensor.info.ggml_type)});
    try writer.writeAll("dims: ");
    try printDimensions(writer, tensor.info.dimensions);
    try writer.writeAll("\n");
    try writer.print("row_len: {d}\n", .{tensor.row_len});
    try writer.print("row_count: {d}\n", .{tensor.row_count});
    try writer.print("row_bytes: {d}\n", .{tensor.row_bytes});
    try writer.print("bench_rows: {d}\n", .{args.bench_rows});
    try writer.print("bench_iters: {d}\n", .{args.bench_iters});
    try writer.print("bench_warmup: {d}\n", .{args.bench_warmup});
    try writer.print("elapsed_s: {d}\n", .{elapsed_s});
    try writer.print("rows_per_s: {d}\n", .{@as(f64, @floatFromInt(total_rows)) / elapsed_s});
    try writer.print("elements_per_s: {d}\n", .{@as(f64, @floatFromInt(total_elements)) / elapsed_s});
    try writer.print("input_mib_per_s: {d}\n", .{bytesToMib(total_input_bytes) / elapsed_s});
    try writer.print(
        "full_tensor_matvecs_per_s: {d}\n",
        .{@as(f64, @floatFromInt(total_rows)) / elapsed_s / @as(f64, @floatFromInt(tensor.row_count))},
    );
    try writer.print("checksum: {d}\n", .{checksum});
    try writer.writeAll(
        "note: this is pure-Zig quantized matvec throughput for one tensor, still not full-model tok/s yet.\n",
    );
}

fn printTokenProjectionDetail(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    store: *const gguf_store.Store,
    args: Args,
) !void {
    const token_id = args.token_id orelse return error.MissingTokenId;
    const project_name = args.project_tensor orelse return error.MissingProjectTensor;

    const embed = try store.tensor(args.embed_tensor);
    if (token_id >= embed.row_count) return error.TokenIdOutOfRange;

    const epsilon = modelRmsEpsilon(&store.parsed);
    const hidden = try allocator.alloc(f32, embed.row_len);
    defer allocator.free(hidden);
    try embed.dequantizeRow(token_id, hidden);

    const normalized = if (args.norm_tensor != null)
        try loadNormalizedTokenHidden(allocator, store, hidden, args.norm_tensor.?, epsilon)
    else
        try allocator.dupe(f32, hidden);
    defer allocator.free(normalized);

    const project = try store.tensor(project_name);
    const rows_to_compute = @min(args.value_limit, project.row_count);
    const output = try allocator.alloc(f32, rows_to_compute);
    defer allocator.free(output);

    for (0..rows_to_compute) |idx| {
        output[idx] = try project.dotRow(idx, normalized);
    }

    try writer.writeAll("Token projection\n");
    try writer.print("token_id: {d}\n", .{token_id});
    try writer.print("embed_tensor: {s}\n", .{embed.info.name});
    if (args.norm_tensor) |name| {
        try writer.print("norm_tensor: {s}\n", .{name});
        try writer.print("rms_epsilon: {d}\n", .{epsilon});
    } else {
        try writer.writeAll("norm_tensor: none\n");
    }
    try writer.print("project_tensor: {s}\n", .{project.info.name});
    try writer.print("project_type: {s}\n", .{gguf.ggmlTypeName(project.info.ggml_type)});
    try writer.writeAll("project_dims: ");
    try printDimensions(writer, project.info.dimensions);
    try writer.writeAll("\n");
    try writer.print("rows_computed: {d}\n", .{rows_to_compute});
    try writer.writeAll("output_preview: [");
    for (output, 0..) |value, idx| {
        if (idx != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{value});
    }
    try writer.writeAll("]\n");
}

fn printTokenProjectionBenchmark(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    store: *const gguf_store.Store,
    args: Args,
) !void {
    const token_id = args.token_id orelse return error.MissingTokenId;
    const project_name = args.project_tensor orelse return error.MissingProjectTensor;
    if (args.bench_rows == 0) return error.InvalidBenchRows;
    if (args.bench_iters == 0) return error.InvalidBenchIters;

    const embed = try store.tensor(args.embed_tensor);
    if (token_id >= embed.row_count) return error.TokenIdOutOfRange;

    const epsilon = modelRmsEpsilon(&store.parsed);
    const hidden = try allocator.alloc(f32, embed.row_len);
    defer allocator.free(hidden);
    try embed.dequantizeRow(token_id, hidden);

    const normalized = if (args.norm_tensor != null)
        try loadNormalizedTokenHidden(allocator, store, hidden, args.norm_tensor.?, epsilon)
    else
        try allocator.dupe(f32, hidden);
    defer allocator.free(normalized);

    const project = try store.tensor(project_name);
    const rows_to_process = @min(args.bench_rows, project.row_count);
    const output = try allocator.alloc(f32, rows_to_process);
    defer allocator.free(output);

    var checksum: f64 = 0;
    for (0..args.bench_warmup) |_| {
        try runTokenProjectionPass(project, normalized, rows_to_process, output, &checksum);
    }

    const start_ns = try monotonicNowNs();
    for (0..args.bench_iters) |_| {
        try runTokenProjectionPass(project, normalized, rows_to_process, output, &checksum);
    }
    const end_ns = try monotonicNowNs();

    const elapsed_ns = end_ns - start_ns;
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, std.time.ns_per_s);
    const total_rows = try std.math.mul(usize, rows_to_process, args.bench_iters);
    const total_elements = try std.math.mul(usize, total_rows, project.row_len);
    const total_input_bytes = try std.math.mul(usize, total_rows, project.row_bytes);

    try writer.writeAll("Token projection benchmark\n");
    try writer.print("token_id: {d}\n", .{token_id});
    try writer.print("embed_tensor: {s}\n", .{embed.info.name});
    if (args.norm_tensor) |name| {
        try writer.print("norm_tensor: {s}\n", .{name});
        try writer.print("rms_epsilon: {d}\n", .{epsilon});
    } else {
        try writer.writeAll("norm_tensor: none\n");
    }
    try writer.print("project_tensor: {s}\n", .{project.info.name});
    try writer.print("project_type: {s}\n", .{gguf.ggmlTypeName(project.info.ggml_type)});
    try writer.writeAll("project_dims: ");
    try printDimensions(writer, project.info.dimensions);
    try writer.writeAll("\n");
    try writer.print("bench_rows: {d}\n", .{rows_to_process});
    try writer.print("bench_iters: {d}\n", .{args.bench_iters});
    try writer.print("bench_warmup: {d}\n", .{args.bench_warmup});
    try writer.print("elapsed_s: {d}\n", .{elapsed_s});
    try writer.print("rows_per_s: {d}\n", .{@as(f64, @floatFromInt(total_rows)) / elapsed_s});
    try writer.print("elements_per_s: {d}\n", .{@as(f64, @floatFromInt(total_elements)) / elapsed_s});
    try writer.print("input_mib_per_s: {d}\n", .{bytesToMib(total_input_bytes) / elapsed_s});
    try writer.print(
        "full_projection_passes_per_s: {d}\n",
        .{@as(f64, @floatFromInt(total_rows)) / elapsed_s / @as(f64, @floatFromInt(project.row_count))},
    );
    try writer.print("checksum: {d}\n", .{checksum});
    try writer.writeAll(
        "note: this is one-token embedding + optional RMSNorm + projection throughput, not full-model tok/s yet.\n",
    );
}

fn printSingleTokenDetail(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    store: *const gguf_store.Store,
    args: Args,
) !void {
    const token_id = args.token_id orelse return error.MissingTokenId;
    var backend_storage: ?metal_backend.Backend = null;
    defer if (backend_storage) |*backend| backend.deinit();
    if (args.metal_decode) {
        backend_storage = try metal_backend.Backend.init(allocator, metal_backend.default_max_cache_bytes);
    }

    var engine = try single_token.Engine.init(
        allocator,
        store,
        if (backend_storage) |*backend| backend else null,
    );
    defer engine.deinit();

    const top_limit = @max(@as(usize, 1), args.value_limit);
    const candidates = try allocator.alloc(single_token.OutputCandidate, top_limit);
    defer allocator.free(candidates);

    const result = try engine.run(token_id, candidates);

    try writer.writeAll("Single-token full forward\n");
    try writer.print("token_id: {d}\n", .{token_id});
    try writer.print("projection_backend: {s}\n", .{if (args.metal_decode) "metal-cache" else "cpu"});
    try writer.print("layers: {d}\n", .{engine.hparams.n_layers});
    try writer.print("embedding_length: {d}\n", .{engine.hparams.n_embd});
    try writer.writeAll("mode: fresh token, zero history, zero recurrent state\n");
    try writer.print("argmax_token_id: {d}\n", .{result.argmax_token_id});
    try writer.print("argmax_logit: {d}\n", .{result.argmax_logit});
    try writer.writeAll("top_logits: [");
    for (candidates[0..result.top_count], 0..) |candidate, idx| {
        if (idx != 0) try writer.writeAll(", ");
        try writer.print("{{token={d}, logit={d}}}", .{ candidate.token_id, candidate.logit });
    }
    try writer.writeAll("]\n");
    try writer.writeAll(
        "note: this is an exact fresh-token pass for one token with no cache/history. With --metal-decode, supported projection tensors run through the native Zig Metal backend.\n",
    );
}

fn printSingleTokenBenchmark(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    store: *const gguf_store.Store,
    args: Args,
) !void {
    const token_id = args.token_id orelse return error.MissingTokenId;
    if (args.bench_iters == 0) return error.InvalidBenchIters;

    var backend_storage: ?metal_backend.Backend = null;
    defer if (backend_storage) |*backend| backend.deinit();
    if (args.metal_decode) {
        backend_storage = try metal_backend.Backend.init(allocator, metal_backend.default_max_cache_bytes);
    }

    var engine = try single_token.Engine.init(
        allocator,
        store,
        if (backend_storage) |*backend| backend else null,
    );
    defer engine.deinit();

    const result = try engine.benchmark(token_id, args.bench_warmup, args.bench_iters);

    try writer.writeAll("Single-token full forward benchmark\n");
    try writer.print("token_id: {d}\n", .{token_id});
    try writer.print("projection_backend: {s}\n", .{if (args.metal_decode) "metal-cache" else "cpu"});
    try writer.print("layers: {d}\n", .{engine.hparams.n_layers});
    try writer.print("embedding_length: {d}\n", .{engine.hparams.n_embd});
    try writer.writeAll("mode: fresh token, zero history, zero recurrent state\n");
    try writer.print("bench_iters: {d}\n", .{args.bench_iters});
    try writer.print("bench_warmup: {d}\n", .{args.bench_warmup});
    try writer.print("elapsed_s: {d}\n", .{result.elapsed_s});
    try writer.print("fresh_token_passes_per_s: {d}\n", .{result.passes_per_s});
    try writer.print("fresh_token_tok_per_s: {d}\n", .{result.passes_per_s});
    try writer.print("checksum: {d}\n", .{result.checksum});
    try writer.writeAll(
        "note: this is a full one-token forward pass with no cache/history. With --metal-decode, supported projection tensors use the native Zig Metal backend.\n",
    );
}

fn printCachedDecodeDetail(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    store: *const gguf_store.Store,
    args: Args,
) !void {
    const token_id = args.token_id orelse return error.MissingTokenId;
    if (args.decode_steps == 0) return error.InvalidDecodeSteps;

    const top_limit = @max(@as(usize, 1), args.value_limit);
    var backend_storage: ?metal_backend.Backend = null;
    defer if (backend_storage) |*backend| backend.deinit();
    if (args.metal_decode) {
        backend_storage = try metal_backend.Backend.init(allocator, metal_backend.default_max_cache_bytes);
    }

    var engine = try cached_decode.Engine.init(
        allocator,
        store,
        if (backend_storage) |*backend| backend else null,
        args.decode_steps,
    );
    defer engine.deinit();

    const candidates = try allocator.alloc(cached_decode.OutputCandidate, top_limit);
    defer allocator.free(candidates);

    const result = try engine.runRepeated(token_id, args.decode_steps, candidates);

    try writer.writeAll("Cached decode\n");
    try writer.print("token_id: {d}\n", .{token_id});
    try writer.print("projection_backend: {s}\n", .{if (args.metal_decode) "metal-cache" else "cpu"});
    try writer.print("decode_steps: {d}\n", .{args.decode_steps});
    try writer.print("final_position: {d}\n", .{result.position});
    try writer.print("argmax_token_id: {d}\n", .{result.argmax_token_id});
    try writer.print("argmax_logit: {d}\n", .{result.argmax_logit});
    try writer.writeAll("top_logits: [");
    for (candidates[0..result.top_count], 0..) |candidate, idx| {
        if (idx != 0) try writer.writeAll(", ");
        try writer.print("{{token={d}, logit={d}}}", .{ candidate.token_id, candidate.logit });
    }
    try writer.writeAll("]\n");
    try writer.writeAll(
        "note: this is repeated-token cached decode with real model state. With --metal-decode, supported projection tensors run through the native Zig Metal backend.\n",
    );
}

fn printCachedDecodeBenchmark(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    store: *const gguf_store.Store,
    args: Args,
) !void {
    const token_id = args.token_id orelse return error.MissingTokenId;
    if (args.bench_iters == 0) return error.InvalidBenchIters;

    const max_seq_len = try std.math.add(usize, args.bench_warmup, args.bench_iters);
    var backend_storage: ?metal_backend.Backend = null;
    defer if (backend_storage) |*backend| backend.deinit();
    if (args.metal_decode) {
        backend_storage = try metal_backend.Backend.init(allocator, metal_backend.default_max_cache_bytes);
    }

    var engine = try cached_decode.Engine.init(
        allocator,
        store,
        if (backend_storage) |*backend| backend else null,
        max_seq_len,
    );
    defer engine.deinit();

    const result = try engine.benchmark(token_id, args.bench_warmup, args.bench_iters);

    try writer.writeAll("Cached decode benchmark\n");
    try writer.print("token_id: {d}\n", .{token_id});
    try writer.print("projection_backend: {s}\n", .{if (args.metal_decode) "metal-cache" else "cpu"});
    try writer.print("warmup_tokens: {d}\n", .{result.warmup_tokens});
    try writer.print("timed_tokens: {d}\n", .{result.timed_tokens});
    try writer.print("elapsed_s: {d}\n", .{result.elapsed_s});
    try writer.print("cached_decode_tok_per_s: {d}\n", .{result.tok_per_s});
    try writer.print("checksum: {d}\n", .{result.checksum});
    try writer.writeAll(
        "note: this is repeated-token autoregressive decode with real recurrent and KV caches. With --metal-decode, supported projection tensors run through the native Zig Metal backend.\n",
    );
}

fn runTensorBenchPass(
    tensor: gguf_store.TensorView,
    start_row: usize,
    rows_to_process: usize,
    row: []f32,
    checksum: *f64,
) !void {
    for (0..rows_to_process) |offset| {
        const row_index = (start_row + offset) % tensor.row_count;
        try tensor.dequantizeRow(row_index, row);
        checksum.* += row[0];
        checksum.* += row[row.len / 2];
        checksum.* += row[row.len - 1];
    }
}

fn runMatVecBenchPass(
    tensor: gguf_store.TensorView,
    start_row: usize,
    rows_to_process: usize,
    input: []const f32,
    output: []f32,
    checksum: *f64,
) !void {
    if (output.len < rows_to_process) return error.OutputBufferTooSmall;

    try parallel_rows.matvecRows(tensor, input, start_row, rows_to_process, output);
    for (output[0..rows_to_process]) |value| {
        checksum.* += value;
    }
}

fn runTokenProjectionPass(
    project: gguf_store.TensorView,
    hidden: []const f32,
    rows_to_process: usize,
    output: []f32,
    checksum: *f64,
) !void {
    if (output.len < rows_to_process) return error.OutputBufferTooSmall;

    try parallel_rows.matvecRows(project, hidden, 0, rows_to_process, output);
    for (output[0..rows_to_process]) |value| {
        checksum.* += value;
    }
}

const RowStats = struct {
    min: f32,
    max: f32,
    mean: f64,
};

fn rowStats(values: []const f32) RowStats {
    std.debug.assert(values.len > 0);

    var min = values[0];
    var max = values[0];
    var sum: f64 = 0;

    for (values) |value| {
        min = @min(min, value);
        max = @max(max, value);
        sum += value;
    }

    return .{
        .min = min,
        .max = max,
        .mean = sum / @as(f64, @floatFromInt(values.len)),
    };
}

fn printDimensions(writer: *std.Io.Writer, dims: []const u64) !void {
    try writer.writeAll("[");
    for (dims, 0..) |dim, idx| {
        if (idx != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{dim});
    }
    try writer.writeAll("]");
}

fn bytesToMib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}

fn fillSyntheticInput(out: []f32) void {
    var state: u64 = 0x4d595df4d0f33173;
    for (out) |*value| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        const upper: u32 = @truncate(state >> 32);
        const centered = @as(i32, @intCast(upper & 0xffff)) - 32768;
        value.* = @as(f32, @floatFromInt(centered)) / 32768.0;
    }
}

fn loadNormalizedTokenHidden(
    allocator: std.mem.Allocator,
    store: *const gguf_store.Store,
    hidden: []const f32,
    norm_tensor_name: []const u8,
    epsilon: f32,
) ![]f32 {
    const norm = try store.tensor(norm_tensor_name);
    if (norm.row_count != 1 or norm.row_len != hidden.len) {
        return error.InvalidNormTensor;
    }

    const weight = try allocator.alloc(f32, norm.row_len);
    defer allocator.free(weight);
    try norm.dequantizeRow(0, weight);

    const out = try allocator.alloc(f32, hidden.len);
    try ops.rmsNorm(out, hidden, weight, epsilon);
    return out;
}

fn modelRmsEpsilon(model: *const gguf.File) f32 {
    if (model.findKey("qwen35moe.attention.layer_norm_rms_epsilon")) |entry| {
        return switch (entry.value) {
            .float32 => entry.value.float32,
            .float64 => @floatCast(entry.value.float64),
            else => 1e-6,
        };
    }
    return 1e-6;
}

fn monotonicNowNs() !u64 {
    var ts: std.posix.timespec = undefined;
    if (std.c.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts) != 0) {
        return error.ClockGetTimeFailed;
    }

    const total_ns = @as(i128, ts.sec) * std.time.ns_per_s + @as(i128, ts.nsec);
    return std.math.cast(u64, total_ns) orelse error.ClockOutOfRange;
}

fn printUsage() void {
    std.debug.print(
        \\Usage: dtree-mlx-zig [options]
        \\
        \\Options:
        \\  --model PATH         GGUF model path.
        \\  --tensor-limit N     Tensor preview limit. Default: 12
        \\  --kv-limit N         Reserved for future filters. Default: 24
        \\  --all-kv             Print every GGUF metadata entry.
        \\  --all-tensors        Print every tensor descriptor.
        \\  --tensor NAME        Inspect one tensor and dequantize a row.
        \\  --row-index N        Row index for --tensor. Default: 0
        \\  --value-limit N      Row preview value count. Default: 16
        \\  --bench              Benchmark pure-Zig row decoding for --tensor.
        \\  --bench-rows N       Rows per benchmark iteration. Default: 4096
        \\  --bench-iters N      Timed benchmark iterations. Default: 10
        \\  --bench-warmup N     Warmup iterations. Default: 2
        \\  --matvec             Run quantized row-dot / matvec on --tensor.
        \\  --token-id N         Token id for embedding -> norm -> projection.
        \\  --embed-tensor NAME  Embedding tensor. Default: token_embd.weight
        \\  --norm-tensor NAME   Optional RMSNorm weight tensor.
        \\  --project-tensor NAME Projection tensor for one-token subgraph.
        \\  --full-token-pass    Run the full fresh-token pass through all layers.
        \\  --cached-decode      Run repeated-token cached decode with real model state.
        \\  --decode-steps N     Token steps for cached decode detail mode. Default: 8
        \\  --metal-decode       Use the native Zig Metal backend for supported projection tensors.
        \\  --help               Print this help text.
        \\
        \\Supported row decoding and row-dot today: f32, q4_K, q6_K.
        \\This default binary is pure Zig and does not depend on llama.cpp or ggml.
        \\
    , .{});
}
