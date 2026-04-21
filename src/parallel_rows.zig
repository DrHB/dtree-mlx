const std = @import("std");
const gguf_store = @import("gguf_store.zig");

const max_parallel_workers: usize = 16;
const min_rows_per_worker: usize = 2048;

pub const ArgmaxResult = struct {
    row_index: usize,
    value: f32,
};

const MatvecTask = struct {
    tensor: gguf_store.TensorView,
    input: []const f32,
    output: []f32,
    start_row: usize,
    start_offset: usize,
    rows_to_process: usize,
};

const ArgmaxTask = struct {
    tensor: gguf_store.TensorView,
    input: []const f32,
    start_row: usize,
    end_row: usize,
    best_row: usize,
    best_value: f32,
};

pub fn matvecRows(
    tensor: gguf_store.TensorView,
    input: []const f32,
    start_row: usize,
    rows_to_process: usize,
    output: []f32,
) !void {
    if (output.len < rows_to_process) return error.OutputBufferTooSmall;

    const worker_count = chooseWorkerCount(rows_to_process);
    if (worker_count == 1) {
        var task = MatvecTask{
            .tensor = tensor,
            .input = input,
            .output = output,
            .start_row = start_row,
            .start_offset = 0,
            .rows_to_process = rows_to_process,
        };
        runMatvecRange(&task);
        return;
    }

    var tasks: [max_parallel_workers]MatvecTask = undefined;
    var threads: [max_parallel_workers - 1]std.Thread = undefined;

    var assigned: usize = 0;
    for (0..worker_count) |worker_idx| {
        const remaining_rows = rows_to_process - assigned;
        const remaining_workers = worker_count - worker_idx;
        const chunk_rows = remaining_rows / remaining_workers;
        tasks[worker_idx] = .{
            .tensor = tensor,
            .input = input,
            .output = output,
            .start_row = start_row + assigned,
            .start_offset = assigned,
            .rows_to_process = chunk_rows,
        };

        if (worker_idx + 1 == worker_count) {
            runMatvecRange(&tasks[worker_idx]);
        } else {
            threads[worker_idx] = try std.Thread.spawn(.{}, runMatvecRange, .{&tasks[worker_idx]});
        }
        assigned += chunk_rows;
    }

    for (threads[0 .. worker_count - 1]) |thread| {
        thread.join();
    }
}

pub fn argmaxRows(
    tensor: gguf_store.TensorView,
    input: []const f32,
) !ArgmaxResult {
    const worker_count = chooseWorkerCount(tensor.row_count);
    if (worker_count == 1) {
        var task = ArgmaxTask{
            .tensor = tensor,
            .input = input,
            .start_row = 0,
            .end_row = tensor.row_count,
            .best_row = 0,
            .best_value = -std.math.inf(f32),
        };
        runArgmaxRange(&task);
        return .{ .row_index = task.best_row, .value = task.best_value };
    }

    var tasks: [max_parallel_workers]ArgmaxTask = undefined;
    var threads: [max_parallel_workers - 1]std.Thread = undefined;

    var assigned: usize = 0;
    for (0..worker_count) |worker_idx| {
        const remaining_rows = tensor.row_count - assigned;
        const remaining_workers = worker_count - worker_idx;
        const chunk_rows = remaining_rows / remaining_workers;
        tasks[worker_idx] = .{
            .tensor = tensor,
            .input = input,
            .start_row = assigned,
            .end_row = assigned + chunk_rows,
            .best_row = 0,
            .best_value = -std.math.inf(f32),
        };

        if (worker_idx + 1 == worker_count) {
            runArgmaxRange(&tasks[worker_idx]);
        } else {
            threads[worker_idx] = try std.Thread.spawn(.{}, runArgmaxRange, .{&tasks[worker_idx]});
        }
        assigned += chunk_rows;
    }

    for (threads[0 .. worker_count - 1]) |thread| {
        thread.join();
    }

    var best_row: usize = 0;
    var best_value = -std.math.inf(f32);
    for (tasks[0..worker_count]) |task| {
        if (task.best_value > best_value) {
            best_value = task.best_value;
            best_row = task.best_row;
        }
    }

    return .{ .row_index = best_row, .value = best_value };
}

fn runMatvecRange(task: *MatvecTask) void {
    for (0..task.rows_to_process) |offset| {
        const row_index = (task.start_row + offset) % task.tensor.row_count;
        task.output[task.start_offset + offset] = task.tensor.dotRow(row_index, task.input) catch unreachable;
    }
}

fn runArgmaxRange(task: *ArgmaxTask) void {
    var best_row = task.start_row;
    var best_value = -std.math.inf(f32);
    for (task.start_row..task.end_row) |row_index| {
        const value = task.tensor.dotRow(row_index, task.input) catch unreachable;
        if (value > best_value) {
            best_value = value;
            best_row = row_index;
        }
    }
    task.best_row = best_row;
    task.best_value = best_value;
}

fn chooseWorkerCount(rows_to_process: usize) usize {
    if (rows_to_process < min_rows_per_worker) return 1;
    const cpu_count = std.Thread.getCpuCount() catch 1;
    const rows_limited = std.math.divCeil(usize, rows_to_process, min_rows_per_worker) catch 1;
    return @max(@as(usize, 1), @min(rows_limited, @min(cpu_count, max_parallel_workers)));
}
