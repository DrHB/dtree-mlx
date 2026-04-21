const std = @import("std");

const buffer_mod = @import("buffer.zig");
const common = @import("common.zig");
const device = @import("device.zig");
const pipeline_mod = @import("pipeline.zig");

const Error = common.Error;

pub fn dispatchAddOne(
    session: *const device.Session,
    pipeline: *const pipeline_mod.Pipeline,
    buffer: *const buffer_mod.DenseBuffer,
    element_count: usize,
) Error!void {
    const threads_per_group_width = chooseThreadgroupWidth(
        pipeline.thread_execution_width,
        pipeline.max_total_threads_per_threadgroup,
        element_count,
    );

    var error_message: [*c]u8 = null;
    try common.checkStatus(
        common.c.dt_metal_dispatch_add_one(
            session.handle,
            pipeline.handle,
            buffer_mod.handle(buffer),
            element_count,
            threads_per_group_width,
            &error_message,
        ),
        error_message,
    );
}

pub fn dispatchDenseMatVec(
    session: *const device.Session,
    pipeline: *const pipeline_mod.Pipeline,
    matrix: *const buffer_mod.DenseBuffer,
    vector: *const buffer_mod.DenseBuffer,
    output: *const buffer_mod.DenseBuffer,
    rows: usize,
    cols: usize,
) Error!void {
    const rows_u32 = std.math.cast(u32, rows) orelse return error.InvalidRowCount;
    const cols_u32 = std.math.cast(u32, cols) orelse return error.InvalidColCount;
    const threadgroup_width = chooseMatVecThreadgroupWidth(
        pipeline.thread_execution_width,
        pipeline.max_total_threads_per_threadgroup,
        cols,
    );
    const threadgroup_count = std.math.divCeil(usize, rows, common.matvec_rows_per_group) catch unreachable;

    var error_message: [*c]u8 = null;
    try common.checkStatus(
        common.c.dt_metal_dispatch_dense_matvec(
            session.handle,
            pipeline.handle,
            buffer_mod.handle(matrix),
            buffer_mod.handle(vector),
            buffer_mod.handle(output),
            threadgroup_count,
            threadgroup_width,
            rows_u32,
            cols_u32,
            &error_message,
        ),
        error_message,
    );
}

fn chooseThreadgroupWidth(thread_execution_width: usize, max_total_threads: usize, element_count: usize) usize {
    const preferred = @min(@max(thread_execution_width * 8, thread_execution_width), max_total_threads);
    const capped = @min(preferred, element_count);
    const rounded = @max(thread_execution_width, (capped / thread_execution_width) * thread_execution_width);
    return @min(rounded, element_count);
}

fn chooseMatVecThreadgroupWidth(thread_execution_width: usize, max_total_threads: usize, cols: usize) usize {
    _ = cols;
    if (thread_execution_width > common.matvec_threadgroup_width) return thread_execution_width;
    return @min(common.matvec_threadgroup_width, max_total_threads);
}
