const std = @import("std");
const zit = @import("../../zit.zig");
const Matrix = zit.Matrix;
const TensorError = zit.TensorError;
const metal = @import("metal");
const metal_context = @import("metal_context.zig");
const cpu_fallback = @import("../simd/SimdBackend.zig").backend;

pub fn matrixMultiply(_: *anyopaque, a: anytype, b: @TypeOf(a), out: *@TypeOf(a)) !void {
    const T = @TypeOf(a);
    const DataType = T.DataType;
    if (T != Matrix(DataType)) {
        @compileError("a, b, and out must be matrix types");
    }

    if (a.columns != b.rows) {
        return TensorError.ShapeMismatch;
    }
    if (!(out.rows == a.rows and out.columns == b.columns)) {
        return TensorError.ShapeMismatch;
    }

    // Get Metal context
    const ctx = metal_context.get() orelse {
        try cpu_fallback.vtable.matrixMultiply(@ptrFromInt(1), a, b, out);
        return;
    };

    // Only support f32 for now
    if (DataType != f32) {
        try cpu_fallback.vtable.matrixMultiply(@ptrFromInt(1), a, b, out);
        return;
    }

    // Create Metal buffers
    const element_size = @sizeOf(DataType);
    const a_size = a.rows * a.columns * element_size;
    const b_size = b.rows * b.columns * element_size;
    const result_size = out.rows * out.columns * element_size;
    const uint_size = @sizeOf(u32);

    var buffer_a = try ctx.device.createBuffer(a_size, .Shared);
    defer buffer_a.deinit();

    var buffer_b = try ctx.device.createBuffer(b_size, .Shared);
    defer buffer_b.deinit();

    var buffer_result = try ctx.device.createBuffer(result_size, .Shared);
    defer buffer_result.deinit();

    var buffer_m = try ctx.device.createBuffer(uint_size, .Shared);
    defer buffer_m.deinit();

    var buffer_n = try ctx.device.createBuffer(uint_size, .Shared);
    defer buffer_n.deinit();

    var buffer_k = try ctx.device.createBuffer(uint_size, .Shared);
    defer buffer_k.deinit();

    // Copy data to Metal buffers
    try buffer_a.copyFromSlice(std.mem.sliceAsBytes(a.data));
    try buffer_b.copyFromSlice(std.mem.sliceAsBytes(b.data));

    // Set dimensions
    const m_slice = buffer_m.getContentsSlice() orelse return TensorError.BackendError;
    const n_slice = buffer_n.getContentsSlice() orelse return TensorError.BackendError;
    const k_slice = buffer_k.getContentsSlice() orelse return TensorError.BackendError;

    const m: u32 = @intCast(a.rows); // Output rows
    const n: u32 = @intCast(b.columns); // Output columns
    const k: u32 = @intCast(a.columns); // Common dimension

    @memcpy(m_slice, std.mem.asBytes(&m));
    @memcpy(n_slice, std.mem.asBytes(&n));
    @memcpy(k_slice, std.mem.asBytes(&k));

    // Create command buffer and encoder
    var command_buffer = try ctx.command_queue.createCommandBuffer();
    defer command_buffer.deinit();

    var encoder = try command_buffer.createComputeCommandEncoder();
    defer encoder.deinit();

    // Set up compute command
    encoder.setComputePipelineState(ctx.matrix_multiply_pipeline);
    encoder.setBuffer(buffer_a, 0, 0);
    encoder.setBuffer(buffer_b, 0, 1);
    encoder.setBuffer(buffer_result, 0, 2);
    encoder.setBuffer(buffer_m, 0, 3);
    encoder.setBuffer(buffer_n, 0, 4);
    encoder.setBuffer(buffer_k, 0, 5);

    // Dispatch a 2D grid of threads - one thread per output element
    encoder.dispatchThreads(n, m, 1); // Width (columns), Height (rows)
    encoder.endEncoding();

    // Execute and wait for completion
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Copy result back to output matrix
    const result_slice = buffer_result.getContentsSlice() orelse return TensorError.BackendError;
    @memcpy(std.mem.sliceAsBytes(out.data), result_slice);
}

pub fn matrixTranspose(_: *anyopaque, m: anytype, out: *@TypeOf(m)) !void {
    const DataType = @TypeOf(m).DataType;
    if (@TypeOf(m) != Matrix(DataType)) {
        @compileError("m must be a Matrix");
    }

    if (!(out.rows == m.columns and out.columns == m.rows)) {
        return TensorError.ShapeMismatch;
    }

    // Get Metal context
    const ctx = metal_context.get() orelse {
        // Fall back to CPU implementation if Metal is not available
        for (0..m.rows) |i| {
            for (0..m.columns) |j| {
                const src_idx = i * m.columns + j;
                const dst_idx = j * m.rows + i;

                out.data[dst_idx] = m.data[src_idx];
            }
        }
        return;
    };

    // Only support f32 for now
    if (DataType != f32) {
        // Fall back to CPU implementation for non-f32 types
        for (0..m.rows) |i| {
            for (0..m.columns) |j| {
                const src_idx = i * m.columns + j;
                const dst_idx = j * m.rows + i;

                out.data[dst_idx] = m.data[src_idx];
            }
        }
        return;
    }

    // Create Metal buffers
    const element_size = @sizeOf(DataType);
    const matrix_size = m.rows * m.columns * element_size;
    const result_size = out.rows * out.columns * element_size;
    const uint_size = @sizeOf(u32);

    var buffer_matrix = try ctx.device.createBuffer(matrix_size, .Shared);
    defer buffer_matrix.deinit();

    var buffer_result = try ctx.device.createBuffer(result_size, .Shared);
    defer buffer_result.deinit();

    var buffer_rows = try ctx.device.createBuffer(uint_size, .Shared);
    defer buffer_rows.deinit();

    var buffer_cols = try ctx.device.createBuffer(uint_size, .Shared);
    defer buffer_cols.deinit();

    // Copy data to Metal buffers
    try buffer_matrix.copyFromSlice(std.mem.sliceAsBytes(m.data));

    // Set dimensions
    const rows_slice = buffer_rows.getContentsSlice() orelse return TensorError.BackendError;
    const cols_slice = buffer_cols.getContentsSlice() orelse return TensorError.BackendError;

    const rows: u32 = @intCast(m.rows);
    const cols: u32 = @intCast(m.columns);

    @memcpy(rows_slice, std.mem.asBytes(&rows));
    @memcpy(cols_slice, std.mem.asBytes(&cols));

    // Create command buffer and encoder
    var command_buffer = try ctx.command_queue.createCommandBuffer();
    defer command_buffer.deinit();

    var encoder = try command_buffer.createComputeCommandEncoder();
    defer encoder.deinit();

    // Set up compute command
    encoder.setComputePipelineState(ctx.matrix_transpose_pipeline);
    encoder.setBuffer(buffer_matrix, 0, 0);
    encoder.setBuffer(buffer_result, 0, 1);
    encoder.setBuffer(buffer_rows, 0, 2);
    encoder.setBuffer(buffer_cols, 0, 3);

    // Dispatch a 2D grid of threads - one thread per input element
    encoder.dispatchThreads(cols, rows, 1); // Width (columns), Height (rows)
    encoder.endEncoding();

    // Execute and wait for completion
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Copy result back to output matrix
    const result_slice = buffer_result.getContentsSlice() orelse return TensorError.BackendError;
    @memcpy(std.mem.sliceAsBytes(out.data), result_slice);
}
