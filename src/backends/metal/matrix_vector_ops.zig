const std = @import("std");
const zit = @import("../../zit.zig");
const Matrix = zit.Matrix;
const Vector = zit.Vector;
const TensorError = zit.TensorError;
const metal = @import("metal");
const metal_context = @import("metal_context.zig");

pub fn matrixVectorMultiply(_: *anyopaque, m: anytype, v: anytype, out: *@TypeOf(v)) !void {
    const MType = @TypeOf(m);
    const VType = @TypeOf(v);
    if (MType != Matrix(MType.DataType)) {
        @compileError("m must be a matrix");
    }
    if (VType != Vector(VType.DataType)) {
        @compileError("v must be a vector");
    }
    if (MType.DataType != VType.DataType) {
        @compileError("m and v must have the same underlying data types");
    }

    // Ensure matrix columns match vector length
    if (m.columns != v.data.len) {
        return TensorError.ShapeMismatch;
    }
    if (m.rows != out.data.len) {
        return TensorError.ShapeMismatch;
    }

    const DataType = MType.DataType;

    // Get Metal context
    const ctx = metal_context.get() orelse {
        // Fall back to CPU implementation if Metal is not available
        @memset(out.data, 0);
        for (0..m.rows) |i| {
            for (0..m.columns) |j| {
                const m_idx = i * m.columns + j;
                out.data[i] += m.data[m_idx] * v.data[j];
            }
        }
        return;
    };

    // Only support f32 for now
    if (DataType != f32) {
        // Fall back to CPU implementation for non-f32 types
        @memset(out.data, 0);
        for (0..m.rows) |i| {
            for (0..m.columns) |j| {
                const m_idx = i * m.columns + j;
                out.data[i] += m.data[m_idx] * v.data[j];
            }
        }
        return;
    }

    // Create Metal buffers
    const element_size = @sizeOf(DataType);
    const matrix_size = m.rows * m.columns * element_size;
    const vector_size = v.data.len * element_size;
    const result_size = out.data.len * element_size;
    const uint_size = @sizeOf(u32);

    var buffer_matrix = try ctx.device.createBuffer(matrix_size, .Shared);
    defer buffer_matrix.deinit();

    var buffer_vector = try ctx.device.createBuffer(vector_size, .Shared);
    defer buffer_vector.deinit();

    var buffer_result = try ctx.device.createBuffer(result_size, .Shared);
    defer buffer_result.deinit();

    var buffer_rows = try ctx.device.createBuffer(uint_size, .Shared);
    defer buffer_rows.deinit();

    var buffer_cols = try ctx.device.createBuffer(uint_size, .Shared);
    defer buffer_cols.deinit();

    // Copy data to Metal buffers
    try buffer_matrix.copyFromSlice(std.mem.sliceAsBytes(m.data));
    try buffer_vector.copyFromSlice(std.mem.sliceAsBytes(v.data));

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
    encoder.setComputePipelineState(ctx.matrix_vector_multiply_pipeline);
    encoder.setBuffer(buffer_matrix, 0, 0);
    encoder.setBuffer(buffer_vector, 0, 1);
    encoder.setBuffer(buffer_result, 0, 2);
    encoder.setBuffer(buffer_rows, 0, 3);
    encoder.setBuffer(buffer_cols, 0, 4);

    // Dispatch threads - one thread per output element (one per row)
    encoder.dispatchThreads(rows, 1, 1);
    encoder.endEncoding();

    // Execute and wait for completion
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Copy result back to output vector
    const result_slice = buffer_result.getContentsSlice() orelse return TensorError.BackendError;
    @memcpy(std.mem.sliceAsBytes(out.data), result_slice);
}
