const std = @import("std");
const zit = @import("../../zit.zig");
const Vector = zit.Vector;
const TensorError = zit.TensorError;
const metal = @import("metal");
const metal_context = @import("metal_context.zig");

pub fn vectorDot(_: *anyopaque, a: anytype, b: @TypeOf(a), out: *@TypeOf(a).DataType) !void {
    const DataType = @TypeOf(a).DataType;
    if (@TypeOf(a) != Vector(DataType)) {
        @compileError("a and b must be Vectors");
    }

    // Ensure vectors have the same length
    if (a.data.len != b.data.len) {
        return TensorError.LengthMismatch;
    }

    // Handle empty vectors
    if (a.data.len == 0) {
        out.* = 0;
        return;
    }

    // Get Metal context
    const ctx = metal_context.get() orelse {
        // Fall back to CPU implementation if Metal is not available
        var result: DataType = 0;
        for (a.data, b.data) |a_val, b_val| {
            result += a_val * b_val;
        }
        out.* = result;
        return;
    };

    // Only support f32 for now
    if (DataType != f32) {
        // Fall back to CPU implementation for non-f32 types
        var result: DataType = 0;
        for (a.data, b.data) |a_val, b_val| {
            result += a_val * b_val;
        }
        out.* = result;
        return;
    }

    // Create Metal buffers
    const element_size = @sizeOf(DataType);
    const vector_size = a.data.len * element_size;
    const uint_size = @sizeOf(u32);

    var buffer_a = try ctx.device.createBuffer(vector_size, .Shared);
    defer buffer_a.deinit();

    var buffer_b = try ctx.device.createBuffer(vector_size, .Shared);
    defer buffer_b.deinit();

    var buffer_result = try ctx.device.createBuffer(element_size, .Shared);
    defer buffer_result.deinit();

    var buffer_length = try ctx.device.createBuffer(uint_size, .Shared);
    defer buffer_length.deinit();

    // Copy data to Metal buffers
    try buffer_a.copyFromSlice(std.mem.sliceAsBytes(a.data));
    try buffer_b.copyFromSlice(std.mem.sliceAsBytes(b.data));

    // Set vector length
    const length_slice = buffer_length.getContentsSlice() orelse return TensorError.BackendError;
    const length: u32 = @intCast(a.data.len);
    @memcpy(length_slice, std.mem.asBytes(&length));

    // Create command buffer and encoder
    var command_buffer = try ctx.command_queue.createCommandBuffer();
    defer command_buffer.deinit();

    var encoder = try command_buffer.createComputeCommandEncoder();
    defer encoder.deinit();

    // Set up compute command
    encoder.setComputePipelineState(ctx.vector_dot_pipeline);
    encoder.setBuffer(buffer_a, 0, 0);
    encoder.setBuffer(buffer_b, 0, 1);
    encoder.setBuffer(buffer_result, 0, 2);
    encoder.setBuffer(buffer_length, 0, 3);

    // Use 32 threads per threadgroup for the reduction
    const threadgroup_size: u32 = 32;
    const grid_size: u32 = @min(threadgroup_size, length);

    encoder.dispatchThreads(grid_size, 1, 1);
    encoder.endEncoding();

    // Execute and wait for completion
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Copy result back
    const result_slice = buffer_result.getContentsSlice() orelse return TensorError.BackendError;
    @memcpy(std.mem.asBytes(out), result_slice[0..element_size]);
}

pub fn vectorNorm(_: *anyopaque, v: anytype, out: *@TypeOf(v).DataType) !void {
    const DataType = @TypeOf(v).DataType;
    if (@TypeOf(v) != Vector(DataType)) {
        @compileError("v must be a Vector");
    }

    // Handle empty vector
    if (v.data.len == 0) {
        out.* = 0;
        return;
    }

    // Get Metal context
    const ctx = metal_context.get() orelse {
        // Fall back to CPU implementation if Metal is not available
        var sum_sq: DataType = 0;
        for (v.data) |val| {
            sum_sq += val * val;
        }
        out.* = @sqrt(sum_sq);
        return;
    };

    // Only support f32 for now
    if (DataType != f32) {
        // Fall back to CPU implementation for non-f32 types
        var sum_sq: DataType = 0;
        for (v.data) |val| {
            sum_sq += val * val;
        }
        out.* = @sqrt(sum_sq);
        return;
    }

    // Create Metal buffers
    const element_size = @sizeOf(DataType);
    const vector_size = v.data.len * element_size;
    const uint_size = @sizeOf(u32);

    var buffer_v = try ctx.device.createBuffer(vector_size, .Shared);
    defer buffer_v.deinit();

    var buffer_result = try ctx.device.createBuffer(element_size, .Shared);
    defer buffer_result.deinit();

    var buffer_length = try ctx.device.createBuffer(uint_size, .Shared);
    defer buffer_length.deinit();

    // Copy data to Metal buffers
    try buffer_v.copyFromSlice(std.mem.sliceAsBytes(v.data));

    // Set vector length
    const length_slice = buffer_length.getContentsSlice() orelse return TensorError.BackendError;
    const length: u32 = @intCast(v.data.len);
    @memcpy(length_slice, std.mem.asBytes(&length));

    // Create command buffer and encoder
    var command_buffer = try ctx.command_queue.createCommandBuffer();
    defer command_buffer.deinit();

    var encoder = try command_buffer.createComputeCommandEncoder();
    defer encoder.deinit();

    // Set up compute command
    encoder.setComputePipelineState(ctx.vector_norm_pipeline);
    encoder.setBuffer(buffer_v, 0, 0);
    encoder.setBuffer(buffer_result, 0, 1);
    encoder.setBuffer(buffer_length, 0, 2);

    // Use 32 threads per threadgroup for the reduction
    const threadgroup_size: u32 = 32;
    const grid_size: u32 = @min(threadgroup_size, length);

    encoder.dispatchThreads(grid_size, 1, 1);
    encoder.endEncoding();

    // Execute and wait for completion
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Copy result back
    const result_slice = buffer_result.getContentsSlice() orelse return TensorError.BackendError;
    @memcpy(std.mem.asBytes(out), result_slice[0..element_size]);
}
