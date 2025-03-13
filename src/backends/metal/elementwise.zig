const std = @import("std");
const Allocator = std.mem.Allocator;
const zit = @import("../../zit.zig");
const Tensor = zit.Tensor;
const Matrix = zit.Matrix;
const Vector = zit.Vector;
const TensorError = zit.TensorError;
const utils = @import("../utils.zig");
const fn_types = @import("../../fn_types.zig");
const metal = @import("metal");
const metal_context = @import("metal_context.zig");

fn isAddOp(comptime op_fn: fn_types.BinaryOpFn) bool {
    const addFn = struct {
        fn add(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
            return x + y;
        }
    }.add;

    return op_fn == addFn;
}

fn isSubtractOp(comptime op_fn: fn_types.BinaryOpFn) bool {
    const subtractFn = struct {
        fn subtract(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
            return x - y;
        }
    }.subtract;

    return op_fn == subtractFn;
}

fn isMultiplyOp(comptime op_fn: fn_types.BinaryOpFn) bool {
    const multiplyFn = struct {
        fn multiply(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
            return x * y;
        }
    }.multiply;

    return op_fn == multiplyFn;
}

fn isDivideOp(comptime op_fn: fn_types.BinaryOpFn) bool {
    const divideFn = struct {
        fn divide(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
            return x / y;
        }
    }.divide;

    return op_fn == divideFn;
}

pub fn op(_: *anyopaque, a: anytype, b: @TypeOf(a), out: *@TypeOf(a), op_fn: fn_types.BinaryOpFn) !void {
    try utils.ensureEqualShape(a, b);
    try utils.ensureEqualShape(a, out.*);

    // Get Metal context
    const ctx = metal_context.get() orelse {
        // Fall back to CPU implementation if Metal is not available
        for (a.data, b.data, out.data) |a_val, b_val, *result| {
            result.* = op_fn(a_val, b_val);
        }
        return;
    };

    // Only support f32 for now
    const DataType = @TypeOf(a).DataType;
    if (DataType != f32) {
        // Fall back to CPU implementation for non-f32 types
        for (a.data, b.data, out.data) |a_val, b_val, *result| {
            result.* = op_fn(a_val, b_val);
        }
        return;
    }

    // Select appropriate pipeline based on operation
    var pipeline_state: metal.ComputePipelineState = undefined;

    if (comptime isAddOp(op_fn)) {
        pipeline_state = ctx.add_pipeline;
    } else if (comptime isSubtractOp(op_fn)) {
        pipeline_state = ctx.subtract_pipeline;
    } else if (comptime isMultiplyOp(op_fn)) {
        pipeline_state = ctx.multiply_pipeline;
    } else if (comptime isDivideOp(op_fn)) {
        pipeline_state = ctx.divide_pipeline;
    } else {
        // Fall back to CPU implementation for custom operations
        for (a.data, b.data, out.data) |a_val, b_val, *result| {
            result.* = op_fn(a_val, b_val);
        }
        return;
    }

    // Create Metal buffers
    const element_size = @sizeOf(DataType);
    const data_size = a.data.len * element_size;

    var buffer_a = try ctx.device.createBuffer(data_size, .Shared);
    defer buffer_a.deinit();

    var buffer_b = try ctx.device.createBuffer(data_size, .Shared);
    defer buffer_b.deinit();

    var buffer_result = try ctx.device.createBuffer(data_size, .Shared);
    defer buffer_result.deinit();

    // Copy data to Metal buffers
    try buffer_a.copyFromSlice(std.mem.sliceAsBytes(a.data));
    try buffer_b.copyFromSlice(std.mem.sliceAsBytes(b.data));

    // Create command buffer and encoder
    var command_buffer = try ctx.command_queue.createCommandBuffer();
    defer command_buffer.deinit();

    var encoder = try command_buffer.createComputeCommandEncoder();
    defer encoder.deinit();

    // Set up compute command
    encoder.setComputePipelineState(pipeline_state);
    encoder.setBuffer(buffer_a, 0, 0);
    encoder.setBuffer(buffer_b, 0, 1);
    encoder.setBuffer(buffer_result, 0, 2);

    // Dispatch threads - one thread per element
    encoder.dispatchThreads(@intCast(a.data.len), 1, 1);
    encoder.endEncoding();

    // Execute and wait for completion
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Copy result back to output tensor
    const result_slice = buffer_result.getContentsSlice() orelse return TensorError.BackendError;
    @memcpy(std.mem.sliceAsBytes(out.data), result_slice);
}

pub fn map(_: *anyopaque, a: anytype, out: *@TypeOf(a), map_fn: fn_types.MapFn) TensorError!void {
    // Metal doesn't easily support custom functions, so fallback to CPU implementation
    try utils.ensureEqualShape(a, out.*);

    for (a.data, out.data) |a_val, *result| {
        result.* = map_fn(a_val);
    }
}

pub fn scalarMultiply(_: *anyopaque, a: anytype, scalar: anytype, out: *@TypeOf(a)) !void {
    const T = @TypeOf(a);
    const DataType = T.DataType;

    if (DataType != @TypeOf(@as(DataType, scalar))) {
        @compileError("scalar must be the same type of data as tensor");
    }

    try utils.ensureEqualShape(a, out.*);

    // Get Metal context
    const ctx = metal_context.get() orelse {
        // Fall back to CPU implementation if Metal is not available
        for (a.data, 0..) |value, i| {
            out.data[i] = value * scalar;
        }
        return;
    };

    // Only support f32 for now
    if (DataType != f32) {
        // Fall back to CPU implementation for non-f32 types
        for (a.data, 0..) |value, i| {
            out.data[i] = value * scalar;
        }
        return;
    }

    // Get buffers from pool instead of creating new ones
    const element_size = @sizeOf(DataType);
    const data_size = a.data.len * element_size;

    // OPTIMIZATION: Use buffer pool instead of creating new buffers
    var buffer_a = try ctx.buffer_pool.getBuffer(data_size, .Shared);
    defer ctx.buffer_pool.returnBuffer(buffer_a) catch {};

    var buffer_result = try ctx.buffer_pool.getBuffer(data_size, .Shared);
    defer ctx.buffer_pool.returnBuffer(buffer_result) catch {};

    // Copy data to Metal buffers
    buffer_a.copyFromSlice(std.mem.sliceAsBytes(a.data)) catch |err| {
        // If copy fails, fall back to CPU implementation
        for (a.data, 0..) |value, i| {
            out.data[i] = value * scalar;
        }
        return err;
    };

    // Create command buffer and encoder
    var command_buffer = try ctx.command_queue.createCommandBuffer();
    defer command_buffer.deinit();

    var encoder = try command_buffer.createComputeCommandEncoder();
    defer encoder.deinit();

    // Set up compute command
    encoder.setComputePipelineState(ctx.scalar_multiply_pipeline);
    encoder.setBuffer(buffer_a.*, 0, 0);
    encoder.setBuffer(buffer_result.*, 0, 1);

    // OPTIMIZATION: Pass scalar as a constant instead of a buffer
    const scalar_bytes = std.mem.asBytes(&scalar);
    encoder.setBytes(scalar_bytes.ptr, scalar_bytes.len, 2);

    // Dispatch threads - one thread per element
    encoder.dispatchThreads(@intCast(a.data.len), 1, 1);
    encoder.endEncoding();

    // Execute and wait for completion
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Copy result back to output tensor
    const result_bytes = buffer_result.getContentsSlice() orelse return TensorError.BackendError;
    @memcpy(std.mem.sliceAsBytes(out.data), result_bytes[0..data_size]);
}
