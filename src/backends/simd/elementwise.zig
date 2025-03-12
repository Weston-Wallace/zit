const std = @import("std");
const Allocator = std.mem.Allocator;
const zit = @import("../../zit.zig");
const Tensor = zit.Tensor;
const Matrix = zit.Matrix;
const Vector = zit.Vector;
const TensorError = zit.TensorError;
const TensorOpError = zit.TensorOpError;
const utils = @import("../utils.zig");
const fn_types = @import("../../fn_types.zig");
const chunk_size = @import("SimdBackend.zig").chunk_size;

pub fn op(_: *anyopaque, a: anytype, b: @TypeOf(a), out: *@TypeOf(a), op_fn: fn_types.BinaryOpFn) TensorOpError!void {
    try utils.ensureEqualShape(a, b);
    try utils.ensureEqualShape(a, out.*);

    const DataType = @TypeOf(a).DataType;

    if (a.data.len >= chunk_size) {
        const chunk_count = a.data.len / chunk_size;

        for (0..chunk_count) |chunk| {
            const offset = chunk * chunk_size;
            const a_chunk: @Vector(chunk_size, DataType) = a.data[offset..][0..chunk_size].*;
            const b_chunk: @Vector(chunk_size, DataType) = b.data[offset..][0..chunk_size].*;

            out.data[offset..][0..chunk_size].* = op_fn(a_chunk, b_chunk);
        }

        var i: usize = chunk_count * chunk_size;
        while (i < a.data.len) : (i += 1) {
            out.data[i] = op_fn(a.data[i], b.data[i]);
        }
    }
    for (a.data, b.data, out.data) |a_val, b_val, *result| {
        result.* = op_fn(a_val, b_val);
    }
}

// Doesn't make sense for SIMD
pub fn map(_: *anyopaque, a: anytype, out: *@TypeOf(a), comptime map_fn: fn_types.MapFn) TensorOpError!void {
    try utils.ensureEqualShape(a, out.*);

    for (a.data, out.data) |a_val, *result| {
        result.* = map_fn(a_val);
    }
}

pub fn scalarMultiply(_: *anyopaque, a: anytype, scalar: anytype, out: *@TypeOf(a)) TensorOpError!void {
    const T = @TypeOf(a);
    const DataType = T.DataType;
    if (DataType != @TypeOf(@as(DataType, scalar))) {
        @compileError("scalar must be the same type as the data type of a");
    }

    // For tensors with sufficient data, use SIMD
    if (a.data.len >= chunk_size) {
        const chunk_count = a.data.len / chunk_size;
        const scalar_vec: @Vector(chunk_size, DataType) = @splat(scalar);

        // Process in chunks
        for (0..chunk_count) |chunk| {
            const offset = chunk * chunk_size;
            const a_chunk: @Vector(chunk_size, DataType) = a.data[offset..][0..chunk_size].*;

            // Multiply by scalar
            const result_chunk = a_chunk * scalar_vec;

            // Store the result
            out.data[offset..][0..chunk_size].* = result_chunk;
        }

        // Handle remaining elements
        var i: usize = chunk_count * chunk_size;
        while (i < a.data.len) : (i += 1) {
            out.data[i] = a.data[i] * scalar;
        }
    } else {
        // Fall back to scalar implementation
        for (a.data, out.data) |a_val, *result| {
            result.* = a_val * scalar;
        }
    }
}

const testing = std.testing;

fn add(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    return x + y;
}

fn emptyCtx() *anyopaque {
    return @ptrFromInt(1);
}

test op {
    const t1 = try Tensor(f32).splat(&.{ 2, 2, 2 }, 3, testing.allocator);
    defer t1.deinit();
    const t2 = try Tensor(f32).splat(&.{ 2, 2, 2 }, 5, testing.allocator);
    defer t2.deinit();
    var result = try Tensor(f32).init(&.{ 2, 2, 2 }, testing.allocator);
    defer result.deinit();

    try op(emptyCtx(), t1, t2, &result, add);
    try testing.expectEqual(8, result.data[0]);
}

fn add5(x: anytype) @TypeOf(x) {
    return x + 5;
}

test map {
    const t1 = try Tensor(f32).splat(&.{ 2, 2, 2 }, 3, testing.allocator);
    defer t1.deinit();
    var result = try Tensor(f32).init(&.{ 2, 2, 2 }, testing.allocator);
    defer result.deinit();

    try map(emptyCtx(), t1, &result, add5);
    try testing.expectEqual(8, result.data[0]);
}

test scalarMultiply {
    // Test with Tensor
    const t = try Tensor(f32).splat(&.{ 2, 2, 2 }, 3.0, testing.allocator);
    defer t.deinit();
    var result_t = try Tensor(f32).init(&.{ 2, 2, 2 }, testing.allocator);
    defer result_t.deinit();

    try scalarMultiply(emptyCtx(), t, 2.0, &result_t);

    try testing.expectEqual(@as(f32, 6.0), result_t.data[0]);

    // Test with Matrix
    const m = try Matrix(f32).splat(2, 2, 4.0, testing.allocator);
    defer m.deinit();
    var result_m = try Matrix(f32).init(2, 2, testing.allocator);
    defer result_m.deinit();

    try scalarMultiply(emptyCtx(), m, 3.0, &result_m);

    try testing.expectEqual(@as(f32, 12.0), result_m.data[0]);

    // Test with Vector
    const v = try Vector(f32).splat(3, 5.0, testing.allocator);
    defer v.deinit();
    var result_v = try Vector(f32).init(3, testing.allocator);
    defer result_v.deinit();

    try scalarMultiply(emptyCtx(), v, 4.0, &result_v);

    try testing.expectEqual(@as(f32, 20.0), result_v.data[0]);
}
