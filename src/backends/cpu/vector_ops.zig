const std = @import("std");
const zit = @import("../../zit.zig");
const Vector = zit.Vector;
const TensorError = zit.TensorError;

pub fn vectorDot(_: *anyopaque, a: anytype, b: @TypeOf(a), out: *@TypeOf(a).DataType) TensorError!void {
    const DataType = @TypeOf(a).DataType;
    if (@TypeOf(a) != Vector(DataType)) {
        @compileError("a and b must be Vectors");
    }

    // Ensure vectors have the same length
    if (a.data.len != b.data.len) {
        return TensorError.LengthMismatch;
    }

    var result: DataType = 0;

    for (a.data, b.data) |a_val, b_val| {
        result += a_val * b_val;
    }

    out.* = result;
}

pub fn vectorNorm(_: *anyopaque, v: anytype, out: *@TypeOf(v).DataType) TensorError!void {
    const DataType = @TypeOf(v).DataType;
    if (@TypeOf(v) != Vector(DataType)) {
        @compileError("v must be a Vector");
    }

    var sum_sq: DataType = 0;

    for (v.data) |val| {
        sum_sq += val * val;
    }

    out.* = @sqrt(sum_sq);
}

const testing = std.testing;

fn emptyCtx() *anyopaque {
    return @ptrFromInt(1);
}

test vectorDot {
    // Initialize vectors with specific values
    const v1 = try Vector(f32).init(3, testing.allocator);
    defer v1.deinit();
    v1.data[0] = 1.0;
    v1.data[1] = 2.0;
    v1.data[2] = 3.0;

    const v2 = try Vector(f32).init(3, testing.allocator);
    defer v2.deinit();
    v2.data[0] = 4.0;
    v2.data[1] = 5.0;
    v2.data[2] = 6.0;

    var result: f32 = undefined;
    try vectorDot(emptyCtx(), v1, v2, &result);
    // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    try testing.expectEqual(32.0, result);

    // Test error case: vectors of different lengths
    const v3 = try Vector(f32).init(4, testing.allocator);
    defer v3.deinit();

    try testing.expectError(TensorError.LengthMismatch, vectorDot(emptyCtx(), v1, v3, &result));
}

test vectorNorm {
    // Initialize vector with specific values
    const v = try Vector(f32).init(3, testing.allocator);
    defer v.deinit();
    v.data[0] = 3.0;
    v.data[1] = 4.0;
    v.data[2] = 0.0;

    var result: f32 = undefined;
    try vectorNorm(emptyCtx(), v, &result);
    // Expected: sqrt(3^2 + 4^2 + 0^2) = sqrt(9 + 16) = sqrt(25) = 5
    try testing.expectEqual(5.0, result);

    // Test with zero vector
    const zero_v = try Vector(f32).splat(3, 0.0, testing.allocator);
    defer zero_v.deinit();

    try vectorNorm(emptyCtx(), zero_v, &result);
    try testing.expectEqual(0.0, result);

    // Test with negative values
    const neg_v = try Vector(f32).init(2, testing.allocator);
    defer neg_v.deinit();
    neg_v.data[0] = -3.0;
    neg_v.data[1] = 4.0;

    try vectorNorm(emptyCtx(), neg_v, &result);
    // Expected: sqrt((-3)^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    try testing.expectEqual(5.0, result);
}

test "edge cases" {
    // Empty vector (length 0)
    const empty_v = try Vector(f32).init(0, testing.allocator);
    defer empty_v.deinit();

    // Vector norm of empty vector should be 0
    var result: f32 = undefined;
    try vectorNorm(emptyCtx(), empty_v, &result);
    try testing.expectEqual(0.0, result);
}

test "boundary values" {
    // Create vectors with extreme values
    const large_v = try Vector(f32).init(3, testing.allocator);
    defer large_v.deinit();
    large_v.data[0] = std.math.floatMax(f32);
    large_v.data[1] = std.math.floatMax(f32) / 2.0;
    large_v.data[2] = std.math.floatMax(f32) / 4.0;

    const small_v = try Vector(f32).init(3, testing.allocator);
    defer small_v.deinit();
    small_v.data[0] = std.math.floatMin(f32);
    small_v.data[1] = std.math.floatMin(f32) * 2.0;
    small_v.data[2] = std.math.floatMin(f32) * 4.0;

    // Test vector norm with large values
    var result: f32 = undefined;
    try vectorNorm(emptyCtx(), large_v, &result);
    try testing.expect(result > 0);
    try testing.expect(!std.math.isNan(result));

    // Test vector norm with small values
    try vectorNorm(emptyCtx(), small_v, &result);
    try testing.expect(result >= 0);
    try testing.expect(!std.math.isNan(result));
}
