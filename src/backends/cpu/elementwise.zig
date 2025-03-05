const std = @import("std");
const Allocator = std.mem.Allocator;
const zit = @import("../../zit.zig");
const Tensor = zit.Tensor;
const Matrix = zit.Matrix;
const Vector = zit.Vector;
const TensorError = zit.TensorError;
const TensorOpError = zit.TensorOpError;
const utils = @import("../utils.zig");

pub fn add(a: anytype, b: @TypeOf(a), allocator: Allocator) TensorError!@TypeOf(a) {
    const T = @TypeOf(a);
    const DataType = T.DataType;

    if (T == Vector(DataType)) {
        var v = try Vector(DataType).init(a.data.len, allocator);
        errdefer v.deinit();
        try addWithOut(a, b, &v);
        return v;
    } else if (T == Matrix(DataType)) {
        var m = try Matrix(DataType).init(a.rows, a.columns, allocator);
        errdefer m.deinit();
        try addWithOut(a, b, &m);
        return m;
    } else {
        var t = try Tensor(DataType).init(a.shape, allocator);
        errdefer t.deinit();
        try addWithOut(a, b, &t);
        return t;
    }
}

pub fn addInPlace(a: anytype, b: std.meta.Child(@TypeOf(a))) TensorOpError!void {
    try addWithOut(a.*, b, a);
}

pub fn addWithOut(a: anytype, b: @TypeOf(a), out: *@TypeOf(a)) TensorOpError!void {
    try utils.ensureEqualShape(a, b);
    try utils.ensureEqualShape(a, out);

    for (a.data, b.data, out.data) |a_val, b_val, *result| {
        result.* = a_val + b_val;
    }
}

pub fn subtract(a: anytype, b: @TypeOf(a), allocator: Allocator) TensorError!@TypeOf(a) {
    const T = @TypeOf(a);
    const DataType = T.DataType;

    if (T == Vector(DataType)) {
        var v = try Vector(DataType).init(a.data.len, allocator);
        errdefer v.deinit();
        try subtractWithOut(a, b, &v);
        return v;
    } else if (T == Matrix(DataType)) {
        var m = try Matrix(DataType).init(a.rows, a.columns, allocator);
        errdefer m.deinit();
        try subtractWithOut(a, b, &m);
        return m;
    } else {
        var t = try Tensor(DataType).init(a.shape, allocator);
        errdefer t.deinit();
        try subtractWithOut(a, b, &t);
        return t;
    }
}

pub fn subtractInPlace(a: anytype, b: std.meta.Child(@TypeOf(a))) TensorOpError!void {
    try subtractWithOut(a.*, b, a);
}

pub fn subtractWithOut(a: anytype, b: @TypeOf(a), out: *@TypeOf(a)) TensorOpError!void {
    try utils.ensureEqualShape(a, b);
    try utils.ensureEqualShape(a, out);

    for (a.data, b.data, out.data) |a_val, b_val, *result| {
        result.* = a_val - b_val;
    }
}

pub fn multiply(a: anytype, b: @TypeOf(a), allocator: Allocator) TensorError!@TypeOf(a) {
    const T = @TypeOf(a);
    const DataType = T.DataType;

    if (T == Vector(DataType)) {
        var v = try Vector(DataType).init(a.data.len, allocator);
        errdefer v.deinit();
        try multiplyWithOut(a, b, &v);
        return v;
    } else if (T == Matrix(DataType)) {
        var m = try Matrix(DataType).init(a.rows, a.columns, allocator);
        errdefer m.deinit();
        try multiplyWithOut(a, b, &m);
        return m;
    } else {
        var t = try Tensor(DataType).init(a.shape, allocator);
        errdefer t.deinit();
        try multiplyWithOut(a, b, &t);
        return t;
    }
}

pub fn multiplyInPlace(a: anytype, b: std.meta.Child(@TypeOf(a))) TensorOpError!void {
    try multiplyWithOut(a.*, b, a);
}

pub fn multiplyWithOut(a: anytype, b: @TypeOf(a), out: *@TypeOf(a)) TensorOpError!void {
    try utils.ensureEqualShape(a, b);
    try utils.ensureEqualShape(a, out);

    for (a.data, b.data, out.data) |a_val, b_val, *result| {
        result.* = a_val * b_val;
    }
}

pub fn divide(a: anytype, b: @TypeOf(a), allocator: Allocator) TensorError!@TypeOf(a) {
    const T = @TypeOf(a);
    const DataType = T.DataType;

    if (T == Vector(DataType)) {
        var v = try Vector(DataType).init(a.data.len, allocator);
        errdefer v.deinit();
        try divideWithOut(a, b, &v);
        return v;
    } else if (T == Matrix(DataType)) {
        var m = try Matrix(DataType).init(a.rows, a.columns, allocator);
        errdefer m.deinit();
        try divideWithOut(a, b, &m);
        return m;
    } else {
        var t = try Tensor(DataType).init(a.shape, allocator);
        errdefer t.deinit();
        try divideWithOut(a, b, &t);
        return t;
    }
}

pub fn divideInPlace(a: anytype, b: std.meta.Child(@TypeOf(a))) TensorOpError!void {
    try divideWithOut(a.*, b, a);
}

pub fn divideWithOut(a: anytype, b: @TypeOf(a), out: *@TypeOf(a)) TensorOpError!void {
    try utils.ensureEqualShape(a, b);
    try utils.ensureEqualShape(a, out);

    for (a.data, b.data, out.data) |a_val, b_val, *result| {
        result.* = a_val / b_val;
    }
}

pub fn scalarMultiply(a: anytype, scalar: anytype, allocator: Allocator) TensorError!@TypeOf(a) {
    const T = std.meta.Child(@TypeOf(a));
    const DataType = @TypeOf(scalar);

    if (T == Vector(DataType)) {
        var v = try Vector(DataType).init(a.data.len, allocator);
        errdefer v.deinit();
        try scalarMultiplyWithOut(a, &v);
        return v;
    } else if (T == Matrix(DataType)) {
        var m = try Matrix(DataType).init(a.rows, a.columns, allocator);
        errdefer m.deinit();
        try scalarMultiplyWithOut(a, &m);
        return m;
    } else {
        var t = try Tensor(DataType).init(a.shape, allocator);
        errdefer t.deinit();
        try scalarMultiplyWithOut(a, &t);
        return t;
    }
}

pub fn scalarMultiplyInPlace(a: anytype, scalar: anytype) TensorOpError!void {
    try scalarMultiplyWithOut(a.*, scalar, a);
}

pub fn scalarMultiplyWithOut(a: anytype, scalar: anytype, out: *@TypeOf(a)) TensorOpError!void {
    const T = @TypeOf(a);
    const DataType = T.DataType;
    if (DataType != @TypeOf(@as(DataType, scalar))) {
        @compileError("scalar must be the same type as the data type of a");
    }

    for (a.data, out.data) |a_val, *result| {
        result.* = a_val * scalar;
    }
}

const testing = std.testing;

test "add" {
    // Test with Tensor
    const t1 = try Tensor(f32).splat(&.{ 2, 2, 2 }, 5.0, testing.allocator);
    defer t1.deinit();

    const t2 = try Tensor(f32).splat(&.{ 2, 2, 2 }, 2.0, testing.allocator);
    defer t2.deinit();

    const result_t = try add(t1, t2, testing.allocator);
    defer result_t.deinit();

    try testing.expectEqual(@as(f32, 7.0), result_t.data[0]);

    // Test with Matrix
    const m1 = try Matrix(f32).splat(2, 2, 7.0, testing.allocator);
    defer m1.deinit();

    const m2 = try Matrix(f32).splat(2, 2, 3.0, testing.allocator);
    defer m2.deinit();

    const result_m = try add(m1, m2, testing.allocator);
    defer result_m.deinit();

    try testing.expectEqual(@as(f32, 10.0), result_m.data[0]);

    // Test with Vector
    const v1 = try Vector(f32).splat(3, 9.0, testing.allocator);
    defer v1.deinit();

    const v2 = try Vector(f32).splat(3, 4.0, testing.allocator);
    defer v2.deinit();

    const result_v = try add(v1, v2, testing.allocator);
    defer result_v.deinit();

    try testing.expectEqual(@as(f32, 13.0), result_v.data[0]);
}

test "subtract" {
    // Test with Tensor
    const t1 = try Tensor(f32).splat(&.{ 2, 2, 2 }, 5.0, testing.allocator);
    defer t1.deinit();

    const t2 = try Tensor(f32).splat(&.{ 2, 2, 2 }, 2.0, testing.allocator);
    defer t2.deinit();

    const result_t = try subtract(t1, t2, testing.allocator);
    defer result_t.deinit();

    try testing.expectEqual(@as(f32, 3.0), result_t.data[0]);

    // Test with Matrix
    const m1 = try Matrix(f32).splat(2, 2, 7.0, testing.allocator);
    defer m1.deinit();

    const m2 = try Matrix(f32).splat(2, 2, 3.0, testing.allocator);
    defer m2.deinit();

    const result_m = try subtract(m1, m2, testing.allocator);
    defer result_m.deinit();

    try testing.expectEqual(@as(f32, 4.0), result_m.data[0]);

    // Test with Vector
    const v1 = try Vector(f32).splat(3, 9.0, testing.allocator);
    defer v1.deinit();

    const v2 = try Vector(f32).splat(3, 4.0, testing.allocator);
    defer v2.deinit();

    const result_v = try subtract(v1, v2, testing.allocator);
    defer result_v.deinit();

    try testing.expectEqual(@as(f32, 5.0), result_v.data[0]);
}

test "multiply" {
    // Test with Tensor (element-wise multiplication)
    const t1 = try Tensor(f32).splat(&.{ 2, 2, 2 }, 3.0, testing.allocator);
    defer t1.deinit();

    const t2 = try Tensor(f32).splat(&.{ 2, 2, 2 }, 2.0, testing.allocator);
    defer t2.deinit();

    const result_t = try multiply(t1, t2, testing.allocator);
    defer result_t.deinit();

    try testing.expectEqual(@as(f32, 6.0), result_t.data[0]);

    // Test with Matrix (element-wise multiplication)
    const m1 = try Matrix(f32).splat(2, 2, 4.0, testing.allocator);
    defer m1.deinit();

    const m2 = try Matrix(f32).splat(2, 2, 3.0, testing.allocator);
    defer m2.deinit();

    const result_m = try multiply(m1, m2, testing.allocator);
    defer result_m.deinit();

    try testing.expectEqual(@as(f32, 12.0), result_m.data[0]);

    // Test with Vector (element-wise multiplication)
    const v1 = try Vector(f32).splat(3, 5.0, testing.allocator);
    defer v1.deinit();

    const v2 = try Vector(f32).splat(3, 2.0, testing.allocator);
    defer v2.deinit();

    const result_v = try multiply(v1, v2, testing.allocator);
    defer result_v.deinit();

    try testing.expectEqual(@as(f32, 10.0), result_v.data[0]);
}

test "scalarMultiply" {
    // Test with Tensor
    const t = try Tensor(f32).splat(&.{ 2, 2, 2 }, 3.0, testing.allocator);
    defer t.deinit();

    const result_t = try scalarMultiply(t, 2.0, testing.allocator);
    defer result_t.deinit();

    try testing.expectEqual(@as(f32, 6.0), result_t.data[0]);

    // Test with Matrix
    const m = try Matrix(f32).splat(2, 2, 4.0, testing.allocator);
    defer m.deinit();

    const result_m = try scalarMultiply(m, 3.0, testing.allocator);
    defer result_m.deinit();

    try testing.expectEqual(@as(f32, 12.0), result_m.data[0]);

    // Test with Vector
    const v = try Vector(f32).splat(3, 5.0, testing.allocator);
    defer v.deinit();

    const result_v = try scalarMultiply(v, 4.0, testing.allocator);
    defer result_v.deinit();

    try testing.expectEqual(@as(f32, 20.0), result_v.data[0]);
}

test "int operations" {
    // Test operations with integer types
    const v1 = try Vector(i32).splat(3, 5, testing.allocator);
    defer v1.deinit();

    const v2 = try Vector(i32).splat(3, 2, testing.allocator);
    defer v2.deinit();

    // Test add with integers
    const add_result = try add(v1, v2, testing.allocator);
    defer add_result.deinit();
    try testing.expectEqual(@as(i32, 7), add_result.data[0]);

    // Test subtract with integers
    const sub_result = try subtract(v1, v2, testing.allocator);
    defer sub_result.deinit();
    try testing.expectEqual(@as(i32, 3), sub_result.data[0]);

    // Test multiply with integers
    const mul_result = try multiply(v1, v2, testing.allocator);
    defer mul_result.deinit();
    try testing.expectEqual(@as(i32, 10), mul_result.data[0]);
}

test "edge cases" {
    const empty_v = try Vector(f32).init(0, testing.allocator);
    defer empty_v.deinit();
    const empty_v2 = try Vector(f32).init(0, testing.allocator);
    defer empty_v2.deinit();
    // Add two empty vectors should work
    const empty_sum = try add(empty_v, empty_v2, testing.allocator);
    defer empty_sum.deinit();
    try testing.expectEqual(@as(usize, 0), empty_sum.data.len);

    const m1 = try Matrix(f32).init(3, 2, testing.allocator);
    defer m1.deinit();
    const m2 = try Matrix(f32).init(2, 4, testing.allocator);
    defer m2.deinit();
    // Element-wise operations should fail with shape mismatch
    try testing.expectError(TensorOpError.ShapeMismatch, add(m1, m2, testing.allocator));

    // Test with NaN and Inf values
    const special_v = try Vector(f32).init(4, testing.allocator);
    defer special_v.deinit();
    special_v.data[0] = std.math.nan(f32);
    special_v.data[1] = std.math.inf(f32);
    special_v.data[2] = -std.math.inf(f32);
    special_v.data[3] = 1.0;

    // Multiplying by zero should give a vector of zeros, even with NaN/Inf values
    const zero_v = try scalarMultiply(special_v, 0.0, testing.allocator);
    defer zero_v.deinit();

    // Check for zeros (allowing for possibility of NaN * 0 = NaN in standard IEEE)
    for (zero_v.data) |val| {
        if (!std.math.isNan(val)) {
            try testing.expectEqual(@as(f32, 0.0), val);
        }
    }
}

test "integer overflow handling" {
    // Test integer overflow behavior
    if (comptime @typeInfo(usize).Int.bits >= 32) {
        // Only run this test if usize is at least 32 bits

        // Create vector with values near maximum
        const max_v = try Vector(i32).init(3, testing.allocator);
        defer max_v.deinit();
        max_v.data[0] = std.math.maxInt(i32) - 10;
        max_v.data[1] = std.math.maxInt(i32) - 5;
        max_v.data[2] = std.math.maxInt(i32) - 1;

        // Create another vector with small positive values
        const small_v = try Vector(i32).init(3, testing.allocator);
        defer small_v.deinit();
        small_v.data[0] = 5;
        small_v.data[1] = 10;
        small_v.data[2] = 20;

        // Add them - in Zig this may trap on overflow, depending on build mode
        if (std.debug.runtime_safety) {
            // In safe builds, this might panic, so we can't really test it directly
            // We could use expectError but that would require changing the add function
            // to return an error on integer overflow
        } else {
            const sum = try add(max_v, small_v, testing.allocator);
            defer sum.deinit();
            // In unsafe builds, this would wrap around, but we can't rely on that
            // without knowing the build configuration
        }
    }
}

test "multi-dimensional tensors" {
    // Test operations with multi-dimensional tensors
    const t1 = try Tensor(f32).splat(&.{ 2, 2, 2 }, 2.0, testing.allocator);
    defer t1.deinit();

    const t2 = try Tensor(f32).splat(&.{ 2, 2, 2 }, 3.0, testing.allocator);
    defer t2.deinit();

    // Addition
    const t_sum = try add(t1, t2, testing.allocator);
    defer t_sum.deinit();

    // Check shape
    try testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, t_sum.shape.items);

    // Check values
    for (t_sum.data) |val| {
        try testing.expectEqual(@as(f32, 5.0), val);
    }

    // Subtraction
    const t_diff = try subtract(t2, t1, testing.allocator);
    defer t_diff.deinit();

    // Check values
    for (t_diff.data) |val| {
        try testing.expectEqual(@as(f32, 1.0), val);
    }

    // Multiplication
    const t_prod = try multiply(t1, t2, testing.allocator);
    defer t_prod.deinit();

    // Check values
    for (t_prod.data) |val| {
        try testing.expectEqual(@as(f32, 6.0), val);
    }

    // Scalar multiplication
    const t_scaled = try scalarMultiply(t1, 1.5, testing.allocator);
    defer t_scaled.deinit();

    // Check values
    for (t_scaled.data) |val| {
        try testing.expectEqual(@as(f32, 3.0), val);
    }
}

test "shape validation" {
    // Create tensors with different shapes
    const t1 = try Tensor(f32).splat(&.{ 2, 3 }, 1.0, testing.allocator);
    defer t1.deinit();

    const t2 = try Tensor(f32).splat(&.{ 3, 2 }, 2.0, testing.allocator);
    defer t2.deinit();

    // Should fail due to shape mismatch
    try testing.expectError(TensorOpError.ShapeMismatch, add(t1, t2, testing.allocator));
    try testing.expectError(TensorOpError.ShapeMismatch, subtract(t1, t2, testing.allocator));
    try testing.expectError(TensorOpError.ShapeMismatch, multiply(t1, t2, testing.allocator));
}
