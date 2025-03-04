const std = @import("std");
const Allocator = std.mem.Allocator;
const zit = @import("../../zit.zig");
const Tensor = zit.Tensor;
const Matrix = zit.Matrix;
const Vector = zit.Vector;
const TensorError = zit.TensorError;
const TensorOpError = zit.TensorOpError;
const utils = @import("../utils.zig");

pub fn opWithOut(a: anytype, b: @TypeOf(a), out: *@TypeOf(a), op_fn: utils.BinaryOpFn) TensorOpError!void {
    try utils.ensureEqualShape(a, b);
    try utils.ensureEqualShape(a, out.*);

    for (a.data, b.data, out.data) |a_val, b_val, *result| {
        result.* = op_fn(a_val, b_val);
    }
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

fn add(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    return x + y;
}

test opWithOut {
    const t1 = try Tensor(f32).splat(&.{ 2, 2, 2 }, 3, testing.allocator);
    defer t1.deinit();
    const t2 = try Tensor(f32).splat(&.{ 2, 2, 2 }, 5, testing.allocator);
    defer t2.deinit();
    var result = try Tensor(f32).init(&.{ 2, 2, 2 }, testing.allocator);
    defer result.deinit();

    try opWithOut(t1, t2, &result, add);
    try testing.expectEqual(8, result.data[0]);
}

test "scalarMultiply" {
    // Test with Tensor
    const t = try Tensor(f32).splat(&.{ 2, 2, 2 }, 3.0, testing.allocator);
    defer t.deinit();
    var result_t = try Tensor(f32).init(&.{ 2, 2, 2 }, testing.allocator);
    defer result_t.deinit();

    try scalarMultiplyWithOut(t, 2.0, &result_t);

    try testing.expectEqual(@as(f32, 6.0), result_t.data[0]);

    // Test with Matrix
    const m = try Matrix(f32).splat(2, 2, 4.0, testing.allocator);
    defer m.deinit();
    var result_m = try Matrix(f32).init(2, 2, testing.allocator);
    defer result_m.deinit();

    try scalarMultiplyWithOut(m, 3.0, &result_m);

    try testing.expectEqual(@as(f32, 12.0), result_m.data[0]);

    // Test with Vector
    const v = try Vector(f32).splat(3, 5.0, testing.allocator);
    defer v.deinit();
    var result_v = try Vector(f32).init(3, testing.allocator);
    defer result_v.deinit();

    try scalarMultiplyWithOut(v, 4.0, &result_v);

    try testing.expectEqual(@as(f32, 20.0), result_v.data[0]);
}
