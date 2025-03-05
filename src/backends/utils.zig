const std = @import("std");
const zit = @import("../zit.zig");
const Tensor = zit.Tensor;
const Matrix = zit.Matrix;
const Vector = zit.Vector;
const TensorOpError = zit.TensorOpError;

// Utility function to check if shapes are equal
pub fn ensureEqualShape(a: anytype, b: @TypeOf(a)) !void {
    const T = @TypeOf(a);
    const DataType = T.DataType;
    if (T == Tensor(DataType)) {
        if (!std.mem.eql(usize, a.shape.items, b.shape.items)) {
            return TensorOpError.ShapeMismatch;
        }
    } else if (T == Matrix(DataType)) {
        if (!(a.columns == b.columns and a.rows == b.rows)) {
            return TensorOpError.ShapeMismatch;
        }
    } else if (T == Vector(DataType)) {
        if (!(a.data.len == b.data.len)) {
            return TensorOpError.LengthMismatch;
        }
    } else {
        return TensorOpError.InvalidType;
    }
}

fn exampleOpFn(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    return x + y;
}

pub const BinaryOpFn = @TypeOf(exampleOpFn);

const testing = std.testing;

test "ensureEqualShape" {
    // Test with Tensors
    {
        const t1 = try Tensor(f32).splat(&.{ 2, 3 }, 1.0, testing.allocator);
        defer t1.deinit();

        const t2 = try Tensor(f32).splat(&.{ 2, 3 }, 2.0, testing.allocator);
        defer t2.deinit();

        // Should not produce error for equal shapes
        try ensureEqualShape(t1, t2);

        const t3 = try Tensor(f32).splat(&.{ 2, 4 }, 3.0, testing.allocator);
        defer t3.deinit();

        // Should error for different shapes
        try testing.expectError(TensorOpError.ShapeMismatch, ensureEqualShape(t1, t3));
    }

    // Test with Matrices
    {
        const m1 = try Matrix(f32).splat(2, 3, 1.0, testing.allocator);
        defer m1.deinit();

        const m2 = try Matrix(f32).splat(2, 3, 2.0, testing.allocator);
        defer m2.deinit();

        // Should not produce error for equal shapes
        try ensureEqualShape(m1, m2);

        const m3 = try Matrix(f32).splat(2, 4, 3.0, testing.allocator);
        defer m3.deinit();

        // Should error for different shapes
        try testing.expectError(TensorOpError.ShapeMismatch, ensureEqualShape(m1, m3));
    }

    // Test with Vectors
    {
        const v1 = try Vector(f32).splat(3, 1.0, testing.allocator);
        defer v1.deinit();

        const v2 = try Vector(f32).splat(3, 2.0, testing.allocator);
        defer v2.deinit();

        // Should not produce error for equal lengths
        try ensureEqualShape(v1, v2);

        const v3 = try Vector(f32).splat(4, 3.0, testing.allocator);
        defer v3.deinit();

        // Should error for different lengths
        try testing.expectError(TensorOpError.LengthMismatch, ensureEqualShape(v1, v3));
    }
}
