const std = @import("std");
const Allocator = std.mem.Allocator;
const zit = @import("../zit.zig");
const TensorError = zit.TensorError;
const Tensor = zit.Tensor;
const Matrix = zit.Matrix;
const Vector = zit.Vector;

// Utility function to check if shapes are equal
pub fn ensureEqualShape(a: anytype, b: @TypeOf(a)) TensorError!void {
    const T = @TypeOf(a);
    const DataType = T.DataType;
    if (T == Tensor(DataType)) {
        if (!std.mem.eql(usize, a.shape.items, b.shape.items)) {
            return TensorError.ShapeMismatch;
        }
    } else if (T == Matrix(DataType)) {
        if (!(a.columns == b.columns and a.rows == b.rows)) {
            return TensorError.ShapeMismatch;
        }
    } else if (T == Vector(DataType)) {
        if (!(a.data.len == b.data.len)) {
            return TensorError.LengthMismatch;
        }
    } else {
        return TensorError.InvalidType;
    }
}

// Helper function to create a matching tensor type
pub fn createMatchingTensor(source: anytype, allocator: Allocator) TensorError!@TypeOf(source) {
    const T = @TypeOf(source);
    const DataType = T.DataType;

    if (T == Vector(DataType)) {
        return Vector(DataType).init(source.data.len, allocator);
    } else if (T == Matrix(DataType)) {
        return Matrix(DataType).init(source.rows, source.columns, allocator);
    } else {
        return Tensor(DataType).init(source.shape.items, allocator);
    }
}

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
        try testing.expectError(TensorError.ShapeMismatch, ensureEqualShape(t1, t3));
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
        try testing.expectError(TensorError.ShapeMismatch, ensureEqualShape(m1, m3));
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
        try testing.expectError(TensorError.LengthMismatch, ensureEqualShape(v1, v3));
    }
}
