const std = @import("std");
const Allocator = std.mem.Allocator;
const zit = @import("../../zit.zig");
const Matrix = zit.Matrix;
const Vector = zit.Vector;
const TensorError = zit.TensorError;
const TensorOpError = zit.TensorOpError;

pub fn matrixVectorMultiply(m: anytype, v: anytype, allocator: Allocator) TensorError!@TypeOf(v) {
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
    const DataType = MType.DataType;

    // Ensure matrix columns match vector length
    if (m.columns != v.data.len) {
        return TensorOpError.ShapeMismatch;
    }

    const result_data = try allocator.alloc(DataType, m.rows);
    errdefer allocator.free(result_data);

    @memset(result_data, 0);

    // Compute M * v
    for (0..m.rows) |i| {
        for (0..m.columns) |j| {
            const m_idx = i * m.columns + j;
            result_data[i] += m.data[m_idx] * v.data[j];
        }
    }

    return try Vector(DataType).fromOwnedData(result_data, allocator);
}

const testing = std.testing;

test "matrixVectorMultiply" {
    // Create and initialize a 2x3 matrix
    const m = try Matrix(f32).init(2, 3, testing.allocator);
    defer m.deinit();
    m.data[0] = 1.0;
    m.data[1] = 2.0;
    m.data[2] = 3.0;
    m.data[3] = 4.0;
    m.data[4] = 5.0;
    m.data[5] = 6.0;

    // Create and initialize a 3-element vector
    const v = try Vector(f32).init(3, testing.allocator);
    defer v.deinit();
    v.data[0] = 7.0;
    v.data[1] = 8.0;
    v.data[2] = 9.0;

    // Multiply matrix by vector
    const result = try matrixVectorMultiply(m, v, testing.allocator);
    defer result.deinit();

    // Expected: [1*7 + 2*8 + 3*9, 4*7 + 5*8 + 6*9] = [50, 122]
    try testing.expectEqual(@as(f32, 50.0), result.data[0]);
    try testing.expectEqual(@as(f32, 122.0), result.data[1]);

    // Test shape mismatch error
    const wrong_v = try Vector(f32).init(4, testing.allocator);
    defer wrong_v.deinit();

    try testing.expectError(TensorOpError.ShapeMismatch, matrixVectorMultiply(m, wrong_v, testing.allocator));
}

test "shape validation" {
    // Create matrices with incompatible dimensions for matrix multiplication
    const m1 = try Matrix(f32).splat(2, 3, 1.0, testing.allocator);
    defer m1.deinit();

    // Create a vector with length not matching matrix columns
    const v = try Vector(f32).splat(4, 1.0, testing.allocator);
    defer v.deinit();

    // Matrix-vector multiplication should fail (columns ≠ vector length)
    try testing.expectError(TensorOpError.ShapeMismatch, matrixVectorMultiply(m1, v, testing.allocator));
}
