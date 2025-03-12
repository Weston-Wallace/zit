const std = @import("std");
const zit = @import("../../zit.zig");
const Matrix = zit.Matrix;
const Vector = zit.Vector;
const TensorOpError = zit.TensorOpError;
const chunk_size = @import("SimdBackend.zig").chunk_size;

pub fn matrixVectorMultiply(_: *anyopaque, m: anytype, v: anytype, out: *@TypeOf(v)) TensorOpError!void {
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
        return TensorOpError.ShapeMismatch;
    }
    if (m.rows != out.data.len) {
        return TensorOpError.ShapeMismatch;
    }

    @memset(out.data, 0);

    const DataType = MType.DataType;

    // Use SIMD acceleration if columns are sufficient
    if (m.columns >= chunk_size) {
        const chunk_count = m.columns / chunk_size;

        for (0..m.rows) |i| {
            var acc: @Vector(chunk_size, DataType) = @splat(0);

            // Process chunks of the row with SIMD
            for (0..chunk_count) |chunk| {
                const m_offset = i * m.columns + chunk * chunk_size;
                const m_chunk: @Vector(chunk_size, DataType) = m.data[m_offset..][0..chunk_size].*;
                const v_chunk: @Vector(chunk_size, DataType) = v.data[chunk * chunk_size ..][0..chunk_size].*;

                // Multiply and accumulate
                acc += m_chunk * v_chunk;
            }

            // Sum the accumulated values
            var row_sum: DataType = 0;
            for (0..chunk_size) |j| {
                row_sum += acc[j];
            }

            // Handle remaining elements
            var j: usize = chunk_count * chunk_size;
            while (j < m.columns) : (j += 1) {
                row_sum += m.data[i * m.columns + j] * v.data[j];
            }

            out.data[i] = row_sum;
        }
    } else {
        // Fall back to scalar implementation
        for (0..m.rows) |i| {
            for (0..m.columns) |j| {
                const m_idx = i * m.columns + j;
                out.data[i] += m.data[m_idx] * v.data[j];
            }
        }
    }
}

const testing = std.testing;

fn emptyCtx() *anyopaque {
    return @ptrFromInt(1);
}

test matrixVectorMultiply {
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

    var result = try Vector(f32).init(m.rows, testing.allocator);
    defer result.deinit();

    // Multiply matrix by vector
    try matrixVectorMultiply(emptyCtx(), m, v, &result);

    // Expected: [1*7 + 2*8 + 3*9, 4*7 + 5*8 + 6*9] = [50, 122]
    try testing.expectEqual(50.0, result.data[0]);
    try testing.expectEqual(122.0, result.data[1]);

    // Test shape mismatch error
    const wrong_v = try Vector(f32).init(4, testing.allocator);
    defer wrong_v.deinit();

    try testing.expectError(TensorOpError.ShapeMismatch, matrixVectorMultiply(emptyCtx(), m, wrong_v, &result));
}

test "shape validation" {
    // Create matrices with incompatible dimensions for matrix multiplication
    const m1 = try Matrix(f32).splat(2, 3, 1.0, testing.allocator);
    defer m1.deinit();

    const correct_v = try Vector(f32).splat(3, 1.0, testing.allocator);
    defer correct_v.deinit();

    // Create a vector with length not matching matrix columns
    const wrong_v = try Vector(f32).splat(4, 1.0, testing.allocator);
    defer wrong_v.deinit();

    var correct_result = try Vector(f32).init(2, testing.allocator);
    defer correct_result.deinit();

    var wrong_result = try Vector(f32).init(3, testing.allocator);
    defer wrong_result.deinit();

    // Matrix-vector multiplication should fail (columns ≠ vector length)
    try testing.expectError(TensorOpError.ShapeMismatch, matrixVectorMultiply(emptyCtx(), m1, wrong_v, &correct_result));
    try testing.expectError(TensorOpError.ShapeMismatch, matrixVectorMultiply(emptyCtx(), m1, correct_v, &wrong_result));
}
