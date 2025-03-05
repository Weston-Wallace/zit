const std = @import("std");
const Allocator = std.mem.Allocator;
const zit = @import("../../zit.zig");
const Matrix = zit.Matrix;
const TensorError = zit.TensorError;
const TensorOpError = zit.TensorOpError;

pub fn matrixMultiply(a: anytype, b: @TypeOf(a), allocator: Allocator) TensorError!@TypeOf(a) {
    const T = @TypeOf(a);
    const DataType = T.DataType;
    if (T != Matrix(DataType)) {
        @compileError("a and b must be matrix types");
    }

    if (a.columns != b.rows) {
        return TensorOpError.ShapeMismatch;
    }

    const result_data = try allocator.alloc(DataType, a.rows * b.columns);
    errdefer allocator.free(result_data);

    @memset(result_data, 0);

    // Basic matrix multiplication
    for (0..a.rows) |i| {
        for (0..b.columns) |j| {
            for (0..a.columns) |k| {
                const a_idx = i * a.columns + k;
                const b_idx = k * b.columns + j;
                const res_idx = i * b.columns + j;

                result_data[res_idx] += a.data[a_idx] * b.data[b_idx];
            }
        }
    }

    return try Matrix(DataType).fromOwnedData(result_data, a.rows, b.columns, allocator);
}

pub fn matrixTranspose(m: anytype, allocator: Allocator) TensorError!@TypeOf(m) {
    const DataType = @TypeOf(m).DataType;
    if (@TypeOf(m) != Matrix(DataType)) {
        @compileError("m must be a Matrix");
    }

    const result_data = try allocator.alloc(DataType, m.rows * m.columns);
    errdefer allocator.free(result_data);

    // Transpose the matrix
    for (0..m.rows) |i| {
        for (0..m.columns) |j| {
            const src_idx = i * m.columns + j;
            const dst_idx = j * m.rows + i;

            result_data[dst_idx] = m.data[src_idx];
        }
    }

    return try Matrix(DataType).fromOwnedData(result_data, m.columns, m.rows, allocator);
}

const testing = std.testing;

test "matrixMultiply" {
    // Create and initialize a 2x3 matrix
    const m1 = try Matrix(f32).init(2, 3, testing.allocator);
    defer m1.deinit();
    m1.data[0] = 1.0;
    m1.data[1] = 2.0;
    m1.data[2] = 3.0;
    m1.data[3] = 4.0;
    m1.data[4] = 5.0;
    m1.data[5] = 6.0;

    // Create and initialize a 3x2 matrix
    const m2 = try Matrix(f32).init(3, 2, testing.allocator);
    defer m2.deinit();
    m2.data[0] = 7.0;
    m2.data[1] = 8.0;
    m2.data[2] = 9.0;
    m2.data[3] = 10.0;
    m2.data[4] = 11.0;
    m2.data[5] = 12.0;

    // Multiply matrices
    const result = try matrixMultiply(m1, m2, testing.allocator);
    defer result.deinit();

    // Expected result (2x2 matrix):
    // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12]
    // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]
    // = [58, 64, 139, 154]
    try testing.expectEqual(@as(f32, 58.0), result.data[0]);
    try testing.expectEqual(@as(f32, 64.0), result.data[1]);
    try testing.expectEqual(@as(f32, 139.0), result.data[2]);
    try testing.expectEqual(@as(f32, 154.0), result.data[3]);

    // Test shape mismatch error
    const wrong_m = try Matrix(f32).init(4, 4, testing.allocator);
    defer wrong_m.deinit();

    try testing.expectError(TensorOpError.ShapeMismatch, matrixMultiply(m1, wrong_m, testing.allocator));
}

test "matrixTranspose" {
    // Create and initialize a 2x3 matrix
    const m = try Matrix(f32).init(2, 3, testing.allocator);
    defer m.deinit();
    m.data[0] = 1.0;
    m.data[1] = 2.0;
    m.data[2] = 3.0;
    m.data[3] = 4.0;
    m.data[4] = 5.0;
    m.data[5] = 6.0;

    // Transpose the matrix
    const result = try matrixTranspose(m, testing.allocator);
    defer result.deinit();

    // Expected result (3x2 matrix):
    // [1, 4]
    // [2, 5]
    // [3, 6]
    try testing.expectEqual(@as(usize, 3), result.rows);
    try testing.expectEqual(@as(usize, 2), result.columns);
    try testing.expectEqual(@as(f32, 1.0), result.data[0]);
    try testing.expectEqual(@as(f32, 4.0), result.data[1]);
    try testing.expectEqual(@as(f32, 2.0), result.data[2]);
    try testing.expectEqual(@as(f32, 5.0), result.data[3]);
    try testing.expectEqual(@as(f32, 3.0), result.data[4]);
    try testing.expectEqual(@as(f32, 6.0), result.data[5]);

    // Test with square matrix
    const sq = try Matrix(f32).init(2, 2, testing.allocator);
    defer sq.deinit();
    sq.data[0] = 1.0;
    sq.data[1] = 2.0;
    sq.data[2] = 3.0;
    sq.data[3] = 4.0;

    const sq_result = try matrixTranspose(sq, testing.allocator);
    defer sq_result.deinit();

    try testing.expectEqual(@as(usize, 2), sq_result.rows);
    try testing.expectEqual(@as(usize, 2), sq_result.columns);
    try testing.expectEqual(@as(f32, 1.0), sq_result.data[0]);
    try testing.expectEqual(@as(f32, 3.0), sq_result.data[1]);
    try testing.expectEqual(@as(f32, 2.0), sq_result.data[2]);
    try testing.expectEqual(@as(f32, 4.0), sq_result.data[3]);
}

test "matrix operations with different dimensions" {
    // Create a 3x2 matrix
    const m1 = try Matrix(f32).init(3, 2, testing.allocator);
    defer m1.deinit();
    m1.data[0] = 1.0;
    m1.data[1] = 2.0;
    m1.data[2] = 3.0;
    m1.data[3] = 4.0;
    m1.data[4] = 5.0;
    m1.data[5] = 6.0;

    // Create a 2x4 matrix
    const m2 = try Matrix(f32).init(2, 4, testing.allocator);
    defer m2.deinit();
    m2.data[0] = 7.0;
    m2.data[1] = 8.0;
    m2.data[2] = 9.0;
    m2.data[3] = 10.0;
    m2.data[4] = 11.0;
    m2.data[5] = 12.0;
    m2.data[6] = 13.0;
    m2.data[7] = 14.0;

    // Matrix multiplication should work (3x2 * 2x4 = 3x4)
    const mm_result = try matrixMultiply(m1, m2, testing.allocator);
    defer mm_result.deinit();

    try testing.expectEqual(@as(usize, 3), mm_result.rows);
    try testing.expectEqual(@as(usize, 4), mm_result.columns);

    // Expected first row: [1*7 + 2*11, 1*8 + 2*12, 1*9 + 2*13, 1*10 + 2*14] = [29, 32, 35, 38]
    try testing.expectEqual(@as(f32, 29.0), mm_result.data[0]);
    try testing.expectEqual(@as(f32, 32.0), mm_result.data[1]);
    try testing.expectEqual(@as(f32, 35.0), mm_result.data[2]);
    try testing.expectEqual(@as(f32, 38.0), mm_result.data[3]);
}

test "shape validation" {
    // Create matrices with incompatible dimensions for matrix multiplication
    const m1 = try Matrix(f32).splat(2, 3, 1.0, testing.allocator);
    defer m1.deinit();

    const m2 = try Matrix(f32).splat(4, 2, 2.0, testing.allocator);
    defer m2.deinit();

    // Matrix multiplication should fail since dimensions don't align (2x3 * 4x2)
    try testing.expectError(TensorOpError.ShapeMismatch, matrixMultiply(m1, m2, testing.allocator));
}

test "helper functions" {
    // Create a matrix with specific values
    const m = try Matrix(f32).init(2, 2, testing.allocator);
    defer m.deinit();
    m.data[0] = 1.0;
    m.data[1] = 2.0;
    m.data[2] = 3.0;
    m.data[3] = 4.0;

    // Transpose and multiply to create matrix square
    const m_trans = try matrixTranspose(m, testing.allocator);
    defer m_trans.deinit();

    const m_squared = try matrixMultiply(m, m_trans, testing.allocator);
    defer m_squared.deinit();

    // Expected for m * m_trans:
    // [1 2] [1 3] = [1*1+2*2 1*3+2*4] = [5 11]
    // [3 4] [2 4]   [3*1+4*2 3*3+4*4]   [11 25]
    try testing.expectEqual(@as(f32, 5.0), m_squared.data[0]);
    try testing.expectEqual(@as(f32, 11.0), m_squared.data[1]);
    try testing.expectEqual(@as(f32, 11.0), m_squared.data[2]);
    try testing.expectEqual(@as(f32, 25.0), m_squared.data[3]);
}
