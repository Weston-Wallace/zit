const std = @import("std");
const zit = @import("../../zit.zig");
const Matrix = zit.Matrix;
const TensorError = zit.TensorError;

pub fn matrixMultiply(_: *anyopaque, a: anytype, b: @TypeOf(a), out: *@TypeOf(a)) TensorError!void {
    const T = @TypeOf(a);
    const DataType = T.DataType;
    if (T != Matrix(DataType)) {
        @compileError("a, b, and out must be matrix types");
    }

    if (a.columns != b.rows) {
        return TensorError.ShapeMismatch;
    }
    if (!(out.rows == a.rows and out.columns == b.columns)) {
        return TensorError.ShapeMismatch;
    }

    @memset(out.data, 0);

    // Basic matrix multiplication
    for (0..a.rows) |i| {
        for (0..b.columns) |j| {
            for (0..a.columns) |k| {
                const a_idx = i * a.columns + k;
                const b_idx = k * b.columns + j;
                const res_idx = i * b.columns + j;

                out.data[res_idx] += a.data[a_idx] * b.data[b_idx];
            }
        }
    }
}

pub fn matrixTranspose(_: *anyopaque, m: anytype, out: *@TypeOf(m)) TensorError!void {
    const DataType = @TypeOf(m).DataType;
    if (@TypeOf(m) != Matrix(DataType)) {
        @compileError("m must be a Matrix");
    }

    if (!(out.rows == m.columns and out.columns == m.rows)) {
        return TensorError.ShapeMismatch;
    }

    // Transpose the matrix
    for (0..m.rows) |i| {
        for (0..m.columns) |j| {
            const src_idx = i * m.columns + j;
            const dst_idx = j * m.rows + i;

            out.data[dst_idx] = m.data[src_idx];
        }
    }
}

const testing = std.testing;

fn emptyCtx() *anyopaque {
    return @ptrFromInt(1);
}

test matrixMultiply {
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

    var result = try Matrix(f32).init(2, 2, testing.allocator);
    defer result.deinit();

    // Multiply matrices
    try matrixMultiply(emptyCtx(), m1, m2, &result);

    // Expected result (2x2 matrix):
    // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12]
    // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]
    // = [58, 64, 139, 154]
    try testing.expectEqual(58.0, result.data[0]);
    try testing.expectEqual(64.0, result.data[1]);
    try testing.expectEqual(139.0, result.data[2]);
    try testing.expectEqual(154.0, result.data[3]);

    // Test shape mismatch error
    const wrong_m = try Matrix(f32).init(4, 4, testing.allocator);
    defer wrong_m.deinit();

    try testing.expectError(TensorError.ShapeMismatch, matrixMultiply(emptyCtx(), m1, wrong_m, &result));
}

test matrixTranspose {
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
    var result = try Matrix(f32).init(3, 2, testing.allocator);
    defer result.deinit();
    try matrixTranspose(emptyCtx(), m, &result);

    // Expected result (3x2 matrix):
    // [1, 4]
    // [2, 5]
    // [3, 6]
    try testing.expectEqual(3, result.rows);
    try testing.expectEqual(2, result.columns);
    try testing.expectEqual(1.0, result.data[0]);
    try testing.expectEqual(4.0, result.data[1]);
    try testing.expectEqual(2.0, result.data[2]);
    try testing.expectEqual(5.0, result.data[3]);
    try testing.expectEqual(3.0, result.data[4]);
    try testing.expectEqual(6.0, result.data[5]);

    // Test with square matrix
    const sq = try Matrix(f32).init(2, 2, testing.allocator);
    defer sq.deinit();
    sq.data[0] = 1.0;
    sq.data[1] = 2.0;
    sq.data[2] = 3.0;
    sq.data[3] = 4.0;

    var sq_result = try Matrix(f32).init(2, 2, testing.allocator);
    defer sq_result.deinit();
    try matrixTranspose(emptyCtx(), sq, &sq_result);

    try testing.expectEqual(2, sq_result.rows);
    try testing.expectEqual(2, sq_result.columns);
    try testing.expectEqual(1.0, sq_result.data[0]);
    try testing.expectEqual(3.0, sq_result.data[1]);
    try testing.expectEqual(2.0, sq_result.data[2]);
    try testing.expectEqual(4.0, sq_result.data[3]);
}

test "shape validation" {
    // Create matrices with incompatible dimensions for matrix multiplication
    const m1 = try Matrix(f32).splat(2, 3, 1.0, testing.allocator);
    defer m1.deinit();

    const correct_m2 = try Matrix(f32).splat(3, 2, 1.0, testing.allocator);
    defer correct_m2.deinit();

    const wrong_m2 = try Matrix(f32).splat(4, 2, 2.0, testing.allocator);
    defer wrong_m2.deinit();

    var correct_result = try Matrix(f32).init(2, 2, testing.allocator);
    defer correct_result.deinit();

    var wrong_result = try Matrix(f32).init(3, 4, testing.allocator);
    defer wrong_result.deinit();

    // Matrix multiplication should fail since dimensions don't align (2x3 * 4x2)
    try testing.expectError(TensorError.ShapeMismatch, matrixMultiply(emptyCtx(), m1, wrong_m2, &correct_result));
    try testing.expectError(TensorError.ShapeMismatch, matrixMultiply(emptyCtx(), m1, correct_m2, &wrong_result));
}
