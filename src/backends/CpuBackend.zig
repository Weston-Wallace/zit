const std = @import("std");
const Allocator = std.mem.Allocator;
const zit = @import("../zit.zig");
const Tensor = zit.Tensor;
const Matrix = zit.Matrix;
const Vector = zit.Vector;
const TensorError = zit.TensorError;
const TensorOpError = zit.TensorOpError;
//-----------------------------------------------
// Generic Tensor Operations Implementation
//-----------------------------------------------

const AnyTensorError = TensorError || TensorOpError;

fn add(a: anytype, b: @TypeOf(a), allocator: Allocator) AnyTensorError!@TypeOf(a) {
    const T = @TypeOf(a);
    const DataType = T.DataType;

    try ensureEqualShape(a, b);

    const data = try allocator.alloc(DataType, a.data.len);
    errdefer allocator.free(data);

    for (a.data, b.data, data) |a_val, b_val, *result| {
        result.* = a_val + b_val;
    }

    if (T == Vector(DataType)) {
        return try Vector(DataType).fromOwnedData(data, allocator);
    } else if (T == Matrix(DataType)) {
        return try Matrix(DataType).fromOwnedData(data, a.rows, a.columns, allocator);
    } else {
        return try Tensor(DataType).fromOwnedData(data, a.shape.items, allocator);
    }
}

fn subtract(a: anytype, b: @TypeOf(a), allocator: Allocator) AnyTensorError!@TypeOf(a) {
    const T = @TypeOf(a);
    const DataType = T.DataType;

    try ensureEqualShape(a, b);

    const data = try allocator.alloc(DataType, a.data.len);
    errdefer allocator.free(data);

    for (a.data, b.data, data) |a_val, b_val, *result| {
        result.* = a_val - b_val;
    }

    if (T == Vector(DataType)) {
        return try Vector(DataType).fromOwnedData(data, allocator);
    } else if (T == Matrix(DataType)) {
        return try Matrix(DataType).fromOwnedData(data, a.rows, a.columns, allocator);
    } else {
        return try Tensor(DataType).fromOwnedData(data, a.shape.items, allocator);
    }
}

fn multiply(a: anytype, b: @TypeOf(a), allocator: Allocator) AnyTensorError!@TypeOf(a) {
    // Element-wise multiplication (Hadamard product)
    const T = @TypeOf(a);
    const DataType = T.DataType;

    try ensureEqualShape(a, b);

    const data = try allocator.alloc(DataType, a.data.len);
    errdefer allocator.free(data);

    for (a.data, b.data, data) |a_val, b_val, *result| {
        result.* = a_val * b_val;
    }

    if (T == Vector(DataType)) {
        return try Vector(DataType).fromOwnedData(data, allocator);
    } else if (T == Matrix(DataType)) {
        return try Matrix(DataType).fromOwnedData(data, a.rows, a.columns, allocator);
    } else {
        return try Tensor(DataType).fromOwnedData(data, a.shape.items, allocator);
    }
}

fn scalarMultiply(a: anytype, scalar: anytype, allocator: Allocator) AnyTensorError!@TypeOf(a) {
    const T = @TypeOf(a);
    const DataType = T.DataType;
    if (DataType != @TypeOf(@as(DataType, scalar))) {
        @compileError("scalar must be the same type as the data type of a");
    }

    const data = try allocator.alloc(DataType, a.data.len);
    errdefer allocator.free(data);

    for (a.data, data) |a_val, *result| {
        result.* = a_val * scalar;
    }

    if (T == Vector(DataType)) {
        return try Vector(DataType).fromOwnedData(data, allocator);
    } else if (T == Matrix(DataType)) {
        return try Matrix(DataType).fromOwnedData(data, a.rows, a.columns, allocator);
    } else {
        return try Tensor(DataType).fromOwnedData(data, a.shape.items, allocator);
    }
}

//-----------------------------------------------
// Vector-specific Operations Implementation
//-----------------------------------------------

fn vectorDot(a: anytype, b: @TypeOf(a)) TensorOpError!@TypeOf(a).DataType {
    const DataType = @TypeOf(a).DataType;
    if (@TypeOf(a) != Vector(DataType)) {
        @compileError("a and b must be Vectors");
    }

    // Ensure vectors have the same length
    if (a.data.len != b.data.len) {
        return TensorOpError.LengthMismatch;
    }

    var result: DataType = 0;

    for (a.data, b.data) |a_val, b_val| {
        result += a_val * b_val;
    }

    return result;
}

fn vectorNorm(v: anytype) TensorOpError!@TypeOf(v).DataType {
    const DataType = @TypeOf(v).DataType;
    if (@TypeOf(v) != Vector(DataType)) {
        @compileError("v must be a Vector");
    }

    var sum_sq: DataType = 0;

    for (v.data) |val| {
        sum_sq += val * val;
    }

    return @sqrt(sum_sq);
}

//-----------------------------------------------
// Matrix-Vector Operations Implementation
//-----------------------------------------------

fn matrixVectorMultiply(m: anytype, v: anytype, allocator: Allocator) AnyTensorError!@TypeOf(v) {
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

//-----------------------------------------------
// Matrix-specific Operations Implementation
//-----------------------------------------------

fn matrixMultiply(a: anytype, b: @TypeOf(a), allocator: Allocator) AnyTensorError!@TypeOf(a) {
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

fn matrixTranspose(m: anytype, allocator: Allocator) AnyTensorError!@TypeOf(m) {
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

// Utility function to check if shapes are equal
fn ensureEqualShape(a: anytype, b: @TypeOf(a)) !void {
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

test "vectorDot" {
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

    const result = try vectorDot(v1, v2);
    // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    try testing.expectEqual(@as(f32, 32.0), result);

    // Test error case: vectors of different lengths
    const v3 = try Vector(f32).init(4, testing.allocator);
    defer v3.deinit();

    try testing.expectError(TensorOpError.LengthMismatch, vectorDot(v1, v3));
}

test "vectorNorm" {
    // Initialize vector with specific values
    const v = try Vector(f32).init(3, testing.allocator);
    defer v.deinit();
    v.data[0] = 3.0;
    v.data[1] = 4.0;
    v.data[2] = 0.0;

    const result = try vectorNorm(v);
    // Expected: sqrt(3^2 + 4^2 + 0^2) = sqrt(9 + 16) = sqrt(25) = 5
    try testing.expectEqual(@as(f32, 5.0), result);

    // Test with zero vector
    const zero_v = try Vector(f32).splat(3, 0.0, testing.allocator);
    defer zero_v.deinit();

    const zero_result = try vectorNorm(zero_v);
    try testing.expectEqual(@as(f32, 0.0), zero_result);

    // Test with negative values
    const neg_v = try Vector(f32).init(2, testing.allocator);
    defer neg_v.deinit();
    neg_v.data[0] = -3.0;
    neg_v.data[1] = 4.0;

    const neg_result = try vectorNorm(neg_v);
    // Expected: sqrt((-3)^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    try testing.expectEqual(@as(f32, 5.0), neg_result);
}

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

    // Test dot product with integers
    try testing.expectEqual(@as(i32, 30), try vectorDot(v1, v2)); // 5*2 + 5*2 + 5*2 = 30
}

test "matrix operations with different dimensions" {
    // Test matrix operations with non-square matrices

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

    // Element-wise operations should fail with shape mismatch
    try testing.expectError(TensorOpError.ShapeMismatch, add(m1, m2, testing.allocator));
}

test "edge cases" {
    // Test with empty tensors and vectors

    // Empty vector (length 0)
    const empty_v = try Vector(f32).init(0, testing.allocator);
    defer empty_v.deinit();

    // Vector norm of empty vector should be 0
    try testing.expectEqual(@as(f32, 0.0), try vectorNorm(empty_v));

    // Empty matrix (0x0)
    const empty_m = try Matrix(f32).init(0, 0, testing.allocator);
    defer empty_m.deinit();

    // Empty tensor (shape {0, 0})
    const empty_t = try Tensor(f32).init(&.{ 0, 0 }, testing.allocator);
    defer empty_t.deinit();

    // Operations with empty tensors/matrices/vectors
    const empty_v2 = try Vector(f32).init(0, testing.allocator);
    defer empty_v2.deinit();

    // Add two empty vectors should work
    const empty_sum = try add(empty_v, empty_v2, testing.allocator);
    defer empty_sum.deinit();
    try testing.expectEqual(@as(usize, 0), empty_sum.data.len);
}

test "boundary values" {
    // Test with very large and very small values

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
    const large_norm = try vectorNorm(large_v);
    try testing.expect(large_norm > 0);
    try testing.expect(!std.math.isNan(large_norm));

    // Test vector norm with small values
    const small_norm = try vectorNorm(small_v);
    try testing.expect(small_norm >= 0);
    try testing.expect(!std.math.isNan(small_norm));

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

test "large tensors" {
    // Create larger tensors to test performance and memory handling
    const side_length: usize = 50;
    const large_m1 = try Matrix(f32).splat(side_length, side_length, 1.0, testing.allocator);
    defer large_m1.deinit();

    const large_m2 = try Matrix(f32).splat(side_length, side_length, 2.0, testing.allocator);
    defer large_m2.deinit();

    // Test element-wise addition with large matrices
    const large_sum = try add(large_m1, large_m2, testing.allocator);
    defer large_sum.deinit();

    // Check a few sample values
    try testing.expectEqual(@as(f32, 3.0), large_sum.data[0]);
    try testing.expectEqual(@as(f32, 3.0), large_sum.data[side_length * side_length - 1]);

    // Create a large vector for matrix-vector multiplication
    const large_v = try Vector(f32).splat(side_length, 1.0, testing.allocator);
    defer large_v.deinit();

    // Test matrix-vector multiplication with large matrix
    const mv_result = try matrixVectorMultiply(large_m1, large_v, testing.allocator);
    defer mv_result.deinit();

    // Since all values are 1.0, each result element should equal the column count
    try testing.expectEqual(@as(f32, @floatFromInt(side_length)), mv_result.data[0]);
}

test "memory management" {
    // Test proper memory management with allocator pattern
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    // Create tensors with the arena allocator
    const t1 = try Tensor(f32).splat(&.{ 10, 10 }, 1.0, arena_allocator);
    const t2 = try Tensor(f32).splat(&.{ 10, 10 }, 2.0, arena_allocator);

    // Perform operations
    const result1 = try add(t1, t2, arena_allocator);
    const result2 = try multiply(result1, t1, arena_allocator);

    // Check results
    try testing.expectEqual(@as(f32, 3.0), result1.data[0]);
    try testing.expectEqual(@as(f32, 3.0), result2.data[0]);

    // No need to call deinit() on individual tensors as the arena will free everything
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
    // Test shape validation for operations

    // Create tensors with different shapes
    const t1 = try Tensor(f32).splat(&.{ 2, 3 }, 1.0, testing.allocator);
    defer t1.deinit();

    const t2 = try Tensor(f32).splat(&.{ 3, 2 }, 2.0, testing.allocator);
    defer t2.deinit();

    // Should fail due to shape mismatch
    try testing.expectError(TensorOpError.ShapeMismatch, add(t1, t2, testing.allocator));
    try testing.expectError(TensorOpError.ShapeMismatch, subtract(t1, t2, testing.allocator));
    try testing.expectError(TensorOpError.ShapeMismatch, multiply(t1, t2, testing.allocator));

    // Create matrices with incompatible dimensions for matrix multiplication
    const m1 = try Matrix(f32).splat(2, 3, 1.0, testing.allocator);
    defer m1.deinit();

    const m2 = try Matrix(f32).splat(4, 2, 2.0, testing.allocator);
    defer m2.deinit();

    // Matrix multiplication should fail since dimensions don't align (2x3 * 4x2)
    try testing.expectError(TensorOpError.ShapeMismatch, matrixMultiply(m1, m2, testing.allocator));

    // Create a vector with length not matching matrix columns
    const v = try Vector(f32).splat(4, 1.0, testing.allocator);
    defer v.deinit();

    // Matrix-vector multiplication should fail (columns ≠ vector length)
    try testing.expectError(TensorOpError.ShapeMismatch, matrixVectorMultiply(m1, v, testing.allocator));
}

// Test helper functions and special cases
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
