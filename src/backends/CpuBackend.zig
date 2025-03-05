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
    const DataType = a.DataType;

    try ensureEqualShape(a, b);

    const data = try allocator.alloc(DataType, a.data.len);
    errdefer allocator.free(data);

    for (a.data, b.data, data) |a_val, b_val, *result| {
        result.* = a_val - b_val;
    }

    if (T == Vector(DataType)) {
        return try Vector(DataType).fromOwnedData(data, allocator);
    } else if (T == Matrix(DataType)) {
        return try Matrix(DataType).fromOwnedData(data, a.rows, a.cols, allocator);
    } else {
        return try Tensor(DataType).fromOwnedData(data, a.shape.items, allocator);
    }
}

fn multiply(a: anytype, b: @TypeOf(a), allocator: Allocator) AnyTensorError!@TypeOf(a) {
    // Element-wise multiplication (Hadamard product)
    const T = @TypeOf(a);
    const DataType = a.DataType;

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
        return try Tensor(DataType).fromOwnedData(data, a.tensor.shape.items, allocator);
    }
}

fn scalarMultiply(a: anytype, scalar: anytype, allocator: Allocator) AnyTensorError!@TypeOf(a) {
    const T = @TypeOf(a);
    const DataType = T.DataType;
    if (DataType != @TypeOf(scalar)) {
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
        return try Matrix(DataType).fromOwnedData(data, a.rows, a.cols, allocator);
    } else {
        return try Tensor(DataType).fromOwnedData(data, a.tensor.shape.items, allocator);
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
    const rows = m.shape.items[0];
    const cols = m.shape.items[1];

    if (cols != v.data.len) {
        return TensorOpError.ShapeMismatch;
    }

    const result_data = try allocator.alloc(DataType, rows);
    errdefer allocator.free(result_data);

    @memset(result_data, 0);

    // Compute M * v
    for (0..rows) |i| {
        for (0..cols) |j| {
            const m_idx = i * cols + j;
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

    const result_data = try allocator.alloc(DataType, a.rows * b.cols);
    errdefer allocator.free(result_data);

    @memset(result_data, 0);

    // Basic matrix multiplication
    for (0..a.rows) |i| {
        for (0..b.cols) |j| {
            for (0..a.cols) |k| {
                const a_idx = i * a.cols + k;
                const b_idx = k * b.cols + j;
                const res_idx = i * b.cols + j;

                result_data[res_idx] += a.tensor.data[a_idx] * b.tensor.data[b_idx];
            }
        }
    }

    return try Matrix(DataType).fromOwnedData(result_data, a.rows, b.cols, allocator);
}

fn matrixTranspose(m: anytype, allocator: Allocator) TensorOpError!@TypeOf(m) {
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
        if (!(a.columns == b.columns) and a.rows == b.rows) {
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

test add {
    const t = try Tensor(f32).splat(&.{ 2, 3, 4 }, 1, testing.allocator);
    defer t.deinit();

    const result = try add(t, t, testing.allocator);
    defer result.deinit();
    try testing.expectEqual(2, result.data[0]);

    const m = try Matrix(f32).splat(2, 2, 1, testing.allocator);
    defer m.deinit();

    const result1 = try add(m, m, testing.allocator);
    defer result1.deinit();
    try testing.expectEqual(2, result1.data[0]);

    const v = try Vector(f32).splat(2, 1, testing.allocator);
    defer v.deinit();

    const result2 = try add(v, v, testing.allocator);
    defer result2.deinit();
    try testing.expectEqual(2, result2.data[0]);
}
