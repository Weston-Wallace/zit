const std = @import("std");
const Allocator = std.mem.Allocator;
const zit = @import("../../zit.zig");
const Tensor = zit.Tensor;
const Matrix = zit.Matrix;
const Vector = zit.Vector;
const TensorError = zit.TensorError;
const TensorOpError = zit.TensorOpError;
const Backend = @import("../../backend.zig").Backend;

const elementwise = @import("elementwise.zig");
const vector_ops = @import("vector_ops.zig");
const matrix_vector_ops = @import("matrix_vector_ops.zig");
const matrix_ops = @import("matrix_ops.zig");

pub fn backend() Backend {
    return Backend{
        .ptr = @ptrFromInt(0),
        .vtable = .{
            .op = elementwise.op,
            .map = elementwise.map,
            .scalarMultiply = elementwise.scalarMultiply,
            .vectorDot = vector_ops.vectorDot,
            .vectorNorm = vector_ops.vectorNorm,
            .matrixVectorMultiply = matrix_vector_ops.matrixVectorMultiply,
            .matrixMultiply = matrix_ops.matrixMultiply,
            .matrixTranspose = matrix_ops.matrixTranspose,
        },
    };
}
