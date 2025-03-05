const std = @import("std");
const Allocator = std.mem.Allocator;
const zit = @import("../../zit.zig");
const Tensor = zit.Tensor;
const Matrix = zit.Matrix;
const Vector = zit.Vector;
const TensorError = zit.TensorError;
const TensorOpError = zit.TensorOpError;
const utils = @import("../utils.zig");

const elementwise = @import("elementwise.zig");
const vector_ops = @import("vector_ops.zig");
const matrix_vector_ops = @import("matrix_vector_ops.zig");
const matrix_ops = @import("matrix_ops.zig");

pub const opWithOut = elementwise.opWithOut;
pub const scalarMultiplyWithOut = elementwise.scalarMultiplyWithOut;

pub const vectorDot = vector_ops.vectorDot;
pub const vectorNorm = vector_ops.vectorNorm;

pub const matrixVectorMultiply = matrix_vector_ops.matrixVectorMultiply;

pub const matrixMultiply = matrix_ops.matrixMultiply;
pub const matrixTranspose = matrix_ops.matrixTranspose;
