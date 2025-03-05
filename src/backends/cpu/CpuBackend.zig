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

pub const add = elementwise.add;
pub const addInPlace = elementwise.addInPlace;
pub const addWithOut = elementwise.addWithOut;
pub const subtract = elementwise.subtract;
pub const subtractInPlace = elementwise.subtractInPlace;
pub const subtractWithOut = elementwise.subtractWithOut;
pub const multiply = elementwise.multiply;
pub const multiplyInPlace = elementwise.multiplyInPlace;
pub const multiplyWithOut = elementwise.multiplyWithOut;
pub const divide = elementwise.divide;
pub const divideInPlace = elementwise.divideInPlace;
pub const divideWithOut = elementwise.divideWithOut;
pub const scalarMultiply = elementwise.scalarMultiply;
pub const scalarMultiplyInPlace = elementwise.scalarMultiplyInPlace;
pub const scalarMultiplyWithOut = elementwise.scalarMultiplyWithOut;

pub const vectorDot = vector_ops.vectorDot;
pub const vectorNorm = vector_ops.vectorNorm;

pub const matrixVectorMultiply = matrix_vector_ops.matrixVectorMultiply;

pub const matrixMultiply = matrix_ops.matrixMultiply;
pub const matrixTranspose = matrix_ops.matrixTranspose;
