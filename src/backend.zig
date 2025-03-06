const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor_module = @import("tensor.zig");
const Tensor = tensor_module.Tensor;
const Vector = tensor_module.Vector;
const Matrix = tensor_module.Matrix;
const CpuBackend = @import("backends/cpu/CpuBackend.zig");
const fn_types = @import("fn_types.zig");

/// Errors that can occur during tensor operations
pub const TensorOpError = error{
    ShapeMismatch,
    LengthMismatch,
    OutOfMemory,
    UnsupportedOperation,
    BackendError,
    InvalidType,
};

pub const Backend = struct {
    ptr: *anyopaque,
    vtable: VTable,

    pub const VTable = struct {
        // elementwise
        op: *const fn (ctx: *anyopaque, a: anytype, b: anytype, out: anytype, op_fn: fn_types.BinaryOpFn) TensorOpError!void,
        map: *const fn (ctx: *anyopaque, a: anytype, out: anytype, map_fn: fn_types.MapFn) TensorOpError!void,
        scalarMultiply: *const fn (ctx: *anyopaque, a: anytype, scalar: anytype, out: anytype) TensorOpError!void,
        // vector
        vectorDot: *const fn (ctx: *anyopaque, a: anytype, b: anytype, out: anytype) TensorOpError!void,
        vectorNorm: *const fn (ctx: *anyopaque, v: anytype, out: anytype) TensorOpError!void,
        // matrix - vector
        matrixVectorMultiply: *const fn (ctx: *anyopaque, m: anytype, v: anytype, out: anytype) TensorOpError!void,
        // matrix
        matrixMultiply: *const fn (ctx: *anyopaque, a: anytype, b: anytype, out: anytype) TensorOpError!void,
        matrixTranspose: *const fn (ctx: *anyopaque, m: anytype, out: anytype) TensorOpError!void,
    };
};
