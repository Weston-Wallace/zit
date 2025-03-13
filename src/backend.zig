const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor_module = @import("tensor.zig");
const Tensor = tensor_module.Tensor;
const Vector = tensor_module.Vector;
const Matrix = tensor_module.Matrix;
const CpuBackend = @import("backends/cpu/CpuBackend.zig");
const fn_types = @import("fn_types.zig");

pub const Backend = struct {
    ptr: *anyopaque,
    vtable: VTable,

    pub const VTable = struct {
        // elementwise
        op: *const fn (ctx: *anyopaque, a: anytype, b: anytype, out: anytype, op_fn: fn_types.BinaryOpFn) anyerror!void,
        map: *const fn (ctx: *anyopaque, a: anytype, out: anytype, map_fn: fn_types.MapFn) anyerror!void,
        scalarMultiply: *const fn (ctx: *anyopaque, a: anytype, scalar: anytype, out: anytype) anyerror!void,
        // vector
        vectorDot: *const fn (ctx: *anyopaque, a: anytype, b: anytype, out: anytype) anyerror!void,
        vectorNorm: *const fn (ctx: *anyopaque, v: anytype, out: anytype) anyerror!void,
        // matrix - vector
        matrixVectorMultiply: *const fn (ctx: *anyopaque, m: anytype, v: anytype, out: anytype) anyerror!void,
        // matrix
        matrixMultiply: *const fn (ctx: *anyopaque, a: anytype, b: anytype, out: anytype) anyerror!void,
        matrixTranspose: *const fn (ctx: *anyopaque, m: anytype, out: anytype) anyerror!void,
    };
};
