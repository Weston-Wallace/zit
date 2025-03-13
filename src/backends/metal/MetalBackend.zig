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
const metal_context = @import("metal_context.zig");

// Initialize the Metal backend once
pub fn init(allocator: Allocator) !void {
    try metal_context.init(allocator);
}

// Clean up Metal resources
pub fn deinit() void {
    metal_context.deinit();
}

// Check if Metal is available on this system
pub fn isAvailable() bool {
    return metal_context.isAvailable();
}

pub const backend = Backend{
    .ptr = @ptrFromInt(1),
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
