const std = @import("std");
const Allocator = std.mem.Allocator;
const Backend = @import("backend.zig").Backend;
const zit = @import("zit.zig");
const Tensor = zit.Tensor;
const Matrix = zit.Matrix;
const Vector = zit.Vector;
const TensorStructError = zit.TensorStructError;

backend: Backend = .cpu,
allocator: Allocator,

pub fn tensorInit(self: Self, comptime T: type, shape: []const usize) TensorStructError!Tensor(T) {
    return try Tensor(T).init(shape, self.allocator);
}

pub fn tensorSplat(self: Self, comptime T: type, shape: []const usize, scalar: T) TensorStructError!Tensor(T) {
    return try Tensor(T).splat(shape, scalar, self.allocator);
}

// TODO: implement the rest of the initialization methods
// TODO: implement the function generation for generating helper functions that use the elementwise operations

const Self = @This();
