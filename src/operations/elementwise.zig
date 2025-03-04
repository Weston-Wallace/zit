const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor_module = @import("../tensor.zig");
const Tensor = tensor_module.Tensor;
const Matrix = tensor_module.Matrix;
const Vector = tensor_module.Vector;
const TensorContext = @import("../tensor_context.zig").TensorContext;

pub const ElementwiseError = error{
    LengthMismatch,
};

/// Expects two vectors of the same length
pub fn addVectors(comptime T: type, a: Vector(T), b: Vector(T), tensor_context: TensorContext, allocator: Allocator) !Vector(T) {
    if (a.tensor.data.len != b.tensor.data.len) {
        return ElementwiseError.LengthMismatch;
    }

    return switch (tensor_context.backend) {
        .cpu => blk: {
            const data = try allocator.alloc(T, a.tensor.data.len);
            for (a.tensor.data, b.tensor.data, data) |a_data, b_data, *data_data| {
                data_data.* = a_data + b_data;
            }
            break :blk try Vector(T).fromOwnedData(data, allocator);
        },
        .gpu => {
            @panic("Not implemented yet");
        },
    };
}

const testing = std.testing;

test addVectors {
    const a = try Vector(f32).ones(10, testing.allocator);
    defer a.deinit();
    const b = try Vector(f32).ones(10, testing.allocator);
    defer b.deinit();

    const result = try addVectors(f32, a, b, .{ .backend = .cpu }, testing.allocator);
    defer result.deinit();
    try testing.expectEqual(2, result.tensor.data[0]);
}
