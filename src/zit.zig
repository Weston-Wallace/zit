const std = @import("std");
const tensor = @import("tensor.zig");
pub const Tensor = tensor.Tensor;
pub const Matrix = tensor.Matrix;
pub const Vector = tensor.Vector;
const Elementwise = @import("operations/elementwise.zig");
pub const ops = .{
    .elementwise = Elementwise,
};

test {
    std.testing.refAllDeclsRecursive(@This());
}
