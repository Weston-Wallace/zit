const std = @import("std");
const tensor = @import("tensor.zig");
pub const Tensor = tensor.Tensor;
pub const Matrix = tensor.Matrix;
pub const Vector = tensor.Vector;
pub const ops = @import("operations/elementwise.zig");

test {
    std.testing.refAllDeclsRecursive(@This());
}
