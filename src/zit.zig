const std = @import("std");
const tensor = @import("tensor.zig");
pub const Tensor = tensor.Tensor;
pub const Matrix = tensor.Matrix;
pub const Vector = tensor.Vector;
pub const TensorError = tensor.TensorError;
pub const TensorContext = @import("TensorContext.zig");
const backend = @import("backend.zig");
pub const Backend = backend.Backend;
pub const TensorOpError = backend.TensorOpError;

test {
    std.testing.refAllDeclsRecursive(@This());
}
