const std = @import("std");
const tensor = @import("tensor.zig");
pub const Tensor = tensor.Tensor;
pub const Matrix = tensor.Matrix;
pub const Vector = tensor.Vector;
pub const TensorStructError = tensor.TensorStructError;
pub const TensorContext = @import("TensorContext.zig");
const backend = @import("backend.zig");
pub const Backend = backend.Backend;
pub const CpuBackend = @import("backends/cpu/CpuBackend.zig");
pub const TensorOpError = backend.TensorOpError;
pub const TensorError = TensorStructError || TensorOpError;
pub const fn_types = @import("fn_types.zig");

test {
    std.testing.refAllDeclsRecursive(@This());
}
