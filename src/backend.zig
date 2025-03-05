const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor_module = @import("tensor.zig");
const Tensor = tensor_module.Tensor;
const Vector = tensor_module.Vector;
const Matrix = tensor_module.Matrix;
const CpuBackend = @import("backends/cpu/CpuBackend.zig");

/// Errors that can occur during tensor operations
pub const TensorOpError = error{
    ShapeMismatch,
    LengthMismatch,
    OutOfMemory,
    UnsupportedOperation,
    BackendError,
    InvalidType,
};

/// Backend interface that defines operations all backends must implement
pub const Backend = union(enum) {
    cpu: CpuBackend,
};
