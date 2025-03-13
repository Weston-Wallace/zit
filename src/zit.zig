const std = @import("std");
const tensor = @import("tensor.zig");
pub const Tensor = tensor.Tensor;
pub const Matrix = tensor.Matrix;
pub const Vector = tensor.Vector;
pub const TensorError = tensor.TensorError;
pub const TensorContext = @import("tensor_context.zig").TensorContext;
const backend = @import("backend.zig");
pub const Backend = backend.Backend;
pub const CpuBackend = @import("backends/cpu/CpuBackend.zig");
pub const SimdBackend = @import("backends/simd/SimdBackend.zig");
pub const MetalBackend = @import("backends/metal/MetalBackend.zig");
pub const Metal = @import("backends/metal/metal.zig");
pub const fn_types = @import("fn_types.zig");

/// Initialize all available backends
pub fn init(allocator: std.mem.Allocator) !void {
    // Initialize Metal backend if available
    if (Metal.isAvailable()) {
        try Metal.init(allocator);
    }
}

/// Clean up resources used by all backends
pub fn deinit() void {
    Metal.deinit();
}

test {
    std.testing.refAllDeclsRecursive(@This());
}
