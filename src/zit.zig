const std = @import("std");
pub const Tensor = @import("tensor.zig").Tensor;

test {
    std.testing.refAllDeclsRecursive(@This());
}
