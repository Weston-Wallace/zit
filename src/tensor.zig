const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn Tensor(comptime T: type) type {
    switch (@typeInfo(T)) {
        .Int => {},
        .Float => {},
        else => {
            @compileError("T must be a numeric type (int or float)");
        },
    }
    return struct {
        data: []T,
        shape: []usize,
        allocator: Allocator,
    };
}
