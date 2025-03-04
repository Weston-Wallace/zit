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

const testing = std.testing;

test "basic tensor creation" {
    var data = [_]f32{ 0, 1, 2, 3 };
    var shape = [_]usize{4};
    const tensor = Tensor(f32){
        .data = &data,
        .shape = &shape,
        .allocator = testing.allocator,
    };

    try testing.expect(data[0] == tensor.data[0]);
}
