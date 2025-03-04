const std = @import("std");
const testing = std.testing;
const zit = @import("zit");

test "create a tensor" {
    var data = [_]f32{ 0, 1, 2, 3 };
    var shape = [_]usize{4};
    const tensor = zit.Tensor(f32){
        .data = &data,
        .shape = &shape,
        .allocator = testing.allocator,
    };
    try testing.expect(tensor.data[0] == 0);
}
