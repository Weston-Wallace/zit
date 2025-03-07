const std = @import("std");
const Allocator = std.mem.Allocator;

const Foo = struct { x: u8 };
const Bar = struct {
    foo: Foo,
    allocator: Allocator,
};

const foo = Foo{ .x = 8 };

pub fn main() void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const bar = Bar{
        .foo = foo,
        .allocator = allocator,
    };

    std.debug.print("allocator: {s}\n", .{@typeName(@TypeOf(bar.allocator))});
}
