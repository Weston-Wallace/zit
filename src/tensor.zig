const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

pub const TensorError = error{
    InvalidDimensions,
};

pub fn Tensor(comptime T: type) type {
    ensureNumericType(T);
    return struct {
        const Self = @This();

        data: []T,
        shape: ArrayList(usize),
        allocator: Allocator,

        pub fn init(shape: []const usize, allocator: Allocator) !Self {
            const owned_shape = try allocator.dupe(usize, shape);

            var total_size: usize = 1;
            for (shape) |length| {
                total_size *= length;
            }
            const data = try allocator.alloc(T, total_size);

            return Self{
                .data = data,
                .shape = ArrayList(usize).fromOwnedSlice(allocator, owned_shape),
                .allocator = allocator,
            };
        }

        pub fn zeros(shape: []const usize, allocator: Allocator) !Self {
            const tensor = try init(shape, allocator);
            @memset(tensor.data, 0);
            return tensor;
        }

        pub fn ones(shape: []const usize, allocator: Allocator) !Self {
            const tensor = try init(shape, allocator);
            @memset(tensor.data, 1);
            return tensor;
        }

        /// data must be initialized with the passed in allocator
        pub fn fromOwnedData(data: []T, shape: []const usize, allocator: Allocator) !Self {
            const owned_shape = try allocator.dupe(usize, shape);

            var total_size: usize = 1;
            for (shape) |length| {
                total_size *= length;
            }
            if (total_size != data.len) {
                return TensorError.InvalidDimensions;
            }

            return Self{
                .data = data,
                .shape = ArrayList(usize).fromOwnedSlice(allocator, owned_shape),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: Self) void {
            self.allocator.free(self.data);
            self.shape.deinit();
        }
    };
}

pub fn Matrix(comptime T: type) type {
    return struct {
        const Self = @This();

        tensor: Tensor(T),

        pub fn init(rows: usize, colunms: usize, allocator: Allocator) !Self {
            return Self{
                .tensor = try Tensor(T).init(&.{ rows, colunms }, allocator),
            };
        }

        pub fn zeros(rows: usize, columns: usize, allocator: Allocator) !Self {
            return Self{
                .tensor = try Tensor(T).zeros(&.{ rows, columns }, allocator),
            };
        }

        pub fn ones(rows: usize, columns: usize, allocator: Allocator) !Self {
            return Self{
                .tensor = try Tensor(T).ones(&.{ rows, columns }, allocator),
            };
        }

        pub fn fromOwnedData(data: []T, rows: usize, columns: usize, allocator: Allocator) !Self {
            return Self{
                .tensor = try Tensor(T).fromOwnedData(data, &.{ rows, columns }, allocator),
            };
        }

        pub fn deinit(self: Self) void {
            self.tensor.deinit();
        }
    };
}

pub fn Vector(comptime T: type) type {
    return struct {
        const Self = @This();

        tensor: Tensor(T),

        pub fn init(length: usize, allocator: Allocator) !Self {
            return Self{
                .tensor = try Tensor(T).init(&.{length}, allocator),
            };
        }

        pub fn zeros(length: usize, allocator: Allocator) !Self {
            return Self{
                .tensor = try Tensor(T).zeros(&.{length}, allocator),
            };
        }

        pub fn ones(length: usize, allocator: Allocator) !Self {
            return Self{
                .tensor = try Tensor(T).ones(&.{length}, allocator),
            };
        }

        pub fn fromOwnedData(data: []T, allocator: Allocator) !Self {
            return Self{
                .tensor = try Tensor(T).fromOwnedData(data, &.{data.len}, allocator),
            };
        }

        pub fn deinit(self: Self) void {
            self.tensor.deinit();
        }
    };
}

fn ensureNumericType(comptime T: type) void {
    switch (@typeInfo(T)) {
        .Int => {},
        .Float => {},
        else => {
            @compileError("T must be a numeric type (int or float)");
        },
    }
}

const testing = std.testing;

test "basic tensor creation" {
    const tensor = try Tensor(f32).zeros(&.{ 2, 2, 2 }, testing.allocator);
    defer tensor.deinit();

    try testing.expectEqual(0, tensor.data[0]);
}

test "basic matrix creation" {
    const matrix = try Matrix(f32).zeros(2, 2, testing.allocator);
    defer matrix.deinit();

    try testing.expectEqual(0, matrix.tensor.data[0]);
}

test "basic vector creation" {
    const vector = try Vector(f32).zeros(2, testing.allocator);
    defer vector.deinit();

    try testing.expectEqual(0, vector.tensor.data[0]);
}
