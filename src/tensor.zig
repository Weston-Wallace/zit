const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

pub const TensorStructError = error{
    InvalidDimensions,
    OutOfBounds,
};

pub fn Tensor(comptime T: type) type {
    ensureNumericType(T);
    return struct {
        const Self = @This();

        pub const DataType = T;

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

        pub fn splat(shape: []const usize, scalar: T, allocator: Allocator) !Self {
            const t = try init(shape, allocator);
            @memset(t.data, scalar);
            return t;
        }

        /// data must be initialized with the passed in allocator
        pub fn fromOwnedData(data: []T, shape: []const usize, allocator: Allocator) !Self {
            const owned_shape = try allocator.dupe(usize, shape);

            var total_size: usize = 1;
            for (shape) |length| {
                total_size *= length;
            }
            if (total_size != data.len) {
                return TensorStructError.InvalidDimensions;
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

        pub fn getFlatIndex(self: Self, indices: []usize) !usize {
            if (indices.len != self.shape.items.len) {
                return TensorStructError.InvalidDimensions;
            }

            var current_stride: usize = 1;
            var flat_index: usize = 0;
            for (0..indices.len) |dim| {
                const backwards_dim = indices.len - dim - 1;
                const index = indices[backwards_dim];
                const length = self.shape.items[backwards_dim];
                if (index < 0 or index >= length) {
                    return TensorStructError.OutOfBounds;
                }
                flat_index += index * current_stride;
                current_stride *= length;
            }
            return flat_index;
        }

        pub fn get(self: Self, indices: []usize) !usize {
            return self.data[try getFlatIndex(self, indices)];
        }
    };
}

pub fn Matrix(comptime T: type) type {
    ensureNumericType(T);
    return struct {
        const Self = @This();

        pub const DataType = T;

        rows: usize,
        columns: usize,
        data: []T,
        allocator: Allocator,

        pub fn init(rows: usize, columns: usize, allocator: Allocator) !Self {
            return Self{
                .rows = rows,
                .columns = columns,
                .data = try allocator.alloc(T, rows * columns),
                .allocator = allocator,
            };
        }

        pub fn splat(rows: usize, columns: usize, scalar: T, allocator: Allocator) !Self {
            const m = try init(rows, columns, allocator);
            @memset(m.data, scalar);
            return m;
        }

        /// data must have been allocated by allocator
        pub fn fromOwnedData(data: []T, rows: usize, columns: usize, allocator: Allocator) !Self {
            if (data.len != rows * columns) {
                return TensorStructError.InvalidDimensions;
            }
            return Self{
                .rows = rows,
                .columns = columns,
                .data = data,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: Self) void {
            self.allocator.free(self.data);
        }

        pub fn get(self: Self, row: usize, column: usize) TensorStructError!T {
            if (row < 0 or row >= self.rows or column < 0 or column >= self.columns) {
                return TensorStructError.OutOfBounds;
            }

            return self.data[row * self.columns + column];
        }

        pub fn getRow(self: Self, row: usize) TensorStructError![]T {
            if (row < 0 or row >= self.rows) {
                return TensorStructError.OutOfBounds;
            }

            return self.data[row .. row + self.columns];
        }
    };
}

pub fn Vector(comptime T: type) type {
    ensureNumericType(T);
    return struct {
        const Self = @This();

        pub const DataType = T;

        data: []T,
        allocator: Allocator,

        pub fn init(length: usize, allocator: Allocator) !Self {
            return Self{
                .data = try allocator.alloc(T, length),
                .allocator = allocator,
            };
        }

        pub fn splat(length: usize, scalar: T, allocator: Allocator) !Self {
            const v = try init(length, allocator);
            @memset(v.data, scalar);
            return v;
        }

        pub fn fromOwnedData(data: []T, allocator: Allocator) !Self {
            return Self{
                .data = data,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: Self) void {
            self.allocator.free(self.data);
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
    const tensor = try Tensor(f32).splat(&.{ 2, 2, 2 }, 0, testing.allocator);
    defer tensor.deinit();

    try testing.expectEqual(0, tensor.data[0]);
}

test "basic matrix creation" {
    const matrix = try Matrix(f32).splat(2, 2, 0, testing.allocator);
    defer matrix.deinit();

    try testing.expectEqual(0, matrix.data[0]);
}

test "basic vector creation" {
    const vector = try Vector(f32).splat(2, 0, testing.allocator);
    defer vector.deinit();

    try testing.expectEqual(0, vector.data[0]);
}
