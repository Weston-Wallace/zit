const std = @import("std");
const Allocator = std.mem.Allocator;
const Backend = @import("backend.zig").Backend;
const zit = @import("zit.zig");
const Tensor = zit.Tensor;
const Matrix = zit.Matrix;
const Vector = zit.Vector;
const TensorError = zit.TensorError;
const utils = @import("backends/utils.zig");
const fn_types = @import("fn_types.zig");

pub fn TensorContext(comptime backend: Backend) type {
    return struct {
        const Self = @This();
        allocator: Allocator,

        pub fn tensorInit(self: Self, comptime T: type, shape: []const usize) !Tensor(T) {
            return Tensor(T).init(shape, self.allocator);
        }

        pub fn tensorSplat(self: Self, comptime T: type, shape: []const usize, scalar: T) !Tensor(T) {
            return Tensor(T).splat(shape, scalar, self.allocator);
        }

        pub fn tensorFromOwnedData(self: Self, data: anytype, shape: []const usize) !Tensor(std.meta.Child(@TypeOf(data))) {
            return Tensor(std.meta.Child(@TypeOf(data))).fromOwnedData(data, shape, self.allocator);
        }

        pub fn matrixInit(self: Self, comptime T: type, rows: usize, columns: usize) !Matrix(T) {
            return Matrix(T).init(rows, columns, self.allocator);
        }

        pub fn matrixSplat(self: Self, comptime T: type, rows: usize, columns: usize, scalar: T) !Matrix(T) {
            return Matrix(T).splat(rows, columns, scalar, self.allocator);
        }

        pub fn matrixFromOwnedData(self: Self, data: anytype, rows: usize, columns: usize) !Matrix(std.meta.Child(@TypeOf(data))) {
            return Matrix(std.meta.Child(@TypeOf(data))).fromOwnedData(data, rows, columns, self.allocator);
        }

        pub fn vectorInit(self: Self, comptime T: type, length: usize) !Vector(T) {
            return Vector(T).init(length, self.allocator);
        }

        pub fn vectorSplat(self: Self, comptime T: type, length: usize, scalar: T) !Vector(T) {
            return Vector(T).splat(length, scalar, self.allocator);
        }

        pub fn vectorFromOwnedData(self: Self, data: anytype, length: usize) !Vector(std.meta.Child(@TypeOf(data))) {
            return Vector(std.meta.Child(@TypeOf(data))).fromOwnedData(data, length, self.allocator);
        }

        pub const add = elementwiseOp(addOp);
        pub const addInPlace = elementwiseOpInPlace(addOp);
        pub const addWithOut = elementwiseOpWithOut(addOp);

        pub const subtract = elementwiseOp(subtractOp);
        pub const subtractInPlace = elementwiseOpInPlace(subtractOp);
        pub const subtractWithOut = elementwiseOpWithOut(subtractOp);

        pub const multiply = elementwiseOp(multiplyOp);
        pub const multiplyInPlace = elementwiseOpInPlace(multiplyOp);
        pub const multiplyWithOut = elementwiseOpWithOut(multiplyOp);

        pub const divide = elementwiseOp(divideOp);
        pub const divideInPlace = elementwiseOpInPlace(divideOp);
        pub const divideWithOut = elementwiseOpWithOut(divideOp);

        pub fn op(self: Self, a: anytype, b: @TypeOf(a), op_fn: fn_types.BinaryOpFn) !@TypeOf(a) {
            var result = try utils.createMatchingTensor(a, self.allocator);
            errdefer result.deinit();

            try backend.vtable.op(backend.ptr, a, b, &result, op_fn);
            return result;
        }

        pub fn opInPlace(_: Self, a: anytype, b: std.meta.Child(@TypeOf(a)), op_fn: fn_types.BinaryOpFn) !void {
            try backend.vtable.op(backend.ptr, a.*, b, a, op_fn);
        }

        pub fn opWithOut(_: Self, a: anytype, b: @TypeOf(a), out: *@TypeOf(a), op_fn: fn_types.BinaryOpFn) !void {
            try backend.vtable.op(backend.ptr, a, b, out, op_fn);
        }

        pub fn map(self: Self, a: anytype, map_fn: fn_types.MapFn) !@TypeOf(a) {
            var result = try utils.createMatchingTensor(a, self.allocator);
            errdefer result.deinit();

            try backend.vtable.map(backend.ptr, a, &result, map_fn);
            return result;
        }

        pub fn mapInPlace(_: Self, a: anytype, map_fn: fn_types.MapFn) !void {
            try backend.vtable.map(backend.ptr, a.*, a, map_fn);
        }

        pub fn mapWithOut(_: Self, a: anytype, out: *@TypeOf(a), map_fn: fn_types.MapFn) !void {
            try backend.vtable.map(backend.ptr, a, out, map_fn);
        }

        pub fn scalarMultiply(self: Self, a: anytype, scalar: @TypeOf(a).DataType) !void {
            var result = try utils.createMatchingTensor(a, self.allocator);
            errdefer result.deinit();

            try backend.vtable.scalarMultiply(backend.ptr, a, scalar, result);
        }

        pub fn scalarMultiplyInPlace(_: Self, a: anytype, scalar: std.meta.Child(@TypeOf(a)).DataType) !void {
            try backend.vtable.scalarMultiply(backend.ptr, a.*, scalar, a);
        }

        pub fn scalarMultiplyWithOut(_: Self, a: anytype, scalar: @TypeOf(a).DataType, out: *@TypeOf(a)) !void {
            try backend.vtable.scalarMultiply(backend.ptr, a, scalar, out);
        }

        pub fn vectorDot(_: Self, a: anytype, b: @TypeOf(a)) !@TypeOf(a).DataType {
            var result: @TypeOf(a).DataType = undefined;
            try backend.vtable.vectorDot(backend.ptr, a, b, &result);
            return result;
        }

        pub fn vectorNorm(_: Self, v: anytype) !@TypeOf(v).DataType {
            var result: @TypeOf(v).DataType = undefined;
            try backend.vtable.vectorNorm(backend.ptr, v, &result);
            return result;
        }

        pub fn matrixVectorMultiply(_: Self, m: anytype, v: anytype) !@TypeOf(v) {
            var result = try Vector(@TypeOf(v).DataType).init(m.rows);
            errdefer result.deinit();
            try backend.vtable.matrixVectorMultiply(backend.ptr, m, v, &result);
            return result;
        }

        pub fn matrixVectorMultiplyWithOut(_: Self, m: anytype, v: anytype, out: *@TypeOf(v)) !void {
            try backend.vtable.matrixVectorMultiply(backend.ptr, m, v, out);
        }

        pub fn matrixMultiply(self: Self, a: anytype, b: @TypeOf(a)) !@TypeOf(a) {
            var result = try Matrix(@TypeOf(a).DataType).init(a.rows, b.columns, self.allocator);
            errdefer result.deinit();
            try backend.vtable.matrixMultiply(backend.ptr, a, b, &result);
            return result;
        }

        pub fn matrixMultiplyWithOut(_: Self, a: anytype, b: @TypeOf(a), out: *@TypeOf(a)) !void {
            try backend.vtable.matrixMultiply(backend.ptr, a, b, out);
        }

        pub fn matrixTranspose(self: Self, a: anytype) !void {
            var result = try utils.createMatchingTensor(a, self.allocator);
            errdefer result.deinit();

            try backend.vtable.matrixTranspose(backend.ptr, a, &result);
            return result;
        }

        pub fn matrixTransposeWithOut(_: Self, a: anytype, out: *@TypeOf(a)) !void {
            try backend.vtable.matrixTranspose(backend.ptr, a, out);
        }

        fn addOp(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
            return x + y;
        }

        fn subtractOp(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
            return x - y;
        }

        fn multiplyOp(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
            return x * y;
        }

        fn divideOp(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
            return x / y;
        }

        fn exampleElementwiseOp(self: Self, a: anytype, b: @TypeOf(a)) anyerror!@TypeOf(a) {
            _ = self;
            _ = b;
            return a;
        }

        fn elementwiseOp(op_fn: fn_types.BinaryOpFn) @TypeOf(exampleElementwiseOp) {
            return struct {
                fn opFn(self: Self, a: anytype, b: @TypeOf(a)) !@TypeOf(a) {
                    return op(self, a, b, op_fn);
                }
            }.opFn;
        }

        fn exampleElementwiseOpInPlace(self: Self, a: anytype, b: std.meta.Child(@TypeOf(a))) anyerror!void {
            _ = self;
            _ = b;
        }

        fn elementwiseOpInPlace(op_fn: fn_types.BinaryOpFn) @TypeOf(exampleElementwiseOpInPlace) {
            return struct {
                fn opInPlaceFn(_: Self, a: anytype, b: std.meta.Child(@TypeOf(a))) !void {
                    try backend.vtable.op(backend.ptr, a.*, b, a, op_fn);
                }
            }.opInPlaceFn;
        }

        fn exampleElementwiseOpWithOut(self: Self, a: anytype, b: @TypeOf(a), result: *@TypeOf(a)) anyerror!void {
            _ = self;
            _ = b;
            _ = result;
        }

        fn elementwiseOpWithOut(op_fn: fn_types.BinaryOpFn) @TypeOf(exampleElementwiseOpWithOut) {
            return struct {
                fn opWithOut(_: Self, a: anytype, b: @TypeOf(a), result: *@TypeOf(a)) !void {
                    try backend.vtable.op(backend.ptr, a, b, result, op_fn);
                }
            }.opWithOut;
        }
    };
}

const CpuBackend = @import("backends/cpu/CpuBackend.zig");

test "tensor context" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    _ = TensorContext(CpuBackend.backend){ .allocator = alloc };
}
