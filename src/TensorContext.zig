const std = @import("std");
const Allocator = std.mem.Allocator;
const Backend = @import("backend.zig").Backend;
const zit = @import("zit.zig");
const Tensor = zit.Tensor;
const Matrix = zit.Matrix;
const Vector = zit.Vector;
const TensorStructError = zit.TensorStructError;
const TensorOpError = zit.TensorOpError;
const TensorError = zit.TensorError;
const utils = @import("backends/utils.zig");
const fn_types = @import("fn_types.zig");

backend: Backend,
allocator: Allocator,

pub fn tensorInit(self: Self, comptime T: type, shape: []const usize) TensorStructError!Tensor(T) {
    return Tensor(T).init(shape, self.allocator);
}

pub fn tensorSplat(self: Self, comptime T: type, shape: []const usize, scalar: T) TensorStructError!Tensor(T) {
    return Tensor(T).splat(shape, scalar, self.allocator);
}

pub fn tensorFromOwnedData(self: Self, data: anytype, shape: []const usize) TensorStructError!Tensor(std.meta.Child(@TypeOf(data))) {
    return Tensor(std.meta.Child(@TypeOf(data))).fromOwnedData(data, shape, self.allocator);
}

pub fn matrixInit(self: Self, comptime T: type, rows: usize, columns: usize) TensorStructError!Matrix(T) {
    return Matrix(T).init(rows, columns, self.allocator);
}

pub fn matrixSplat(self: Self, comptime T: type, rows: usize, columns: usize, scalar: T) TensorStructError!Matrix(T) {
    return Matrix(T).splat(rows, columns, scalar, self.allocator);
}

pub fn matrixFromOwnedData(self: Self, data: anytype, rows: usize, columns: usize) TensorStructError!Matrix(std.meta.Child(@TypeOf(data))) {
    return Matrix(std.meta.Child(@TypeOf(data))).fromOwnedData(data, rows, columns, self.allocator);
}

pub fn vectorInit(self: Self, comptime T: type, length: usize) TensorStructError!Vector(T) {
    return Vector(T).init(length, self.allocator);
}

pub fn vectorSplat(self: Self, comptime T: type, length: usize, scalar: T) TensorStructError!Vector(T) {
    return Vector(T).splat(length, scalar, self.allocator);
}

pub fn vectorFromOwnedData(self: Self, data: anytype, length: usize) TensorStructError!Vector(std.meta.Child(@TypeOf(data))) {
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

pub fn op(self: Self, a: anytype, b: @TypeOf(a), op_fn: fn_types.BinaryOpFn) TensorError!@TypeOf(a) {
    var result = try utils.createMatchingTensor(a, self.allocator);
    errdefer result.deinit();

    try self.backend.vtable.op(self.backend.ptr, a, b, &result, op_fn);
    return result;
}

pub fn opInPlace(self: Self, a: anytype, b: std.meta.Child(@TypeOf(a)), op_fn: fn_types.BinaryOpFn) TensorOpError!void {
    try self.backend.vtable.op(self.backend.ptr, a.*, b, a, op_fn);
}

pub fn opWithOut(self: Self, a: anytype, b: @TypeOf(a), out: *@TypeOf(a), op_fn: fn_types.BinaryOpFn) TensorOpError!void {
    try self.backend.vtable.op(self.backend.ptr, a, b, out, op_fn);
}

pub fn map(self: Self, a: anytype, map_fn: fn_types.MapFn) TensorError!@TypeOf(a) {
    var result = try utils.createMatchingTensor(a, self.allocator);
    errdefer result.deinit();

    try self.backend.vtable.map(self.backend.ptr, a, &result, map_fn);
    return result;
}

pub fn mapInPlace(self: Self, a: anytype, map_fn: fn_types.MapFn) TensorOpError!void {
    try self.backend.vtable.map(self.backend.ptr, a.*, a, map_fn);
}

pub fn mapWithOut(self: Self, a: anytype, out: *@TypeOf(a), map_fn: fn_types.MapFn) TensorOpError!void {
    try self.backend.vtable.map(self.backend.ptr, a, out, map_fn);
}

pub fn vectorDot(self: Self, a: anytype, b: @TypeOf(a)) TensorOpError!@TypeOf(a).DataType {
    var result: @TypeOf(a).DataType = undefined;
    try self.backend.vtable.vectorDot(self.backend.ptr, a, b, &result);
    return result;
}

pub fn vectorNorm(self: Self, v: anytype) TensorOpError!@TypeOf(v).DataType {
    var result: @TypeOf(v).DataType = undefined;
    try self.backend.vtable.vectorNorm(self.backend.ptr, v, &result);
    return result;
}

pub fn matrixVectorMultiply(self: Self, m: anytype, v: anytype) TensorError!@TypeOf(v) {
    var result = try Vector(@TypeOf(v).DataType).init(m.rows);
    errdefer result.deinit();
    try self.backend.vtable.matrixVectorMultiply(self.backend.ptr, m, v, &result);
    return result;
}

pub fn matrixVectorMultiplyWithOut(self: Self, m: anytype, v: anytype, out: *@TypeOf(v)) TensorOpError!void {
    try self.backend.vtable.matrixVectorMultiply(self.backend.ptr, m, v, out);
}

pub fn matrixMultiply(self: Self, a: anytype, b: @TypeOf(a)) TensorError!@TypeOf(a) {
    var result = try Matrix(@TypeOf(a).DataType).init(a.rows, b.columns, self.allocator);
    errdefer result.deinit();
    try self.backend.vtable.matrixMultiply(self.backend.ptr, a, b, &result);
    return result;
}

pub fn matrixMultiplyWithOut(self: Self, a: anytype, b: @TypeOf(a), out: *@TypeOf(a)) TensorOpError!void {
    try self.backend.vtable.matrixMultiply(self.backend.ptr, a, b, out);
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

fn exampleElementwiseOp(self: Self, a: anytype, b: @TypeOf(a)) TensorError!@TypeOf(a) {
    _ = self;
    _ = b;
    return a;
}

fn elementwiseOp(op_fn: fn_types.BinaryOpFn) @TypeOf(exampleElementwiseOp) {
    return struct {
        fn opFn(self: Self, a: anytype, b: @TypeOf(a)) TensorError!@TypeOf(a) {
            return op(self, a, b, op_fn);
        }
    }.opFn;
}

fn exampleElementwiseOpInPlace(self: Self, a: anytype, b: std.meta.Child(@TypeOf(a))) TensorOpError!void {
    _ = self;
    _ = b;
}

fn elementwiseOpInPlace(op_fn: fn_types.BinaryOpFn) @TypeOf(exampleElementwiseOpInPlace) {
    return struct {
        fn opInPlaceFn(self: Self, a: anytype, b: std.meta.Child(@TypeOf(a))) TensorOpError!void {
            try self.backend.vtable.op(self.backend.ptr, a.*, b, a, op_fn);
        }
    }.opInPlaceFn;
}

fn exampleElementwiseOpWithOut(self: Self, a: anytype, b: @TypeOf(a), result: *@TypeOf(a)) TensorOpError!void {
    _ = self;
    _ = b;
    _ = result;
}

fn elementwiseOpWithOut(op_fn: fn_types.BinaryOpFn) @TypeOf(exampleElementwiseOpWithOut) {
    return struct {
        fn opWithOut(self: Self, a: anytype, b: @TypeOf(a), result: *@TypeOf(a)) TensorOpError!void {
            self.backend.vtable.op(self.backend.ptr, a, b, result, op_fn);
        }
    }.opWithOut;
}

const Self = @This();
const CpuBackend = @import("backends/cpu/CpuBackend.zig");

test "tensor context" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();
    const cpu_ctx = Self{
        .backend = CpuBackend.backend,
        .allocator = alloc,
    };
    std.debug.print("{s}\n", .{@typeName(@TypeOf(cpu_ctx.allocator))});
}
