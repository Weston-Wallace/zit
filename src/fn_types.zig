const zit = @import("zit.zig");
const TensorContext = zit.TensorContext;
const TensorError = zit.TensorError;

fn exampleOpFn(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    return x + y;
}
pub const BinaryOpFn = @TypeOf(exampleOpFn);

fn exampleMapFn(x: anytype) @TypeOf(x) {
    return x + 5;
}
pub const MapFn = @TypeOf(exampleMapFn);

fn exampleElementwiseOp(ctx: TensorContext, a: anytype, b: @TypeOf(a)) TensorError!@TypeOf(a) {
    _ = ctx;
    _ = b;
    return a;
}
pub const ElementwiseOpFn = @TypeOf(exampleElementwiseOp);
