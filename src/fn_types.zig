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
