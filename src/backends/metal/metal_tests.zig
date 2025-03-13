const std = @import("std");
const testing = std.testing;
const zit = @import("../../zit.zig");
const TensorContext = zit.TensorContext;
const Metal = zit.Metal;
const MetalBackend = zit.MetalBackend;

test "metal backend availability" {
    const is_available = Metal.isAvailable();

    // This test is more informational than a strict pass/fail
    std.debug.print("\nMetal backend is available: {}\n", .{is_available});

    // Skip other tests if Metal is not available
    if (!is_available) {
        return error.SkipZigTest;
    }
}

test "metal matrix operations" {
    // Skip on platforms where Metal isn't supported
    if (!Metal.isAvailable()) {
        return error.SkipZigTest;
    }

    // Initialize Metal
    try Metal.init(testing.allocator);
    defer Metal.deinit();

    const ctx = TensorContext(MetalBackend.backend){
        .allocator = testing.allocator,
    };

    // Matrix multiplication test
    {
        const m1 = try zit.Matrix(f32).init(2, 3, testing.allocator);
        defer m1.deinit();
        m1.data[0] = 1.0;
        m1.data[1] = 2.0;
        m1.data[2] = 3.0;
        m1.data[3] = 4.0;
        m1.data[4] = 5.0;
        m1.data[5] = 6.0;

        const m2 = try zit.Matrix(f32).init(3, 2, testing.allocator);
        defer m2.deinit();
        m2.data[0] = 7.0;
        m2.data[1] = 8.0;
        m2.data[2] = 9.0;
        m2.data[3] = 10.0;
        m2.data[4] = 11.0;
        m2.data[5] = 12.0;

        var result = try zit.Matrix(f32).init(2, 2, testing.allocator);
        defer result.deinit();

        try ctx.matrixMultiplyWithOut(m1, m2, &result);

        // Expected: [58, 64, 139, 154]
        try testing.expectApproxEqAbs(@as(f32, 58.0), result.data[0], 0.001);
        try testing.expectApproxEqAbs(@as(f32, 64.0), result.data[1], 0.001);
        try testing.expectApproxEqAbs(@as(f32, 139.0), result.data[2], 0.001);
        try testing.expectApproxEqAbs(@as(f32, 154.0), result.data[3], 0.001);
    }

    // Matrix transpose test
    {
        const m = try zit.Matrix(f32).init(2, 3, testing.allocator);
        defer m.deinit();
        m.data[0] = 1.0;
        m.data[1] = 2.0;
        m.data[2] = 3.0;
        m.data[3] = 4.0;
        m.data[4] = 5.0;
        m.data[5] = 6.0;

        var result = try zit.Matrix(f32).init(3, 2, testing.allocator);
        defer result.deinit();

        try ctx.matrixTransposeWithOut(m, &result);

        // Expected: [1, 4, 2, 5, 3, 6]
        try testing.expectApproxEqAbs(@as(f32, 1.0), result.data[0], 0.001);
        try testing.expectApproxEqAbs(@as(f32, 4.0), result.data[1], 0.001);
        try testing.expectApproxEqAbs(@as(f32, 2.0), result.data[2], 0.001);
        try testing.expectApproxEqAbs(@as(f32, 5.0), result.data[3], 0.001);
        try testing.expectApproxEqAbs(@as(f32, 3.0), result.data[4], 0.001);
        try testing.expectApproxEqAbs(@as(f32, 6.0), result.data[5], 0.001);
    }
}

test "metal vector operations" {
    // Skip on platforms where Metal isn't supported
    if (!Metal.isAvailable()) {
        return error.SkipZigTest;
    }

    // Initialize Metal
    try Metal.init(testing.allocator);
    defer Metal.deinit();

    const ctx = TensorContext(MetalBackend.backend){
        .allocator = testing.allocator,
    };

    // Vector dot product test
    {
        const v1 = try zit.Vector(f32).init(3, testing.allocator);
        defer v1.deinit();
        v1.data[0] = 1.0;
        v1.data[1] = 2.0;
        v1.data[2] = 3.0;

        const v2 = try zit.Vector(f32).init(3, testing.allocator);
        defer v2.deinit();
        v2.data[0] = 4.0;
        v2.data[1] = 5.0;
        v2.data[2] = 6.0;

        const result = try ctx.vectorDot(v1, v2);

        // Expected: 1*4 + 2*5 + 3*6 = 32
        try testing.expectApproxEqAbs(@as(f32, 32.0), result, 0.001);
    }

    // Vector norm test
    {
        const v = try zit.Vector(f32).init(3, testing.allocator);
        defer v.deinit();
        v.data[0] = 3.0;
        v.data[1] = 4.0;
        v.data[2] = 0.0;

        const result = try ctx.vectorNorm(v);

        // Expected: sqrt(3^2 + 4^2 + 0^2) = 5
        try testing.expectApproxEqAbs(@as(f32, 5.0), result, 0.001);
    }
}

test "metal elementwise operations" {
    // Skip on platforms where Metal isn't supported
    if (!Metal.isAvailable()) {
        return error.SkipZigTest;
    }

    // Initialize Metal
    try Metal.init(testing.allocator);
    defer Metal.deinit();

    const ctx = TensorContext(MetalBackend.backend){
        .allocator = testing.allocator,
    };

    // Addition test
    {
        const t1 = try zit.Tensor(f32).splat(&.{ 2, 3 }, 3.0, testing.allocator);
        defer t1.deinit();

        const t2 = try zit.Tensor(f32).splat(&.{ 2, 3 }, 5.0, testing.allocator);
        defer t2.deinit();

        var result = try zit.Tensor(f32).init(&.{ 2, 3 }, testing.allocator);
        defer result.deinit();

        try ctx.addWithOut(t1, t2, &result);

        // Expected: 3.0 + 5.0 = 8.0
        try testing.expectApproxEqAbs(@as(f32, 8.0), result.data[0], 0.001);
    }

    // Scalar multiply test
    {
        const t = try zit.Tensor(f32).splat(&.{ 2, 3 }, 3.0, testing.allocator);
        defer t.deinit();

        var result = try zit.Tensor(f32).init(&.{ 2, 3 }, testing.allocator);
        defer result.deinit();

        try ctx.scalarMultiplyWithOut(t, 2.0, &result);

        // Expected: 3.0 * 2.0 = 6.0
        try testing.expectApproxEqAbs(@as(f32, 6.0), result.data[0], 0.001);
    }
}
