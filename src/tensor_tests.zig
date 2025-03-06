const std = @import("std");
const testing = std.testing;
const zit = @import("zit");
const TensorContext = zit.TensorContext;
const CpuBackend = zit.CpuBackend;
const SimdBackend = zit.SimdBackend;

test "matrix multiply" {
    const ctx = TensorContext{
        .backend = CpuBackend.backend(),
        .allocator = testing.allocator,
    };

    const m_1 = try ctx.matrixSplat(f32, 2, 3, 1);
    defer m_1.deinit();
    const m_2 = try ctx.matrixSplat(f32, 3, 4, 3);
    defer m_2.deinit();

    const result = try ctx.matrixMultiply(m_1, m_2);
    defer result.deinit();
}
