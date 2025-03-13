const std = @import("std");
const zit = @import("zit");
const zbench = @import("zbench");
const TensorContext = zit.TensorContext;
const CpuBackend = zit.CpuBackend;
const SimdBackend = zit.SimdBackend;
const MetalBackend = zit.MetalBackend;
const Metal = zit.Metal;
const Matrix = zit.Matrix;
const Vector = zit.Vector;

var gpa = std.heap.DebugAllocator(.{}).init;
const allocator = gpa.allocator();

var benchmark_data: BenchmarkData = undefined;

const DataType = f32;

fn beforeAll() void {
    benchmark_data = .{
        .m1 = Matrix(DataType).splat(500, 1000, 7, allocator) catch unreachable,
        .m2 = Matrix(DataType).splat(1000, 2000, 7, allocator) catch unreachable,
        .result = Matrix(DataType).init(500, 2000, allocator) catch unreachable,
        .add_m = Matrix(DataType).splat(1000, 1000, 7, allocator) catch unreachable,
        .add_result = Matrix(DataType).init(1000, 1000, allocator) catch unreachable,
        .v1 = Vector(DataType).splat(1000, 7, allocator) catch unreachable,
        .v_result = Vector(DataType).init(1000, allocator) catch unreachable,
        .v2 = Vector(DataType).init(1_000_000, allocator) catch unreachable,
        .v3 = Vector(DataType).splat(100_000_000, 7, allocator) catch unreachable,
        .v3_result = Vector(DataType).init(100_000_000, allocator) catch unreachable,
    };

    // Initialize Metal backend if available
    if (Metal.isAvailable()) {
        Metal.init(allocator) catch {};
    }
}

fn afterAll() void {
    benchmark_data.m1.deinit();
    benchmark_data.m2.deinit();
    benchmark_data.result.deinit();
    benchmark_data.add_m.deinit();
    benchmark_data.add_result.deinit();
    benchmark_data.v1.deinit();
    benchmark_data.v_result.deinit();
    benchmark_data.v2.deinit();
    benchmark_data.v3.deinit();
    benchmark_data.v3_result.deinit();

    // Clean up Metal resources
    Metal.deinit();
}

const BenchmarkData = struct {
    m1: Matrix(DataType),
    m2: Matrix(DataType),
    result: Matrix(DataType),
    add_m: Matrix(DataType),
    add_result: Matrix(DataType),
    v1: Vector(DataType),
    v_result: Vector(DataType),
    v2: Vector(DataType),
    v3: Vector(DataType),
    v3_result: Vector(DataType),
};

pub fn MatMulBench(backend: zit.Backend) type {
    return struct {
        const Self = @This();
        const ctx = TensorContext(backend){
            .allocator = allocator,
        };

        pub const init = Self{};

        pub fn run(_: Self, _: std.mem.Allocator) void {
            std.mem.doNotOptimizeAway(ctx.matrixMultiplyWithOut(benchmark_data.m1, benchmark_data.m2, &benchmark_data.result));
        }
    };
}

pub fn AddBench(backend: zit.Backend) type {
    return struct {
        const Self = @This();
        const ctx = TensorContext(backend){
            .allocator = allocator,
        };

        pub const init = Self{};

        pub fn run(_: Self, _: std.mem.Allocator) void {
            std.mem.doNotOptimizeAway(ctx.addWithOut(benchmark_data.add_m, benchmark_data.add_m, &benchmark_data.add_result));
        }
    };
}

pub fn ScalarMultiplyBench(backend: zit.Backend) type {
    return struct {
        const Self = @This();
        const ctx = TensorContext(backend){
            .allocator = allocator,
        };

        pub const init = Self{};

        pub fn run(_: Self, _: std.mem.Allocator) void {
            std.mem.doNotOptimizeAway(ctx.scalarMultiplyWithOut(benchmark_data.v3, 7, &benchmark_data.v3_result));
        }
    };
}

pub fn TransposeBench(backend: zit.Backend) type {
    return struct {
        const Self = @This();
        const ctx = TensorContext(backend){
            .allocator = allocator,
        };

        pub const init = Self{};

        pub fn run(_: Self, _: std.mem.Allocator) void {
            std.mem.doNotOptimizeAway(ctx.matrixTransposeWithOut(benchmark_data.add_m, &benchmark_data.add_result));
        }
    };
}

pub fn MVMulBench(backend: zit.Backend) type {
    return struct {
        const Self = @This();
        const ctx = TensorContext(backend){
            .allocator = allocator,
        };

        pub const init = Self{};

        pub fn run(_: Self, _: std.mem.Allocator) void {
            std.mem.doNotOptimizeAway(ctx.matrixVectorMultiplyWithOut(benchmark_data.add_m, benchmark_data.v1, &benchmark_data.v_result));
        }
    };
}

pub fn DotBench(backend: zit.Backend) type {
    return struct {
        const Self = @This();
        const ctx = TensorContext(backend){
            .allocator = allocator,
        };

        pub const init = Self{};

        pub fn run(_: Self, _: std.mem.Allocator) void {
            std.mem.doNotOptimizeAway(ctx.vectorDot(benchmark_data.v2, benchmark_data.v2));
        }
    };
}

pub fn NormBench(backend: zit.Backend) type {
    return struct {
        const Self = @This();
        const ctx = TensorContext(backend){
            .allocator = allocator,
        };

        pub const init = Self{};

        pub fn run(_: Self, _: std.mem.Allocator) void {
            std.mem.doNotOptimizeAway(ctx.vectorNorm(benchmark_data.v2));
        }
    };
}

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    var bench = zbench.Benchmark.init(std.heap.page_allocator, .{
        .hooks = .{
            .before_all = beforeAll,
            .after_all = afterAll,
        },
    });
    defer bench.deinit();

    // try bench.addParam("Cpu matmul", &MatMulBench(CpuBackend.backend).init, .{});
    // try bench.addParam("Simd matmul", &MatMulBench(SimdBackend.backend).init, .{});
    // if (Metal.isAvailable()) try bench.addParam("Metal matmul", &MatMulBench(MetalBackend.backend).init, .{});

    // try bench.addParam("Cpu add", &AddBench(CpuBackend.backend).init, .{});
    // try bench.addParam("Simd add", &AddBench(SimdBackend.backend).init, .{});
    // if (Metal.isAvailable()) try bench.addParam("Metal add", &AddBench(MetalBackend.backend).init, .{});

    try bench.addParam("Cpu scalar multiply", &ScalarMultiplyBench(CpuBackend.backend).init, .{});
    try bench.addParam("Simd scalar multiply", &ScalarMultiplyBench(SimdBackend.backend).init, .{});
    if (Metal.isAvailable()) try bench.addParam("Metal scalar multiply", &ScalarMultiplyBench(MetalBackend.backend).init, .{});

    // try bench.addParam("Cpu transpose", &TransposeBench(CpuBackend.backend).init, .{});
    // try bench.addParam("Simd transpose", &TransposeBench(SimdBackend.backend).init, .{});
    // if (Metal.isAvailable()) try bench.addParam("Metal transpose", &TransposeBench(MetalBackend.backend).init, .{});

    // try bench.addParam("Cpu matrix vector multiply", &MVMulBench(CpuBackend.backend).init, .{});
    // try bench.addParam("Simd matrix vector multiply", &MVMulBench(SimdBackend.backend).init, .{});
    // if (Metal.isAvailable()) try bench.addParam("Metal matrix vector multiply", &MVMulBench(MetalBackend.backend).init, .{});

    // try bench.addParam("Cpu dot", &DotBench(CpuBackend.backend).init, .{});
    // try bench.addParam("Simd dot", &DotBench(SimdBackend.backend).init, .{});
    // if (Metal.isAvailable()) try bench.addParam("Metal dot", &DotBench(MetalBackend.backend).init, .{});

    // try bench.addParam("Cpu norm", &NormBench(CpuBackend.backend).init, .{});
    // try bench.addParam("Simd norm", &NormBench(SimdBackend.backend).init, .{});
    // if (Metal.isAvailable()) try bench.addParam("Metal norm", &NormBench(MetalBackend.backend).init, .{});

    try stdout.writeAll("\n");
    try bench.run(stdout);
}
