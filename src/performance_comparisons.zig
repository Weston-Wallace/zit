const std = @import("std");
const zit = @import("zit");
const zbench = @import("zbench");
const TensorContext = zit.TensorContext;
const CpuBackend = zit.CpuBackend;
const SimdBackend = zit.SimdBackend;
const Matrix = zit.Matrix;

var gpa = std.heap.DebugAllocator(.{}).init;
const allocator = gpa.allocator();

var benchmark_data: BenchmarkData = undefined;

const DataType = f16;

fn beforeAll() void {
    benchmark_data = .{
        .m1 = Matrix(DataType).splat(100, 200, 7, allocator) catch unreachable,
        .m2 = Matrix(DataType).splat(200, 300, 7, allocator) catch unreachable,
        .result = Matrix(DataType).init(100, 300, allocator) catch unreachable,
        .add_m = Matrix(DataType).splat(1000, 1000, 7, allocator) catch unreachable,
        .add_result = Matrix(DataType).init(1000, 1000, allocator) catch unreachable,
    };
}

fn afterAll() void {
    benchmark_data.m1.deinit();
    benchmark_data.m2.deinit();
    benchmark_data.result.deinit();
    benchmark_data.add_m.deinit();
    benchmark_data.add_result.deinit();
}

const BenchmarkData = struct {
    m1: Matrix(DataType),
    m2: Matrix(DataType),
    result: Matrix(DataType),
    add_m: Matrix(DataType),
    add_result: Matrix(DataType),
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

pub fn main() !void {
    // var gpa = std.heap.DebugAllocator(.{}).init;
    // const allocator = gpa.allocator();
    const stdout = std.io.getStdOut().writer();
    var bench = zbench.Benchmark.init(std.heap.page_allocator, .{
        .hooks = .{
            .before_all = beforeAll,
            .after_all = afterAll,
        },
    });
    defer bench.deinit();

    try bench.addParam("Cpu matmul", &MatMulBench(CpuBackend.backend).init, .{});
    try bench.addParam("Simd matmul", &MatMulBench(SimdBackend.backend).init, .{});
    try bench.addParam("Cpu add", &AddBench(CpuBackend.backend).init, .{});
    try bench.addParam("Simd add", &AddBench(SimdBackend.backend).init, .{});

    try stdout.writeAll("\n");
    try bench.run(stdout);

    // try compareFunctions(
    //     @TypeOf(CpuCtx.matrixMultiplyWithOut),
    //     @TypeOf(SimdCtx.matrixMultiplyWithOut),
    //     "cpu matrix multiply",
    //     "simd matrix multiply",
    //     CpuCtx.matrixMultiplyWithOut,
    //     SimdCtx.matrixMultiplyWithOut,
    //     .{ cpu_ctx, m_1, m_2, &result },
    //     .{ simd_ctx, m_1, m_2, &result },
    //     config,
    // );
}
