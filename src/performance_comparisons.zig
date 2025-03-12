const std = @import("std");
const zit = @import("zit");
const zbench = @import("zbench");
const TensorContext = zit.TensorContext;
const CpuBackend = zit.CpuBackend;
const SimdBackend = zit.SimdBackend;
const Matrix = zit.Matrix;

/// BenchmarkResult contains the performance metrics from a benchmark run
pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: u64,
    total_time_ns: u64,
    avg_time_ns: u64,
    min_time_ns: u64,
    max_time_ns: u64,

    pub fn format(
        self: BenchmarkResult,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "{s}: {d} iterations in {d:.2}ms (avg: {d:.2}µs, min: {d:.2}µs, max: {d:.2}µs)",
            .{
                self.name,
                self.iterations,
                @as(f64, @floatFromInt(self.total_time_ns)) / 1_000_000.0,
                @as(f64, @floatFromInt(self.avg_time_ns)) / 1_000.0,
                @as(f64, @floatFromInt(self.min_time_ns)) / 1_000.0,
                @as(f64, @floatFromInt(self.max_time_ns)) / 1_000.0,
            },
        );
    }

    pub fn compareWith(self: BenchmarkResult, other: BenchmarkResult) f64 {
        return @as(f64, @floatFromInt(other.avg_time_ns)) / @as(f64, @floatFromInt(self.avg_time_ns));
    }
};

/// Configuration for running benchmarks
pub const BenchmarkConfig = struct {
    warmup_iterations: u64 = 3,
    iterations: u64 = 10,
    print_results: bool = true,
};

/// Function to benchmark a given function with the specified configuration
pub fn benchmark(
    comptime Function: type,
    name: []const u8,
    func: Function,
    args: anytype,
    config: BenchmarkConfig,
) !BenchmarkResult {
    // Validate that Function is a function type
    comptime {
        if (@typeInfo(Function) != .@"fn") {
            @compileError("Expected function type, got " ++ @typeName(Function));
        }
    }

    var timer = try std.time.Timer.start();

    // Warmup runs to eliminate cache effects
    for (0..config.warmup_iterations) |_| {
        std.mem.doNotOptimizeAway(@call(.auto, func, args));
    }

    // Benchmarking runs
    var times = std.ArrayList(u64).init(std.heap.page_allocator);
    defer times.deinit();

    for (0..config.iterations) |_| {
        timer.reset();
        std.mem.doNotOptimizeAway(@call(.auto, func, args));
        const elapsed = timer.read();
        try times.append(elapsed);
    }

    // Calculate statistics
    var total_time: u64 = 0;
    var min_time: u64 = std.math.maxInt(u64);
    var max_time: u64 = 0;

    for (times.items) |time| {
        total_time += time;
        min_time = @min(min_time, time);
        max_time = @max(max_time, time);
    }

    const avg_time = total_time / config.iterations;

    const result = BenchmarkResult{
        .name = name,
        .iterations = config.iterations,
        .total_time_ns = total_time,
        .avg_time_ns = avg_time,
        .min_time_ns = min_time,
        .max_time_ns = max_time,
    };

    if (config.print_results) {
        std.debug.print("{}\n", .{result});
    }

    return result;
}

/// Compare two functions and report the performance difference
pub fn compareFunctions(
    comptime FuncA: type,
    comptime FuncB: type,
    name_a: []const u8,
    name_b: []const u8,
    func_a: FuncA,
    func_b: FuncB,
    func_a_args: anytype,
    func_b_args: anytype,
    config: BenchmarkConfig,
) !void {
    std.debug.print("Running benchmark comparison...\n", .{});

    const result_a = try benchmark(
        FuncA,
        name_a,
        func_a,
        func_a_args,
        config,
    );
    const result_b = try benchmark(
        FuncB,
        name_b,
        func_b,
        func_b_args,
        config,
    );

    const ratio = result_a.compareWith(result_b);

    std.debug.print("\nComparison:\n", .{});
    if (ratio > 1.0) {
        std.debug.print("{s} is {d:.2}x faster than {s}\n", .{ name_a, ratio, name_b });
    } else if (ratio < 1.0) {
        std.debug.print("{s} is {d:.2}x faster than {s}\n", .{ name_b, 1.0 / ratio, name_a });
    } else {
        std.debug.print("{s} and {s} have identical performance\n", .{ name_a, name_b });
    }
}

var gpa = std.heap.DebugAllocator(.{}).init;
const allocator = gpa.allocator();

var benchmark_data: BenchmarkData = undefined;

fn beforeAll() void {
    benchmark_data = .{
        .m1 = Matrix(f32).splat(100, 200, 7, allocator) catch unreachable,
        .m2 = Matrix(f32).splat(200, 300, 7, allocator) catch unreachable,
        .result = Matrix(f32).init(100, 300, allocator) catch unreachable,
    };
}

fn afterAll() void {
    benchmark_data.m1.deinit();
    benchmark_data.m2.deinit();
    benchmark_data.result.deinit();
}

const BenchmarkData = struct {
    m1: Matrix(f32),
    m2: Matrix(f32),
    result: Matrix(f32),
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

    try bench.addParam("Cpu", &MatMulBench(CpuBackend.backend).init, .{});
    try bench.addParam("Simd", &MatMulBench(SimdBackend.backend).init, .{});

    try stdout.writeAll("testing\n");
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
