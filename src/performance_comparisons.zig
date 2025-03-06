const std = @import("std");
const zit = @import("zit");
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
        if (@typeInfo(Function) != .Fn) {
            @compileError("Expected function type, got " ++ @typeName(Function));
        }
    }

    // Type check the function to ensure it's callable with no arguments
    const ReturnType = @typeInfo(Function).Fn.return_type orelse void;

    var timer = try std.time.Timer.start();

    // Warmup runs to eliminate JIT/cache effects
    for (0..config.warmup_iterations) |_| {
        const result = @call(.auto, func, args);
        // Prevent compiler from optimizing away the function call
        if (@typeInfo(ReturnType) != .Void) {
            // Add some volatile operation to prevent the compiler from
            // optimizing away the function call
            const volatile_ptr: *volatile ReturnType = @ptrFromInt(@intFromPtr(&result));
            _ = volatile_ptr.*;
        }
    }

    // Benchmarking runs
    var times = std.ArrayList(u64).init(std.heap.page_allocator);
    defer times.deinit();

    for (0..config.iterations) |_| {
        timer.reset();
        const result = @call(.auto, func, args);
        const elapsed = timer.read();

        // Prevent compiler from optimizing away the function call
        if (@typeInfo(ReturnType) != .Void) {
            const volatile_ptr: *volatile ReturnType = @ptrFromInt(@intFromPtr(&result));
            _ = volatile_ptr.*;
        }

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

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();
const cpu_ctx = TensorContext{
    .backend = CpuBackend.backend(),
    .allocator = allocator,
};
const simd_ctx = TensorContext{
    .backend = SimdBackend.backend(),
    .allocator = allocator,
};

pub fn main() !void {
    const m_1 = try cpu_ctx.matrixSplat(f32, 1000, 2000, 15);
    const m_2 = try cpu_ctx.matrixSplat(f32, 2000, 3000, 8);
    var result = try cpu_ctx.matrixInit(f32, 1000, 3000);

    const config = BenchmarkConfig{
        .warmup_iterations = 1000,
        .iterations = 100000,
        .print_results = true,
    };

    try compareFunctions(
        cpu_ctx.matrixMultiplyWithOut,
        simd_ctx.matrixMultiplyWithOut,
        "cpu matrix multiply",
        "simd matrix multiply",
        cpu_ctx.matrixMultiplyWithOut,
        simd_ctx.matrixMultiplyWithOut,
        .{ cpu_ctx, m_1, m_2, &result },
        .{ simd_ctx, m_1, m_2, &result },
        config,
    );
}
