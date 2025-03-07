const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zit = b.addModule("zit", .{
        .root_source_file = b.path("src/zit.zig"),
        .target = target,
        .optimize = optimize,
    });

    const test_step = b.step("test", "Run all tests");

    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/zit.zig"),
        .target = target,
        .optimize = optimize,
    });

    const unit_test_run = b.addRunArtifact(unit_tests);
    const unit_test_step = b.step("unit_test", "Run unit tests");
    unit_test_step.dependOn(&unit_test_run.step);
    test_step.dependOn(unit_test_step);

    const integration_tests = b.addTest(.{
        .root_source_file = b.path("src/tensor_tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    integration_tests.root_module.addImport("zit", zit);

    const integration_test_run = b.addRunArtifact(integration_tests);
    const integration_test_step = b.step("integration_test", "Run integration tests");
    integration_test_step.dependOn(&integration_test_run.step);
    test_step.dependOn(integration_test_step);

    addExecutable(
        b,
        target,
        optimize,
        "performance_comparisons",
        false,
        zit,
    );

    addExecutable(
        b,
        target,
        optimize,
        "foo",
        false,
        null,
    );
}

fn addExecutable(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    comptime name: []const u8,
    should_install: bool,
    module: ?*std.Build.Module,
) void {
    const exe = b.addExecutable(.{
        .name = name,
        .root_source_file = b.path(std.fmt.comptimePrint("src/{s}.zig", .{name})),
        .target = target,
        .optimize = optimize,
    });

    if (should_install) {
        b.installArtifact(exe);
    }

    if (module) |mod| {
        exe.root_module.addImport("zit", mod);
    }

    const exe_step = b.step(name, std.fmt.comptimePrint("Run {s}", .{name}));
    exe_step.dependOn(&b.addRunArtifact(exe).step);
}
