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
}
