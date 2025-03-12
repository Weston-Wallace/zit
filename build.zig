const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zit = b.addModule("zit", .{
        .root_source_file = b.path("src/zit.zig"),
        .target = target,
        .optimize = optimize,
    });

    const zbench = b.dependency("zbench", .{
        .target = target,
        .optimize = optimize,
    }).module("zbench");

    const test_step = b.step("test", "Run all tests");

    const unit_tests = b.addTest(.{
        .name = "unit_tests",
        .root_module = zit,
    });

    const unit_test_run = b.addRunArtifact(unit_tests);
    const unit_test_step = b.step("unit_test", "Run unit tests");
    unit_test_step.dependOn(&unit_test_run.step);
    test_step.dependOn(unit_test_step);

    const integration_tests = b.addTest(.{
        .name = "integration_tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tensor_tests.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{
                .name = "zit",
                .module = zit,
            }},
        }),
    });

    const integration_test_run = b.addRunArtifact(integration_tests);
    const integration_test_step = b.step("integration_test", "Run integration tests");
    integration_test_step.dependOn(&integration_test_run.step);
    test_step.dependOn(integration_test_step);

    const performance_comparisons_mod = b.createModule(.{
        .root_source_file = b.path("src/performance_comparisons.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{
                .name = "zit",
                .module = zit,
            },
            .{
                .name = "zbench",
                .module = zbench,
            },
        },
    });
    const performance_comparisons_exe = b.addExecutable(.{
        .name = "performance_comparisons",
        .root_module = performance_comparisons_mod,
    });

    b.step("performance_comparisons", "Run performance comparisons").dependOn(&b.addRunArtifact(performance_comparisons_exe).step);
}
