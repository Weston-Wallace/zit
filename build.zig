const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zit = b.addModule("zit", .{
        .root_source_file = b.path("src/zit.zig"),
        .target = target,
        .optimize = optimize,
    });

    const tests = b.addTest(.{
        .root_source_file = b.path("src/tensor_tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    tests.root_module.addImport("zit", zit);

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
}
