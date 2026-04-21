const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseFast,
    });

    const exe = b.addExecutable(.{
        .name = "dtree-mlx-zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/pure_main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the pure-Zig GGUF inspector.");
    run_step.dependOn(&run_cmd.step);

    if (target.result.os.tag == .macos) {
        const metal_exe = b.addExecutable(.{
            .name = "dtree-mlx-metal-bootstrap",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/metal.zig"),
                .target = target,
                .optimize = optimize,
            }),
        });
        b.installArtifact(metal_exe);

        const metal_run = b.addRunArtifact(metal_exe);
        if (b.args) |args| {
            metal_run.addArgs(args);
        }

        const metal_step = b.step("metal-bootstrap", "Run the native macOS Metal bootstrap.");
        metal_step.dependOn(&metal_run.step);
    }
}
