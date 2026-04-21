const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseFast,
    });
    const metal_options = b.addOptions();
    metal_options.addOption([]const u8, "project_shader_dir", b.pathFromRoot("src/shaders/metal"));

    const exe = b.addExecutable(.{
        .name = "dtree-mlx-zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/pure_main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    if (target.result.os.tag == .macos) {
        configureMetalModule(b, exe.root_module, metal_options);
    }
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
        configureMetalModule(b, metal_exe.root_module, metal_options);
        b.installArtifact(metal_exe);
        b.installDirectory(.{
            .source_dir = b.path("src/shaders/metal"),
            .install_dir = .bin,
            .install_subdir = "shaders/metal",
            .include_extensions = &.{".metal"},
        });

        const metal_run = b.addRunArtifact(metal_exe);
        if (b.args) |args| {
            metal_run.addArgs(args);
        }

        const metal_step = b.step("metal-bootstrap", "Run the native macOS Metal bootstrap.");
        metal_step.dependOn(&metal_run.step);
    }
}

fn configureMetalModule(
    b: *std.Build,
    module: *std.Build.Module,
    metal_options: *std.Build.Step.Options,
) void {
    module.addOptions("metal_build_options", metal_options);
    module.addIncludePath(b.path("src/metal"));
    module.addCSourceFile(.{
        .file = b.path("src/metal/shim.m"),
        .language = .objective_c,
    });
    module.linkFramework("Foundation", .{});
    module.linkFramework("Metal", .{});
    module.linkSystemLibrary("objc", .{});
    module.linkSystemLibrary("c", .{});
}
