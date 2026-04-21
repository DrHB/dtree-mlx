const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const build_llama_bootstrap = b.option(
        bool,
        "llama-bootstrap",
        "Build the temporary llama.cpp-backed bootstrap binary.",
    ) orelse false;

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

    if (build_llama_bootstrap) {
        const llama_prefix = b.option(
            []const u8,
            "llama-prefix",
            "Path to the installed llama.cpp prefix (defaults to a Homebrew install).",
        ) orelse "/opt/homebrew/opt/llama.cpp";

        const ggml_prefix = b.option(
            []const u8,
            "ggml-prefix",
            "Path to the installed ggml prefix (defaults to a Homebrew install).",
        ) orelse "/opt/homebrew/opt/ggml";

        const bootstrap = b.addExecutable(.{
            .name = "dtree-mlx-zig-bootstrap",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/main.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
                .link_libcpp = true,
            }),
        });

        const llama_include = b.pathJoin(&.{ llama_prefix, "include" });
        const llama_lib = b.pathJoin(&.{ llama_prefix, "lib" });
        const ggml_include = b.pathJoin(&.{ ggml_prefix, "include" });
        const ggml_lib = b.pathJoin(&.{ ggml_prefix, "lib" });

        bootstrap.root_module.addIncludePath(.{ .cwd_relative = llama_include });
        bootstrap.root_module.addIncludePath(.{ .cwd_relative = ggml_include });
        bootstrap.root_module.addLibraryPath(.{ .cwd_relative = llama_lib });
        bootstrap.root_module.addLibraryPath(.{ .cwd_relative = ggml_lib });
        bootstrap.root_module.addRPath(.{ .cwd_relative = llama_lib });
        bootstrap.root_module.addRPath(.{ .cwd_relative = ggml_lib });
        bootstrap.root_module.linkSystemLibrary("ggml", .{});
        bootstrap.root_module.linkSystemLibrary("ggml-base", .{});
        bootstrap.root_module.linkSystemLibrary("llama", .{});

        b.installArtifact(bootstrap);

        const run_bootstrap = b.addRunArtifact(bootstrap);
        if (b.args) |args| {
            run_bootstrap.addArgs(args);
        }
        const run_bootstrap_step = b.step(
            "run-bootstrap",
            "Run the temporary llama.cpp-backed bootstrap binary.",
        );
        run_bootstrap_step.dependOn(&run_bootstrap.step);
    }
}
