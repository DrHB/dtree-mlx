const std = @import("std");

const c = @cImport({
    @cInclude("llama.h");
});

const Args = struct {
    model: []const u8 = "models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf",
    prompt: []const u8 = "Write a terse summary of speculative decoding and explain why a plain baseline matters before adding draft verification.",
    max_new_tokens: usize = 128,
    ctx_size: u32 = 4096,
    gpu_layers: i32 = -1,
    threads: i32 = 8,
    temperature: f32 = 0.0,
    top_k: i32 = 40,
    top_p: f32 = 0.95,
    seed: u32 = c.LLAMA_DEFAULT_SEED,
    use_chat_template: bool = true,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var arg_it = init.minimal.args.iterate();
    const args = try parseArgs(allocator, &arg_it);
    defer freeArgs(allocator, args);

    c.llama_backend_init();
    defer c.llama_backend_free();
    c.ggml_backend_load_all();

    var model_params = c.llama_model_default_params();
    model_params.n_gpu_layers = args.gpu_layers;
    model_params.use_mmap = true;

    const model = c.llama_model_load_from_file(args.model.ptr, model_params) orelse {
        std.debug.print("failed to load model from {s}\n", .{args.model});
        return error.ModelLoadFailed;
    };
    defer c.llama_model_free(model);

    var ctx_params = c.llama_context_default_params();
    ctx_params.n_ctx = args.ctx_size;
    ctx_params.n_batch = args.ctx_size;
    ctx_params.n_ubatch = @min(args.ctx_size, @as(u32, 1024));
    ctx_params.n_threads = args.threads;
    ctx_params.n_threads_batch = args.threads;
    ctx_params.offload_kqv = true;

    const ctx = c.llama_init_from_model(model, ctx_params) orelse {
        std.debug.print("failed to create llama context\n", .{});
        return error.ContextInitFailed;
    };
    defer c.llama_free(ctx);

    const sampler = try initSampler(args, model);
    defer c.llama_sampler_free(sampler);

    const prompt_text = try renderPrompt(allocator, model, args.prompt, args.use_chat_template);
    defer allocator.free(prompt_text);

    const prompt_tokens = try tokenize(allocator, model, prompt_text, !args.use_chat_template, args.use_chat_template);
    defer allocator.free(prompt_tokens);

    if (prompt_tokens.len == 0) {
        return error.EmptyPrompt;
    }
    if (prompt_tokens.len + args.max_new_tokens > args.ctx_size) {
        std.debug.print(
            "prompt + generation length ({d}) exceeds ctx_size={d}\n",
            .{ prompt_tokens.len + args.max_new_tokens, args.ctx_size },
        );
        return error.ContextOverflow;
    }

    try decodeTokens(ctx, prompt_tokens);

    var generated: std.ArrayList(c.llama_token) = .empty;
    defer generated.deinit(allocator);

    const vocab = c.llama_model_get_vocab(model);
    const start_us = c.llama_time_us();
    var next = c.llama_sampler_sample(sampler, ctx, -1);

    while (generated.items.len < args.max_new_tokens) {
        if (c.llama_vocab_is_eog(vocab, next)) {
            break;
        }

        try generated.append(allocator, next);
        try decodeOne(ctx, next);
        next = c.llama_sampler_sample(sampler, ctx, -1);
    }

    const elapsed_us = c.llama_time_us() - start_us;
    const elapsed_s = @as(f64, @floatFromInt(elapsed_us)) / 1_000_000.0;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    if (generated.items.len > 0) {
        const output = try detokenize(allocator, model, generated.items);
        defer allocator.free(output);
        try stdout.interface.writeAll(output);
    }
    try stdout.interface.writeAll("\n");
    try stdout.interface.flush();

    try printSummary(model, prompt_tokens.len, generated.items.len, elapsed_s);
}

fn parseArgs(allocator: std.mem.Allocator, arg_it: *std.process.Args.Iterator) !Args {
    const defaults = Args{};
    var out = Args{
        .model = try allocator.dupe(u8, defaults.model),
        .prompt = try allocator.dupe(u8, defaults.prompt),
    };

    _ = arg_it.next();
    while (arg_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "--model")) {
            const value = arg_it.next() orelse return error.MissingValue;
            allocator.free(out.model);
            out.model = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            const value = arg_it.next() orelse return error.MissingValue;
            allocator.free(out.prompt);
            out.prompt = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, arg, "--max-new-tokens")) {
            const value = arg_it.next() orelse return error.MissingValue;
            out.max_new_tokens = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--ctx-size")) {
            const value = arg_it.next() orelse return error.MissingValue;
            out.ctx_size = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, arg, "--gpu-layers")) {
            const value = arg_it.next() orelse return error.MissingValue;
            out.gpu_layers = try std.fmt.parseInt(i32, value, 10);
        } else if (std.mem.eql(u8, arg, "--threads")) {
            const value = arg_it.next() orelse return error.MissingValue;
            out.threads = try std.fmt.parseInt(i32, value, 10);
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            const value = arg_it.next() orelse return error.MissingValue;
            out.temperature = try std.fmt.parseFloat(f32, value);
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            const value = arg_it.next() orelse return error.MissingValue;
            out.top_k = try std.fmt.parseInt(i32, value, 10);
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            const value = arg_it.next() orelse return error.MissingValue;
            out.top_p = try std.fmt.parseFloat(f32, value);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            const value = arg_it.next() orelse return error.MissingValue;
            out.seed = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, arg, "--raw-prompt")) {
            out.use_chat_template = false;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            printUsage();
            return error.UnknownArgument;
        }
    }

    return out;
}

fn freeArgs(allocator: std.mem.Allocator, args: Args) void {
    allocator.free(args.model);
    allocator.free(args.prompt);
}

fn initSampler(args: Args, model: *c.llama_model) !*c.llama_sampler {
    if (args.temperature <= 0.0) {
        return c.llama_sampler_init_greedy() orelse error.SamplerInitFailed;
    }

    const params = c.llama_sampler_chain_default_params();
    const sampler = c.llama_sampler_chain_init(params) orelse return error.SamplerInitFailed;
    errdefer c.llama_sampler_free(sampler);

    if (args.top_k > 0) {
        c.llama_sampler_chain_add(
            sampler,
            c.llama_sampler_init_top_k(args.top_k) orelse return error.SamplerInitFailed,
        );
    }
    if (args.top_p < 1.0) {
        c.llama_sampler_chain_add(
            sampler,
            c.llama_sampler_init_top_p(args.top_p, 1) orelse return error.SamplerInitFailed,
        );
    }
    c.llama_sampler_chain_add(
        sampler,
        c.llama_sampler_init_temp(args.temperature) orelse return error.SamplerInitFailed,
    );
    c.llama_sampler_chain_add(
        sampler,
        c.llama_sampler_init_dist(args.seed) orelse return error.SamplerInitFailed,
    );

    _ = model;
    return sampler;
}

fn renderPrompt(
    allocator: std.mem.Allocator,
    model: *c.llama_model,
    prompt: []const u8,
    use_chat_template: bool,
) ![]u8 {
    if (!use_chat_template) {
        return allocator.dupe(u8, prompt);
    }

    const template = c.llama_model_chat_template(model, null) orelse {
        return allocator.dupe(u8, prompt);
    };

    var buffer = try allocator.alloc(u8, @max(prompt.len * 4, 1024));
    errdefer allocator.free(buffer);

    const messages = [_]c.llama_chat_message{
        .{
            .role = "user",
            .content = prompt.ptr,
        },
    };

    var written = c.llama_chat_apply_template(
        template,
        &messages,
        messages.len,
        true,
        buffer.ptr,
        @intCast(buffer.len),
    );
    if (written < 0) {
        return allocator.dupe(u8, prompt);
    }
    if (@as(usize, @intCast(written)) >= buffer.len) {
        allocator.free(buffer);
        buffer = try allocator.alloc(u8, @as(usize, @intCast(written)) + 1);
        written = c.llama_chat_apply_template(
            template,
            &messages,
            messages.len,
            true,
            buffer.ptr,
            @intCast(buffer.len),
        );
        if (written < 0) {
            allocator.free(buffer);
            return allocator.dupe(u8, prompt);
        }
    }
    const rendered = try allocator.dupe(u8, buffer[0..@as(usize, @intCast(written))]);
    allocator.free(buffer);
    return rendered;
}

fn tokenize(
    allocator: std.mem.Allocator,
    model: *c.llama_model,
    text: []const u8,
    add_special: bool,
    parse_special: bool,
) ![]c.llama_token {
    const vocab = c.llama_model_get_vocab(model);
    var capacity = @max(text.len + 32, 256);
    var tokens = try allocator.alloc(c.llama_token, capacity);
    errdefer allocator.free(tokens);

    while (true) {
        const rc = c.llama_tokenize(
            vocab,
            text.ptr,
            @intCast(text.len),
            tokens.ptr,
            @intCast(tokens.len),
            add_special,
            parse_special,
        );
        if (rc >= 0) {
            return try allocator.realloc(tokens, @as(usize, @intCast(rc)));
        }

        capacity = @as(usize, @intCast(-rc));
        allocator.free(tokens);
        tokens = try allocator.alloc(c.llama_token, capacity);
    }
}

fn detokenize(
    allocator: std.mem.Allocator,
    model: *c.llama_model,
    tokens: []const c.llama_token,
) ![]u8 {
    const vocab = c.llama_model_get_vocab(model);
    var capacity = @max(tokens.len * 8, 256);
    var text = try allocator.alloc(u8, capacity);
    errdefer allocator.free(text);

    while (true) {
        const rc = c.llama_detokenize(
            vocab,
            tokens.ptr,
            @intCast(tokens.len),
            text.ptr,
            @intCast(text.len),
            true,
            true,
        );
        if (rc >= 0) {
            return try allocator.realloc(text, @as(usize, @intCast(rc)));
        }

        capacity = @as(usize, @intCast(-rc)) + 1;
        allocator.free(text);
        text = try allocator.alloc(u8, capacity);
    }
}

fn decodeTokens(ctx: *c.llama_context, tokens: []const c.llama_token) !void {
    const batch = c.llama_batch_get_one(@constCast(tokens.ptr), @intCast(tokens.len));
    const rc = c.llama_decode(ctx, batch);
    if (rc != 0) {
        std.debug.print("llama_decode(prompt) failed with rc={d}\n", .{rc});
        return error.DecodeFailed;
    }
}

fn decodeOne(ctx: *c.llama_context, token: c.llama_token) !void {
    var value = token;
    const batch = c.llama_batch_get_one(&value, 1);
    const rc = c.llama_decode(ctx, batch);
    if (rc != 0) {
        std.debug.print("llama_decode(token) failed with rc={d}\n", .{rc});
        return error.DecodeFailed;
    }
}

fn printSummary(
    model: *c.llama_model,
    prompt_tokens: usize,
    generated_tokens: usize,
    elapsed_s: f64,
) !void {
    var desc_buf: [256]u8 = undefined;
    const desc_len = c.llama_model_desc(model, &desc_buf, desc_buf.len);
    const desc = if (desc_len > 0)
        desc_buf[0..@as(usize, @intCast(desc_len))]
    else
        "unknown";

    const tok_s = if (elapsed_s > 0.0)
        @as(f64, @floatFromInt(generated_tokens)) / elapsed_s
    else
        0.0;

    std.debug.print(
        "[zig-baseline] model={s} layers={d} ctx_train={d} prompt_tokens={d} generated_tokens={d} tok_s={d:.2}\n",
        .{
            desc,
            c.llama_model_n_layer(model),
            c.llama_model_n_ctx_train(model),
            prompt_tokens,
            generated_tokens,
            tok_s,
        },
    );
}

fn printUsage() void {
    std.debug.print(
        \\Usage: dtree-mlx-zig [options]
        \\
        \\Options:
        \\  --model PATH              GGUF model path.
        \\  --prompt TEXT             User prompt to run.
        \\  --max-new-tokens N        Max tokens to generate. Default: 128
        \\  --ctx-size N              Context window to allocate. Default: 4096
        \\  --gpu-layers N            Layers to offload. -1 means all. Default: -1
        \\  --threads N               CPU threads for prompt and decode. Default: 8
        \\  --temperature F           Sampling temperature. 0 = greedy. Default: 0
        \\  --top-k N                 Top-k cutoff when temperature > 0. Default: 40
        \\  --top-p F                 Top-p cutoff when temperature > 0. Default: 0.95
        \\  --seed N                  RNG seed when temperature > 0.
        \\  --raw-prompt              Skip the model chat template and tokenize the prompt directly.
        \\  --help                    Print this help text.
        \\
        \\This binary is the plain-generation baseline for the Zig port. DFlash and DTree are future layers.
        \\
    , .{});
}
