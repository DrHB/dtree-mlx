const std = @import("std");
const gguf = @import("gguf.zig");
const gguf_store = @import("gguf_store.zig");
const metal_backend = @import("metal_backend.zig");
const ops = @import("ops.zig");
const parallel_rows = @import("parallel_rows.zig");

pub const OutputCandidate = struct {
    token_id: usize,
    logit: f32,
};

pub const RunResult = struct {
    argmax_token_id: usize,
    argmax_logit: f32,
    top_count: usize,
};

pub const BenchmarkResult = struct {
    elapsed_s: f64,
    passes_per_s: f64,
    checksum: f64,
};

pub const HParams = struct {
    n_layers: usize,
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    full_attention_interval: usize,
    n_expert: usize,
    n_expert_used: usize,
    expert_ffn_len: usize,
    expert_shared_ffn_len: usize,
    attention_key_len: usize,
    attention_value_len: usize,
    ssm_conv_kernel: usize,
    ssm_state_size: usize,
    ssm_group_count: usize,
    ssm_time_step_rank: usize,
    ssm_inner_size: usize,
    rms_epsilon: f32,

    pub fn load(model: *const gguf.File) !HParams {
        return .{
            .n_layers = try kvUsize(model, "qwen35moe.block_count"),
            .n_embd = try kvUsize(model, "qwen35moe.embedding_length"),
            .n_head = try kvUsize(model, "qwen35moe.attention.head_count"),
            .n_head_kv = try kvUsize(model, "qwen35moe.attention.head_count_kv"),
            .full_attention_interval = try kvUsize(model, "qwen35moe.full_attention_interval"),
            .n_expert = try kvUsize(model, "qwen35moe.expert_count"),
            .n_expert_used = try kvUsize(model, "qwen35moe.expert_used_count"),
            .expert_ffn_len = try kvUsize(model, "qwen35moe.expert_feed_forward_length"),
            .expert_shared_ffn_len = try kvUsize(model, "qwen35moe.expert_shared_feed_forward_length"),
            .attention_key_len = try kvUsize(model, "qwen35moe.attention.key_length"),
            .attention_value_len = try kvUsize(model, "qwen35moe.attention.value_length"),
            .ssm_conv_kernel = try kvUsize(model, "qwen35moe.ssm.conv_kernel"),
            .ssm_state_size = try kvUsize(model, "qwen35moe.ssm.state_size"),
            .ssm_group_count = try kvUsize(model, "qwen35moe.ssm.group_count"),
            .ssm_time_step_rank = try kvUsize(model, "qwen35moe.ssm.time_step_rank"),
            .ssm_inner_size = try kvUsize(model, "qwen35moe.ssm.inner_size"),
            .rms_epsilon = kvF32(model, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6),
        };
    }
};

const RecurrentWeights = struct {
    qkv: gguf_store.TensorView,
    gate: gguf_store.TensorView,
    beta: gguf_store.TensorView,
    conv: gguf_store.TensorView,
    ssm_norm: gguf_store.TensorView,
    out: gguf_store.TensorView,
};

const FullAttentionWeights = struct {
    q_gate: gguf_store.TensorView,
    v: gguf_store.TensorView,
    out: gguf_store.TensorView,
};

const FfnWeights = struct {
    post_norm: gguf_store.TensorView,
    gate_inp: gguf_store.TensorView,
    up_exps: gguf_store.TensorView,
    gate_exps: gguf_store.TensorView,
    down_exps: gguf_store.TensorView,
    gate_inp_shexp: gguf_store.TensorView,
    up_shexp: gguf_store.TensorView,
    gate_shexp: gguf_store.TensorView,
    down_shexp: gguf_store.TensorView,
};

const LayerKind = union(enum) {
    recurrent: RecurrentWeights,
    full: FullAttentionWeights,
};

const LayerWeights = struct {
    attn_norm: gguf_store.TensorView,
    kind: LayerKind,
    ffn: FfnWeights,
};

const Scratch = struct {
    hidden0: []f32,
    hidden1: []f32,
    hidden2: []f32,
    weight: []f32,
    wide0: []f32,
    wide1: []f32,
    wide2: []f32,
    expert0: []f32,
    expert1: []f32,
    router_logits: []f32,
    selected_logits: []f32,
    selected_indices: []usize,
    selected_weights: []f32,
    small32: []f32,
    small128: []f32,

    fn init(allocator: std.mem.Allocator, hparams: HParams) !Scratch {
        const max_wide = @max(
            hparams.ssm_inner_size + 2 * hparams.ssm_group_count * hparams.ssm_state_size,
            hparams.n_head * hparams.attention_value_len,
        );
        const max_expert = @max(hparams.expert_ffn_len, hparams.expert_shared_ffn_len);
        const max_small32 = @max(hparams.ssm_time_step_rank, hparams.n_head);

        return .{
            .hidden0 = try allocator.alloc(f32, hparams.n_embd),
            .hidden1 = try allocator.alloc(f32, hparams.n_embd),
            .hidden2 = try allocator.alloc(f32, hparams.n_embd),
            .weight = try allocator.alloc(f32, hparams.n_embd),
            .wide0 = try allocator.alloc(f32, max_wide),
            .wide1 = try allocator.alloc(f32, max_wide),
            .wide2 = try allocator.alloc(f32, max_wide),
            .expert0 = try allocator.alloc(f32, max_expert),
            .expert1 = try allocator.alloc(f32, max_expert),
            .router_logits = try allocator.alloc(f32, hparams.n_expert),
            .selected_logits = try allocator.alloc(f32, hparams.n_expert_used),
            .selected_indices = try allocator.alloc(usize, hparams.n_expert_used),
            .selected_weights = try allocator.alloc(f32, hparams.n_expert_used),
            .small32 = try allocator.alloc(f32, max_small32),
            .small128 = try allocator.alloc(f32, hparams.ssm_state_size),
        };
    }

    fn deinit(self: *Scratch, allocator: std.mem.Allocator) void {
        allocator.free(self.hidden0);
        allocator.free(self.hidden1);
        allocator.free(self.hidden2);
        allocator.free(self.weight);
        allocator.free(self.wide0);
        allocator.free(self.wide1);
        allocator.free(self.wide2);
        allocator.free(self.expert0);
        allocator.free(self.expert1);
        allocator.free(self.router_logits);
        allocator.free(self.selected_logits);
        allocator.free(self.selected_indices);
        allocator.free(self.selected_weights);
        allocator.free(self.small32);
        allocator.free(self.small128);
        self.* = undefined;
    }
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    store: *const gguf_store.Store,
    backend: ?*metal_backend.Backend,
    hparams: HParams,
    token_embd: gguf_store.TensorView,
    output_norm: gguf_store.TensorView,
    output: gguf_store.TensorView,
    layers: []LayerWeights,
    scratch: Scratch,
    output_logits: ?[]f32,

    pub fn init(
        allocator: std.mem.Allocator,
        store: *const gguf_store.Store,
        backend: ?*metal_backend.Backend,
    ) !Engine {
        const hparams = try HParams.load(&store.parsed);
        const layers = try allocator.alloc(LayerWeights, hparams.n_layers);
        errdefer allocator.free(layers);

        const token_embd = try store.tensor("token_embd.weight");
        const output_norm = try store.tensor("output_norm.weight");
        const output = try store.tensor("output.weight");
        const scratch = try Scratch.init(allocator, hparams);
        errdefer {
            var scratch_copy = scratch;
            scratch_copy.deinit(allocator);
        }
        const output_logits = if (backend != null) try allocator.alloc(f32, output.row_count) else null;
        errdefer if (output_logits) |logits| allocator.free(logits);

        var engine = Engine{
            .allocator = allocator,
            .store = store,
            .backend = backend,
            .hparams = hparams,
            .token_embd = token_embd,
            .output_norm = output_norm,
            .output = output,
            .layers = layers,
            .scratch = scratch,
            .output_logits = output_logits,
        };

        for (0..hparams.n_layers) |layer_idx| {
            engine.layers[layer_idx] = try engine.loadLayer(layer_idx);
        }

        return engine;
    }

    pub fn deinit(self: *Engine) void {
        if (self.output_logits) |logits| self.allocator.free(logits);
        self.scratch.deinit(self.allocator);
        self.allocator.free(self.layers);
        self.* = undefined;
    }

    pub fn run(
        self: *Engine,
        token_id: usize,
        top_out: []OutputCandidate,
    ) !RunResult {
        if (token_id >= self.token_embd.row_count) return error.TokenIdOutOfRange;

        var current = self.scratch.hidden0;
        try self.token_embd.dequantizeRow(token_id, current);

        for (self.layers) |layer| {
            try self.runLayer(layer, current);
            current = self.scratch.hidden0;
        }

        try loadWeightRow(self.output_norm, self.scratch.weight[0..self.hparams.n_embd]);
        try ops.rmsNorm(
            self.scratch.hidden1[0..self.hparams.n_embd],
            current,
            self.scratch.weight[0..self.hparams.n_embd],
            self.hparams.rms_epsilon,
        );

        return self.scanOutput(self.scratch.hidden1[0..self.hparams.n_embd], top_out);
    }

    pub fn benchmark(
        self: *Engine,
        token_id: usize,
        warmup: usize,
        iters: usize,
    ) !BenchmarkResult {
        if (iters == 0) return error.InvalidBenchIters;

        var checksum: f64 = 0;
        var warmup_top: [1]OutputCandidate = undefined;
        for (0..warmup) |_| {
            const result = try self.run(token_id, warmup_top[0..0]);
            checksum += result.argmax_logit;
        }

        const start_ns = try monotonicNowNs();
        for (0..iters) |_| {
            const result = try self.run(token_id, warmup_top[0..0]);
            checksum += result.argmax_logit;
        }
        const end_ns = try monotonicNowNs();
        const elapsed_ns = end_ns - start_ns;
        const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, std.time.ns_per_s);

        return .{
            .elapsed_s = elapsed_s,
            .passes_per_s = @as(f64, @floatFromInt(iters)) / elapsed_s,
            .checksum = checksum,
        };
    }

    fn loadLayer(self: *Engine, layer_idx: usize) !LayerWeights {
        var name_buffer: [96]u8 = undefined;
        const attn_norm = try self.store.tensor(layerTensorName(&name_buffer, layer_idx, "attn_norm.weight"));

        const kind = blk: {
            var full_name_buffer: [96]u8 = undefined;
            const full_name = layerTensorName(&full_name_buffer, layer_idx, "attn_output.weight");
            if (self.store.tensor(full_name)) |full_out| {
                break :blk LayerKind{
                    .full = .{
                        .q_gate = try self.store.tensor(layerTensorName(&full_name_buffer, layer_idx, "attn_q.weight")),
                        .v = try self.store.tensor(layerTensorName(&full_name_buffer, layer_idx, "attn_v.weight")),
                        .out = full_out,
                    },
                };
            } else |err| switch (err) {
                error.TensorNotFound => break :blk LayerKind{
                    .recurrent = .{
                        .qkv = try self.store.tensor(layerTensorName(&full_name_buffer, layer_idx, "attn_qkv.weight")),
                        .gate = try self.store.tensor(layerTensorName(&full_name_buffer, layer_idx, "attn_gate.weight")),
                        .beta = try self.store.tensor(layerTensorName(&full_name_buffer, layer_idx, "ssm_beta.weight")),
                        .conv = try self.store.tensor(layerTensorName(&full_name_buffer, layer_idx, "ssm_conv1d.weight")),
                        .ssm_norm = try self.store.tensor(layerTensorName(&full_name_buffer, layer_idx, "ssm_norm.weight")),
                        .out = try self.store.tensor(layerTensorName(&full_name_buffer, layer_idx, "ssm_out.weight")),
                    },
                },
                else => return err,
            }
        };

        var ffn_name_buffer: [96]u8 = undefined;
        return .{
            .attn_norm = attn_norm,
            .kind = kind,
            .ffn = .{
                .post_norm = try self.store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "post_attention_norm.weight")),
                .gate_inp = try self.store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_gate_inp.weight")),
                .up_exps = try self.store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_up_exps.weight")),
                .gate_exps = try self.store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_gate_exps.weight")),
                .down_exps = try self.store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_down_exps.weight")),
                .gate_inp_shexp = try self.store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_gate_inp_shexp.weight")),
                .up_shexp = try self.store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_up_shexp.weight")),
                .gate_shexp = try self.store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_gate_shexp.weight")),
                .down_shexp = try self.store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_down_shexp.weight")),
            },
        };
    }

    fn runLayer(self: *Engine, layer: LayerWeights, hidden: []const f32) !void {
        const embd = self.hparams.n_embd;
        try loadWeightRow(layer.attn_norm, self.scratch.weight[0..embd]);
        try ops.rmsNorm(
            self.scratch.hidden1[0..embd],
            hidden,
            self.scratch.weight[0..embd],
            self.hparams.rms_epsilon,
        );

        switch (layer.kind) {
            .recurrent => |weights| try self.runRecurrentAttention(weights, self.scratch.hidden1[0..embd], self.scratch.hidden2[0..embd]),
            .full => |weights| try self.runFullAttention(weights, self.scratch.hidden1[0..embd], self.scratch.hidden2[0..embd]),
        }

        addResidual(self.scratch.hidden2[0..embd], hidden);

        try loadWeightRow(layer.ffn.post_norm, self.scratch.weight[0..embd]);
        try ops.rmsNorm(
            self.scratch.hidden1[0..embd],
            self.scratch.hidden2[0..embd],
            self.scratch.weight[0..embd],
            self.hparams.rms_epsilon,
        );

        try self.runFfn(layer.ffn, self.scratch.hidden1[0..embd], self.scratch.hidden0[0..embd]);
        addResidual(self.scratch.hidden0[0..embd], self.scratch.hidden2[0..embd]);
    }

    fn runRecurrentAttention(
        self: *Engine,
        weights: RecurrentWeights,
        input: []const f32,
        out_hidden: []f32,
    ) !void {
        const head_k_dim = self.hparams.ssm_state_size;
        const num_k_heads = self.hparams.ssm_group_count;
        const num_v_heads = self.hparams.ssm_time_step_rank;
        const head_v_dim = self.hparams.ssm_inner_size / num_v_heads;
        const q_len = head_k_dim * num_k_heads;
        const k_len = q_len;
        const v_offset = q_len + k_len;
        const v_len = head_v_dim * num_v_heads;
        const qkv_len = v_offset + v_len;

        try projectAllRows(self.backend, weights.qkv, input, self.scratch.wide0[0..qkv_len]);
        try projectAllRows(self.backend, weights.gate, input, self.scratch.wide1[0..v_len]);
        try projectAllRows(self.backend, weights.beta, input, self.scratch.small32[0..num_v_heads]);

        for (self.scratch.small32[0..num_v_heads]) |*value| {
            value.* = sigmoid(value.*);
        }

        try applyConvTap(weights.conv, self.scratch.wide0[0..qkv_len], self.scratch.wide0[0..qkv_len]);

        l2NormalizeHeadsInPlace(self.scratch.wide0[0..q_len], head_k_dim, self.hparams.rms_epsilon);
        l2NormalizeHeadsInPlace(self.scratch.wide0[q_len..][0..k_len], head_k_dim, self.hparams.rms_epsilon);

        const q_heads = self.scratch.wide0[0..q_len];
        const k_heads = self.scratch.wide0[q_len..][0..k_len];
        const v_heads = self.scratch.wide0[v_offset..][0..v_len];
        const q_scale = @as(f32, @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(head_k_dim)))));

        for (0..num_v_heads) |v_head| {
            const k_head = v_head / (num_v_heads / num_k_heads);
            const q_base = k_head * head_k_dim;
            const k_base = k_head * head_k_dim;
            var dot: f32 = 0;
            for (0..head_k_dim) |idx| {
                dot += q_heads[q_base + idx] * k_heads[k_base + idx];
            }
            self.scratch.small32[v_head] *= dot * q_scale;
        }

        for (0..num_v_heads) |v_head| {
            const scale = self.scratch.small32[v_head];
            const v_base = v_head * head_v_dim;
            const out_base = v_head * head_v_dim;
            for (0..head_v_dim) |idx| {
                self.scratch.wide2[out_base + idx] = scale * v_heads[v_base + idx];
            }
        }

        try loadWeightRow(weights.ssm_norm, self.scratch.small128[0..head_v_dim]);
        for (0..num_v_heads) |v_head| {
            const start = v_head * head_v_dim;
            try ops.rmsNorm(
                self.scratch.wide0[start .. start + head_v_dim],
                self.scratch.wide2[start .. start + head_v_dim],
                self.scratch.small128[0..head_v_dim],
                self.hparams.rms_epsilon,
            );
        }

        for (self.scratch.wide0[0..v_len], self.scratch.wide1[0..v_len]) |*value, gate| {
            value.* *= silu(gate);
        }

        try projectAllRows(self.backend, weights.out, self.scratch.wide0[0..v_len], out_hidden);
    }

    fn runFullAttention(
        self: *Engine,
        weights: FullAttentionWeights,
        input: []const f32,
        out_hidden: []f32,
    ) !void {
        const head_dim = self.hparams.attention_value_len;
        const num_heads = self.hparams.n_head;
        const num_kv_heads = self.hparams.n_head_kv;
        const group_size = num_heads / num_kv_heads;
        const gate_len = num_heads * head_dim;
        const v_len = num_kv_heads * head_dim;

        try projectInterleavedGate(
            self.backend,
            weights.q_gate,
            input,
            num_heads,
            head_dim,
            self.scratch.wide2[0 .. gate_len * 2],
            self.scratch.wide1[0..gate_len],
        );
        try projectAllRows(self.backend, weights.v, input, self.scratch.wide2[0..v_len]);

        for (0..num_heads) |head_idx| {
            const kv_head = head_idx / group_size;
            const gate_base = head_idx * head_dim;
            const v_base = kv_head * head_dim;
            for (0..head_dim) |dim_idx| {
                self.scratch.wide0[gate_base + dim_idx] =
                    sigmoid(self.scratch.wide1[gate_base + dim_idx]) *
                    self.scratch.wide2[v_base + dim_idx];
            }
        }

        try projectAllRows(self.backend, weights.out, self.scratch.wide0[0..gate_len], out_hidden);
    }

    fn runFfn(
        self: *Engine,
        weights: FfnWeights,
        input: []const f32,
        out_hidden: []f32,
    ) !void {
        const embd = self.hparams.n_embd;
        const n_expert_used = self.hparams.n_expert_used;
        const expert_ff = self.hparams.expert_ffn_len;
        const shared_ff = self.hparams.expert_shared_ffn_len;

        try projectAllRows(self.backend, weights.gate_inp, input, self.scratch.router_logits);
        selectTopK(
            self.scratch.router_logits,
            self.scratch.selected_indices,
            self.scratch.selected_logits,
        );
        softmaxInPlace(self.scratch.selected_logits[0..n_expert_used], self.scratch.selected_weights[0..n_expert_used]);

        @memset(out_hidden, 0);
        for (0..n_expert_used) |slot| {
            const expert_idx = self.scratch.selected_indices[slot];
            const expert_weight = self.scratch.selected_weights[slot];

            try projectExpertUpGate(
                weights.up_exps,
                weights.gate_exps,
                expert_idx,
                input,
                expert_ff,
                self.scratch.expert0[0..expert_ff],
            );

            for (0..embd) |row_idx| {
                const down_row = expertRowIndex(weights.down_exps, row_idx, expert_idx);
                out_hidden[row_idx] += expert_weight * try weights.down_exps.dotRow(
                    down_row,
                    self.scratch.expert0[0..expert_ff],
                );
            }
        }

        const shared_gate = sigmoid(try weights.gate_inp_shexp.dotRow(0, input));
        try projectSharedExpert(
            weights.up_shexp,
            weights.gate_shexp,
            input,
            shared_ff,
            self.scratch.expert1[0..shared_ff],
        );
        for (0..embd) |row_idx| {
            out_hidden[row_idx] += shared_gate * try weights.down_shexp.dotRow(
                row_idx,
                self.scratch.expert1[0..shared_ff],
            );
        }
    }

    fn scanOutput(
        self: *Engine,
        hidden: []const f32,
        top_out: []OutputCandidate,
    ) !RunResult {
        if (top_out.len == 0) {
            if (self.backend) |backend| {
                if (self.output_logits) |logits| {
                    if (try backend.projectAllRows(self.output, hidden, logits)) {
                        return scanProjectedOutput(logits, top_out);
                    }
                }
            }

            const best = try parallel_rows.argmaxRows(self.output, hidden);
            return .{
                .argmax_token_id = best.row_index,
                .argmax_logit = best.value,
                .top_count = 0,
            };
        }

        if (self.backend) |backend| {
            if (self.output_logits) |logits| {
                if (try backend.projectAllRows(self.output, hidden, logits)) {
                    return scanProjectedOutput(logits, top_out);
                }
            }
        }

        var argmax_token_id: usize = 0;
        var argmax_logit = -std.math.inf(f32);
        initCandidates(top_out);

        for (0..self.output.row_count) |token_idx| {
            const logit = try self.output.dotRow(token_idx, hidden);
            if (logit > argmax_logit) {
                argmax_logit = logit;
                argmax_token_id = token_idx;
            }
            insertCandidate(top_out, .{
                .token_id = token_idx,
                .logit = logit,
            });
        }

        return .{
            .argmax_token_id = argmax_token_id,
            .argmax_logit = argmax_logit,
            .top_count = countCandidates(top_out),
        };
    }
};

fn kvUsize(model: *const gguf.File, key: []const u8) !usize {
    const entry = model.findKey(key) orelse return error.MissingMetadata;
    return switch (entry.value) {
        .uint8 => entry.value.uint8,
        .int8 => try castUsizeSigned(entry.value.int8),
        .uint16 => entry.value.uint16,
        .int16 => try castUsizeSigned(entry.value.int16),
        .uint32 => entry.value.uint32,
        .int32 => try castUsizeSigned(entry.value.int32),
        .uint64 => std.math.cast(usize, entry.value.uint64) orelse error.ValueTooLarge,
        .int64 => try castUsizeSigned(entry.value.int64),
        else => error.InvalidMetadataType,
    };
}

fn kvF32(model: *const gguf.File, key: []const u8, default: f32) f32 {
    const entry = model.findKey(key) orelse return default;
    return switch (entry.value) {
        .float32 => entry.value.float32,
        .float64 => @floatCast(entry.value.float64),
        else => default,
    };
}

fn castUsizeSigned(value: anytype) !usize {
    if (value < 0) return error.InvalidMetadataValue;
    return std.math.cast(usize, value) orelse error.ValueTooLarge;
}

fn loadWeightRow(tensor: gguf_store.TensorView, out: []f32) !void {
    if (tensor.row_count != 1 or tensor.row_len > out.len) {
        return error.InvalidWeightTensor;
    }
    try tensor.dequantizeRow(0, out[0..tensor.row_len]);
}

fn projectAllRows(
    backend: ?*metal_backend.Backend,
    tensor: gguf_store.TensorView,
    input: []const f32,
    out: []f32,
) !void {
    if (out.len < tensor.row_count) return error.OutputBufferTooSmall;
    if (backend) |metal| {
        if (try metal.projectAllRows(tensor, input, out)) return;
    }
    try parallel_rows.matvecRows(tensor, input, 0, tensor.row_count, out);
}

fn scanProjectedOutput(
    logits: []const f32,
    top_out: []OutputCandidate,
) RunResult {
    var argmax_token_id: usize = 0;
    var argmax_logit = -std.math.inf(f32);
    initCandidates(top_out);

    for (logits, 0..) |logit, token_idx| {
        if (logit > argmax_logit) {
            argmax_logit = logit;
            argmax_token_id = token_idx;
        }
        if (top_out.len != 0) {
            insertCandidate(top_out, .{
                .token_id = token_idx,
                .logit = logit,
            });
        }
    }

    return .{
        .argmax_token_id = argmax_token_id,
        .argmax_logit = argmax_logit,
        .top_count = countCandidates(top_out),
    };
}

fn projectInterleavedGate(
    backend: ?*metal_backend.Backend,
    tensor: gguf_store.TensorView,
    input: []const f32,
    num_heads: usize,
    head_dim: usize,
    temp: []f32,
    out: []f32,
) !void {
    const total = num_heads * head_dim;
    if (out.len < total) return error.OutputBufferTooSmall;
    if (temp.len < total * 2) return error.OutputBufferTooSmall;

    if (backend) |metal| {
        if (try metal.projectAllRows(tensor, input, temp[0 .. total * 2])) {
            const stride = head_dim * 2;
            for (0..num_heads) |head_idx| {
                const base = head_idx * stride + head_dim;
                const out_idx = head_idx * head_dim;
                @memcpy(out[out_idx .. out_idx + head_dim], temp[base .. base + head_dim]);
            }
            return;
        }
    }

    const stride = head_dim * 2;
    for (0..num_heads) |head_idx| {
        for (0..head_dim) |dim_idx| {
            const row_idx = head_idx * stride + head_dim + dim_idx;
            out[head_idx * head_dim + dim_idx] = try tensor.dotRow(row_idx, input);
        }
    }
}

fn applyConvTap(
    conv: gguf_store.TensorView,
    input: []const f32,
    out: []f32,
) !void {
    if (conv.row_len == 0 or conv.row_count > input.len or out.len < conv.row_count) {
        return error.InvalidConvTensor;
    }

    const tap_idx = conv.row_len - 1;
    var tap_weight: [4]f32 = undefined;
    if (conv.row_len > tap_weight.len) return error.UnsupportedConvKernel;

    for (0..conv.row_count) |channel| {
        try conv.dequantizeRow(channel, tap_weight[0..conv.row_len]);
        out[channel] = silu(input[channel] * tap_weight[tap_idx]);
    }
}

fn l2NormalizeHeadsInPlace(
    values: []f32,
    head_dim: usize,
    epsilon: f32,
) void {
    std.debug.assert(values.len % head_dim == 0);
    const n_heads = values.len / head_dim;
    for (0..n_heads) |head_idx| {
        const start = head_idx * head_dim;
        const head = values[start .. start + head_dim];
        var sum_sq: f64 = 0;
        for (head) |value| {
            sum_sq += @as(f64, value) * @as(f64, value);
        }
        const inv_norm = @as(f32, @floatCast(1.0 / std.math.sqrt(sum_sq + epsilon)));
        for (head) |*value| {
            value.* *= inv_norm;
        }
    }
}

fn projectExpertUpGate(
    up_tensor: gguf_store.TensorView,
    gate_tensor: gguf_store.TensorView,
    expert_idx: usize,
    input: []const f32,
    expert_ff: usize,
    out: []f32,
) !void {
    if (out.len < expert_ff) return error.OutputBufferTooSmall;
    for (0..expert_ff) |row_idx| {
        const expert_row = expertRowIndex(up_tensor, row_idx, expert_idx);
        const gate_row = expertRowIndex(gate_tensor, row_idx, expert_idx);
        const up_value = try up_tensor.dotRow(expert_row, input);
        const gate_value = try gate_tensor.dotRow(gate_row, input);
        out[row_idx] = silu(gate_value) * up_value;
    }
}

fn projectSharedExpert(
    up_tensor: gguf_store.TensorView,
    gate_tensor: gguf_store.TensorView,
    input: []const f32,
    expert_ff: usize,
    out: []f32,
) !void {
    if (out.len < expert_ff) return error.OutputBufferTooSmall;
    for (0..expert_ff) |row_idx| {
        const up_value = try up_tensor.dotRow(row_idx, input);
        const gate_value = try gate_tensor.dotRow(row_idx, input);
        out[row_idx] = silu(gate_value) * up_value;
    }
}

fn expertRowIndex(
    tensor: gguf_store.TensorView,
    row_idx: usize,
    expert_idx: usize,
) usize {
    std.debug.assert(tensor.info.dimensions.len >= 3);
    const rows_per_expert = @as(usize, @intCast(tensor.info.dimensions[1]));
    return expert_idx * rows_per_expert + row_idx;
}

fn selectTopK(
    values: []const f32,
    indices_out: []usize,
    logits_out: []f32,
) void {
    std.debug.assert(indices_out.len == logits_out.len);
    for (indices_out, logits_out) |*index, *logit| {
        index.* = 0;
        logit.* = -std.math.inf(f32);
    }

    for (values, 0..) |value, idx| {
        var insert_at: ?usize = null;
        for (logits_out, 0..) |current, top_idx| {
            if (value > current) {
                insert_at = top_idx;
                break;
            }
        }
        if (insert_at) |pos| {
            var shift = logits_out.len;
            while (shift > pos + 1) : (shift -= 1) {
                logits_out[shift - 1] = logits_out[shift - 2];
                indices_out[shift - 1] = indices_out[shift - 2];
            }
            logits_out[pos] = value;
            indices_out[pos] = idx;
        }
    }
}

fn softmaxInPlace(
    logits: []const f32,
    out: []f32,
) void {
    std.debug.assert(logits.len == out.len);
    if (logits.len == 0) return;

    var max_logit = logits[0];
    for (logits[1..]) |value| {
        max_logit = @max(max_logit, value);
    }

    var sum: f64 = 0;
    for (logits, out) |value, *dst| {
        dst.* = @as(f32, @floatCast(std.math.exp(@as(f64, value - max_logit))));
        sum += dst.*;
    }

    const inv_sum = @as(f32, @floatCast(1.0 / sum));
    for (out) |*value| {
        value.* *= inv_sum;
    }
}

fn addResidual(dst: []f32, residual: []const f32) void {
    std.debug.assert(dst.len == residual.len);
    for (dst, residual) |*value, addend| {
        value.* += addend;
    }
}

fn initCandidates(candidates: []OutputCandidate) void {
    for (candidates) |*candidate| {
        candidate.* = .{
            .token_id = std.math.maxInt(usize),
            .logit = -std.math.inf(f32),
        };
    }
}

fn insertCandidate(candidates: []OutputCandidate, candidate: OutputCandidate) void {
    if (candidates.len == 0) return;

    var insert_at: ?usize = null;
    for (candidates, 0..) |current, idx| {
        if (candidate.logit > current.logit) {
            insert_at = idx;
            break;
        }
    }
    if (insert_at) |pos| {
        var shift = candidates.len;
        while (shift > pos + 1) : (shift -= 1) {
            candidates[shift - 1] = candidates[shift - 2];
        }
        candidates[pos] = candidate;
    }
}

fn countCandidates(candidates: []const OutputCandidate) usize {
    var count: usize = 0;
    for (candidates) |candidate| {
        if (candidate.token_id == std.math.maxInt(usize)) break;
        count += 1;
    }
    return count;
}

fn layerTensorName(
    buffer: []u8,
    layer_idx: usize,
    suffix: []const u8,
) []const u8 {
    return std.fmt.bufPrint(buffer, "blk.{d}.{s}", .{ layer_idx, suffix }) catch unreachable;
}

fn sigmoid(value: f32) f32 {
    if (value >= 0) {
        const exp_neg = @as(f32, @floatCast(std.math.exp(@as(f64, -value))));
        return 1.0 / (1.0 + exp_neg);
    }
    const exp_pos = @as(f32, @floatCast(std.math.exp(@as(f64, value))));
    return exp_pos / (1.0 + exp_pos);
}

fn silu(value: f32) f32 {
    return value * sigmoid(value);
}

fn monotonicNowNs() !u64 {
    var ts: std.posix.timespec = undefined;
    if (std.c.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts) != 0) {
        return error.ClockGetTimeFailed;
    }

    const total_ns = @as(i128, ts.sec) * std.time.ns_per_s + @as(i128, ts.nsec);
    return std.math.cast(u64, total_ns) orelse error.ClockOutOfRange;
}

test "top-k logits stay sorted" {
    const values = [_]f32{ -1.0, 2.0, 0.5, 3.5, 2.5 };
    var indices: [3]usize = undefined;
    var logits: [3]f32 = undefined;

    selectTopK(&values, &indices, &logits);

    try std.testing.expectEqual(@as(usize, 3), indices[0]);
    try std.testing.expectEqual(@as(usize, 4), indices[1]);
    try std.testing.expectEqual(@as(usize, 1), indices[2]);
}

test "selected-logit softmax normalizes" {
    const logits = [_]f32{ 3.0, 1.0, -2.0 };
    var weights: [3]f32 = undefined;

    softmaxInPlace(&logits, &weights);

    var sum: f32 = 0;
    for (weights) |value| sum += value;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.0001);
    try std.testing.expect(weights[0] > weights[1]);
    try std.testing.expect(weights[1] > weights[2]);
}
