const std = @import("std");
const gguf = @import("gguf.zig");
const gguf_store = @import("gguf_store.zig");
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
    position: usize,
};

pub const BenchmarkResult = struct {
    elapsed_s: f64,
    tok_per_s: f64,
    checksum: f64,
    timed_tokens: usize,
    warmup_tokens: usize,
};

const HParams = struct {
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
    rope_freq_base: f32,
    rope_sections: [4]usize,

    fn load(model: *const gguf.File) !HParams {
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
            .rope_freq_base = kvF32(model, "qwen35moe.rope.freq_base", 10000.0),
            .rope_sections = try kvSections4(model, "qwen35moe.rope.dimension_sections"),
        };
    }
};

const RecurrentWeights = struct {
    qkv: gguf_store.TensorView,
    gate: gguf_store.TensorView,
    alpha: gguf_store.TensorView,
    beta: gguf_store.TensorView,
    conv: gguf_store.TensorView,
    dt: gguf_store.TensorView,
    a: gguf_store.TensorView,
    ssm_norm: gguf_store.TensorView,
    out: gguf_store.TensorView,
};

const FullAttentionWeights = struct {
    q_gate: gguf_store.TensorView,
    q_norm: gguf_store.TensorView,
    k: gguf_store.TensorView,
    k_norm: gguf_store.TensorView,
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
    cache_index: usize,
};

const RecurrentCache = struct {
    conv_history: []f32,
    state: []f32,
    conv_slot: usize,
};

const FullCache = struct {
    k: []f32,
    v: []f32,
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
    small_head: []f32,
    scores: []f32,

    fn init(
        allocator: std.mem.Allocator,
        hparams: HParams,
        max_seq_len: usize,
    ) !Scratch {
        const max_wide = @max(
            hparams.ssm_inner_size + 2 * hparams.ssm_group_count * hparams.ssm_state_size,
            hparams.n_head * hparams.attention_value_len,
        );
        const max_expert = @max(hparams.expert_ffn_len, hparams.expert_shared_ffn_len);
        const max_small32 = @max(hparams.ssm_time_step_rank, hparams.n_head);
        const max_head = @max(hparams.ssm_state_size, hparams.attention_key_len);

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
            .small_head = try allocator.alloc(f32, max_head),
            .scores = try allocator.alloc(f32, max_seq_len),
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
        allocator.free(self.small_head);
        allocator.free(self.scores);
        self.* = undefined;
    }
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    store: *const gguf_store.Store,
    hparams: HParams,
    max_seq_len: usize,
    position: usize,
    token_embd: gguf_store.TensorView,
    output_norm: gguf_store.TensorView,
    output: gguf_store.TensorView,
    layers: []LayerWeights,
    recurrent_caches: []RecurrentCache,
    full_caches: []FullCache,
    scratch: Scratch,

    pub fn init(
        allocator: std.mem.Allocator,
        store: *const gguf_store.Store,
        max_seq_len: usize,
    ) !Engine {
        if (max_seq_len == 0) return error.InvalidMaxSeqLen;

        const hparams = try HParams.load(&store.parsed);
        const token_embd = try store.tensor("token_embd.weight");
        const output_norm = try store.tensor("output_norm.weight");
        const output = try store.tensor("output.weight");

        var full_count: usize = 0;
        for (0..hparams.n_layers) |layer_idx| {
            if (try isFullLayer(store, layer_idx)) {
                full_count += 1;
            }
        }
        const recurrent_count = hparams.n_layers - full_count;

        const layers = try allocator.alloc(LayerWeights, hparams.n_layers);
        errdefer allocator.free(layers);

        const recurrent_caches = try allocator.alloc(RecurrentCache, recurrent_count);
        errdefer allocator.free(recurrent_caches);

        const full_caches = try allocator.alloc(FullCache, full_count);
        errdefer allocator.free(full_caches);

        const scratch = try Scratch.init(allocator, hparams, max_seq_len);
        errdefer {
            var scratch_copy = scratch;
            scratch_copy.deinit(allocator);
        }

        var recurrent_index: usize = 0;
        var full_index: usize = 0;
        for (0..hparams.n_layers) |layer_idx| {
            const layer = try loadLayer(store, layer_idx, recurrent_index, full_index);
            layers[layer_idx] = layer;

            switch (layer.kind) {
                .recurrent => {
                    recurrent_caches[recurrent_index] = try initRecurrentCache(
                        allocator,
                        hparams,
                    );
                    recurrent_index += 1;
                },
                .full => {
                    full_caches[full_index] = try initFullCache(
                        allocator,
                        hparams,
                        max_seq_len,
                    );
                    full_index += 1;
                },
            }
        }

        return .{
            .allocator = allocator,
            .store = store,
            .hparams = hparams,
            .max_seq_len = max_seq_len,
            .position = 0,
            .token_embd = token_embd,
            .output_norm = output_norm,
            .output = output,
            .layers = layers,
            .recurrent_caches = recurrent_caches,
            .full_caches = full_caches,
            .scratch = scratch,
        };
    }

    pub fn deinit(self: *Engine) void {
        for (self.recurrent_caches) |cache| {
            self.allocator.free(cache.conv_history);
            self.allocator.free(cache.state);
        }
        for (self.full_caches) |cache| {
            self.allocator.free(cache.k);
            self.allocator.free(cache.v);
        }
        self.allocator.free(self.recurrent_caches);
        self.allocator.free(self.full_caches);
        self.scratch.deinit(self.allocator);
        self.allocator.free(self.layers);
        self.* = undefined;
    }

    pub fn reset(self: *Engine) void {
        self.position = 0;
        for (self.recurrent_caches) |*cache| {
            @memset(cache.conv_history, 0);
            @memset(cache.state, 0);
            cache.conv_slot = 0;
        }
        for (self.full_caches) |cache| {
            @memset(cache.k, 0);
            @memset(cache.v, 0);
        }
    }

    pub fn step(
        self: *Engine,
        token_id: usize,
        top_out: []OutputCandidate,
    ) !RunResult {
        if (token_id >= self.token_embd.row_count) return error.TokenIdOutOfRange;
        if (self.position >= self.max_seq_len) return error.DecodeCacheFull;

        try self.token_embd.dequantizeRow(token_id, self.scratch.hidden0[0..self.hparams.n_embd]);

        for (self.layers) |layer| {
            try self.stepLayer(layer, self.scratch.hidden0[0..self.hparams.n_embd]);
        }

        try loadWeightRow(self.output_norm, self.scratch.weight[0..self.hparams.n_embd]);
        try ops.rmsNorm(
            self.scratch.hidden1[0..self.hparams.n_embd],
            self.scratch.hidden0[0..self.hparams.n_embd],
            self.scratch.weight[0..self.hparams.n_embd],
            self.hparams.rms_epsilon,
        );

        const result = try scanOutput(
            self.output,
            self.scratch.hidden1[0..self.hparams.n_embd],
            top_out,
        );
        self.position += 1;

        return .{
            .argmax_token_id = result.argmax_token_id,
            .argmax_logit = result.argmax_logit,
            .top_count = result.top_count,
            .position = self.position,
        };
    }

    pub fn runRepeated(
        self: *Engine,
        token_id: usize,
        steps: usize,
        top_out: []OutputCandidate,
    ) !RunResult {
        if (steps == 0) return error.InvalidDecodeSteps;
        self.reset();

        var result: RunResult = undefined;
        for (0..steps) |step_idx| {
            result = try self.step(token_id, if (step_idx + 1 == steps) top_out else top_out[0..0]);
        }
        return result;
    }

    pub fn benchmark(
        self: *Engine,
        token_id: usize,
        warmup_tokens: usize,
        timed_tokens: usize,
    ) !BenchmarkResult {
        if (timed_tokens == 0) return error.InvalidBenchIters;
        if (warmup_tokens + timed_tokens > self.max_seq_len) return error.DecodeCacheFull;

        self.reset();
        var checksum: f64 = 0;
        var sink: [1]OutputCandidate = undefined;

        for (0..warmup_tokens) |_| {
            const result = try self.step(token_id, sink[0..0]);
            checksum += result.argmax_logit;
        }

        const start_ns = try monotonicNowNs();
        for (0..timed_tokens) |_| {
            const result = try self.step(token_id, sink[0..0]);
            checksum += result.argmax_logit;
        }
        const end_ns = try monotonicNowNs();
        const elapsed_ns = end_ns - start_ns;
        const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, std.time.ns_per_s);

        return .{
            .elapsed_s = elapsed_s,
            .tok_per_s = @as(f64, @floatFromInt(timed_tokens)) / elapsed_s,
            .checksum = checksum,
            .timed_tokens = timed_tokens,
            .warmup_tokens = warmup_tokens,
        };
    }

    fn stepLayer(self: *Engine, layer: LayerWeights, hidden: []const f32) !void {
        const embd = self.hparams.n_embd;
        try loadWeightRow(layer.attn_norm, self.scratch.weight[0..embd]);
        try ops.rmsNorm(
            self.scratch.hidden1[0..embd],
            hidden,
            self.scratch.weight[0..embd],
            self.hparams.rms_epsilon,
        );

        switch (layer.kind) {
            .recurrent => |weights| try self.stepRecurrent(
                &self.recurrent_caches[layer.cache_index],
                weights,
                self.scratch.hidden1[0..embd],
                self.scratch.hidden2[0..embd],
            ),
            .full => |weights| try self.stepFullAttention(
                self.full_caches[layer.cache_index],
                weights,
                self.scratch.hidden1[0..embd],
                self.scratch.hidden2[0..embd],
            ),
        }

        addInPlace(self.scratch.hidden2[0..embd], hidden);

        try loadWeightRow(layer.ffn.post_norm, self.scratch.weight[0..embd]);
        try ops.rmsNorm(
            self.scratch.hidden1[0..embd],
            self.scratch.hidden2[0..embd],
            self.scratch.weight[0..embd],
            self.hparams.rms_epsilon,
        );

        try runFfn(
            self.hparams,
            &self.scratch,
            layer.ffn,
            self.scratch.hidden1[0..embd],
            self.scratch.hidden0[0..embd],
        );
        addInPlace(self.scratch.hidden0[0..embd], self.scratch.hidden2[0..embd]);
    }

    fn stepRecurrent(
        self: *Engine,
        cache: *RecurrentCache,
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
        const conv_hist_stride = qkv_len;
        const conv_slots = self.hparams.ssm_conv_kernel - 1;
        const q_scale = @as(f32, @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(head_k_dim)))));

        try projectAllRows(weights.qkv, input, self.scratch.wide0[0..qkv_len]);
        try projectAllRows(weights.gate, input, self.scratch.wide1[0..v_len]);
        try projectAllRows(weights.alpha, input, self.scratch.expert0[0..num_v_heads]);
        try projectAllRows(weights.beta, input, self.scratch.small32[0..num_v_heads]);
        try loadWeightRow(weights.dt, self.scratch.expert1[0..num_v_heads]);
        try loadWeightRow(weights.a, self.scratch.weight[0..num_v_heads]);

        for (0..num_v_heads) |head_idx| {
            self.scratch.small32[head_idx] = sigmoid(self.scratch.small32[head_idx]);
            const alpha_biased = self.scratch.expert0[head_idx] + self.scratch.expert1[head_idx];
            self.scratch.expert0[head_idx] = @as(
                f32,
                @floatCast(std.math.exp(@as(f64, softplus(alpha_biased) * self.scratch.weight[head_idx]))),
            );
        }

        if (weights.conv.row_len != self.hparams.ssm_conv_kernel) return error.UnsupportedConvKernel;
        if (conv_slots != 3) return error.UnsupportedConvKernel;
        const hist0 = cache.conv_slot * conv_hist_stride;
        const hist1 = ((cache.conv_slot + 1) % conv_slots) * conv_hist_stride;
        const hist2 = ((cache.conv_slot + 2) % conv_slots) * conv_hist_stride;
        var kernel: [4]f32 = undefined;
        for (0..qkv_len) |channel_idx| {
            try weights.conv.dequantizeRow(channel_idx, kernel[0..weights.conv.row_len]);
            const h0 = cache.conv_history[hist0 + channel_idx];
            const h1 = cache.conv_history[hist1 + channel_idx];
            const h2 = cache.conv_history[hist2 + channel_idx];
            const current = self.scratch.wide0[channel_idx];
            self.scratch.wide2[channel_idx] = silu(
                h0 * kernel[0] +
                    h1 * kernel[1] +
                    h2 * kernel[2] +
                    current * kernel[3],
            );
            cache.conv_history[hist0 + channel_idx] = current;
        }
        cache.conv_slot = (cache.conv_slot + 1) % conv_slots;

        l2NormalizeHeadsInPlace(
            self.scratch.wide2[0..q_len],
            head_k_dim,
            self.hparams.rms_epsilon,
        );
        l2NormalizeHeadsInPlace(
            self.scratch.wide2[q_len .. q_len + k_len],
            head_k_dim,
            self.hparams.rms_epsilon,
        );

        @memset(self.scratch.wide0[0..v_len], 0);

        for (0..num_v_heads) |v_head| {
            const k_head = v_head / (num_v_heads / num_k_heads);
            const q_vec = self.scratch.wide2[k_head * head_k_dim ..][0..head_k_dim];
            const k_vec = self.scratch.wide2[q_len + k_head * head_k_dim ..][0..head_k_dim];
            const v_vec = self.scratch.wide2[v_offset + v_head * head_v_dim ..][0..head_v_dim];
            const decay = self.scratch.expert0[v_head];
            const beta = self.scratch.small32[v_head];

            const state = cache.state[
                v_head * head_v_dim * head_v_dim ..
            ][0 .. head_v_dim * head_v_dim];

            const sk = self.scratch.expert0[0..head_v_dim];
            const delta = self.scratch.expert1[0..head_v_dim];
            @memset(sk, 0);

            for (0..head_v_dim) |k_idx| {
                const row = state[k_idx * head_v_dim ..][0..head_v_dim];
                const k_value = k_vec[k_idx];
                for (0..head_v_dim) |v_idx| {
                    row[v_idx] *= decay;
                    sk[v_idx] += row[v_idx] * k_value;
                }
            }

            for (0..head_v_dim) |v_idx| {
                delta[v_idx] = beta * (v_vec[v_idx] - sk[v_idx]);
            }

            const output_head = self.scratch.wide0[v_head * head_v_dim ..][0..head_v_dim];
            @memset(output_head, 0);
            for (0..head_v_dim) |k_idx| {
                const row = state[k_idx * head_v_dim ..][0..head_v_dim];
                const k_value = k_vec[k_idx];
                const q_value = q_vec[k_idx] * q_scale;
                for (0..head_v_dim) |v_idx| {
                    row[v_idx] += k_value * delta[v_idx];
                    output_head[v_idx] += row[v_idx] * q_value;
                }
            }
        }

        try loadWeightRow(weights.ssm_norm, self.scratch.small_head[0..head_v_dim]);
        for (0..num_v_heads) |v_head| {
            const start = v_head * head_v_dim;
            try ops.rmsNorm(
                self.scratch.wide2[start .. start + head_v_dim],
                self.scratch.wide0[start .. start + head_v_dim],
                self.scratch.small_head[0..head_v_dim],
                self.hparams.rms_epsilon,
            );
        }

        for (0..v_len) |idx| {
            self.scratch.wide2[idx] *= silu(self.scratch.wide1[idx]);
        }

        try projectAllRows(weights.out, self.scratch.wide2[0..v_len], out_hidden);
    }

    fn stepFullAttention(
        self: *Engine,
        cache: FullCache,
        weights: FullAttentionWeights,
        input: []const f32,
        out_hidden: []f32,
    ) !void {
        const head_dim = self.hparams.attention_key_len;
        const num_heads = self.hparams.n_head;
        const num_kv_heads = self.hparams.n_head_kv;
        const group_size = num_heads / num_kv_heads;
        const q_len = num_heads * head_dim;
        const kv_len = num_kv_heads * head_dim;
        const attn_out_offset = kv_len * 2;
        const q_scale = @as(f32, @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(head_dim)))));
        const seq_len = self.position + 1;

        try projectQAndGate(
            weights.q_gate,
            input,
            num_heads,
            head_dim,
            self.scratch.wide0[0..q_len],
            self.scratch.wide1[0..q_len],
        );
        try loadWeightRow(weights.q_norm, self.scratch.small_head[0..head_dim]);
        for (0..num_heads) |head_idx| {
            const start = head_idx * head_dim;
            try ops.rmsNorm(
                self.scratch.wide0[start .. start + head_dim],
                self.scratch.wide0[start .. start + head_dim],
                self.scratch.small_head[0..head_dim],
                self.hparams.rms_epsilon,
            );
            applyTextImropeNeoxInPlace(
                self.scratch.wide0[start .. start + head_dim],
                self.position,
                self.hparams.rope_sections,
                self.hparams.rope_freq_base,
            );
        }

        try projectAllRows(weights.k, input, self.scratch.wide2[0..kv_len]);
        try loadWeightRow(weights.k_norm, self.scratch.small_head[0..head_dim]);
        for (0..num_kv_heads) |head_idx| {
            const start = head_idx * head_dim;
            try ops.rmsNorm(
                self.scratch.wide2[start .. start + head_dim],
                self.scratch.wide2[start .. start + head_dim],
                self.scratch.small_head[0..head_dim],
                self.hparams.rms_epsilon,
            );
            applyTextImropeNeoxInPlace(
                self.scratch.wide2[start .. start + head_dim],
                self.position,
                self.hparams.rope_sections,
                self.hparams.rope_freq_base,
            );
        }

        try projectAllRows(weights.v, input, self.scratch.wide2[kv_len .. kv_len * 2]);

        for (0..num_kv_heads) |head_idx| {
            const cache_base = (self.position * num_kv_heads + head_idx) * head_dim;
            const src_base = head_idx * head_dim;
            @memcpy(
                cache.k[cache_base .. cache_base + head_dim],
                self.scratch.wide2[src_base .. src_base + head_dim],
            );
            @memcpy(
                cache.v[cache_base .. cache_base + head_dim],
                self.scratch.wide2[kv_len + src_base .. kv_len + src_base + head_dim],
            );
        }

        @memset(self.scratch.wide2[attn_out_offset .. attn_out_offset + q_len], 0);
        for (0..num_heads) |head_idx| {
            const kv_head = head_idx / group_size;
            const q_head = self.scratch.wide0[head_idx * head_dim ..][0..head_dim];
            const attn_head = self.scratch.wide2[attn_out_offset + head_idx * head_dim ..][0..head_dim];

            for (0..seq_len) |token_pos| {
                const key = cache.k[(token_pos * num_kv_heads + kv_head) * head_dim ..][0..head_dim];
                self.scratch.scores[token_pos] = q_scale * dot(q_head, key);
            }
            softmaxSliceInPlace(self.scratch.scores[0..seq_len]);

            for (0..seq_len) |token_pos| {
                const weight = self.scratch.scores[token_pos];
                const value = cache.v[(token_pos * num_kv_heads + kv_head) * head_dim ..][0..head_dim];
                for (0..head_dim) |dim_idx| {
                    attn_head[dim_idx] += weight * value[dim_idx];
                }
            }
        }

        for (0..q_len) |idx| {
            self.scratch.wide2[attn_out_offset + idx] *= sigmoid(self.scratch.wide1[idx]);
        }

        try projectAllRows(
            weights.out,
            self.scratch.wide2[attn_out_offset .. attn_out_offset + q_len],
            out_hidden,
        );
    }
};

fn initRecurrentCache(
    allocator: std.mem.Allocator,
    hparams: HParams,
) !RecurrentCache {
    const qkv_len = hparams.ssm_inner_size + 2 * hparams.ssm_group_count * hparams.ssm_state_size;
    const conv_history_len = (hparams.ssm_conv_kernel - 1) * qkv_len;
    const head_v_dim = hparams.ssm_inner_size / hparams.ssm_time_step_rank;
    const state_len = hparams.ssm_time_step_rank * head_v_dim * head_v_dim;

    const conv_history = try allocator.alloc(f32, conv_history_len);
    errdefer allocator.free(conv_history);
    const state = try allocator.alloc(f32, state_len);
    errdefer allocator.free(state);

    @memset(conv_history, 0);
    @memset(state, 0);

    return .{
        .conv_history = conv_history,
        .state = state,
        .conv_slot = 0,
    };
}

fn initFullCache(
    allocator: std.mem.Allocator,
    hparams: HParams,
    max_seq_len: usize,
) !FullCache {
    const kv_len = max_seq_len * hparams.n_head_kv * hparams.attention_key_len;
    const k = try allocator.alloc(f32, kv_len);
    errdefer allocator.free(k);
    const v = try allocator.alloc(f32, kv_len);
    errdefer allocator.free(v);

    @memset(k, 0);
    @memset(v, 0);

    return .{
        .k = k,
        .v = v,
    };
}

fn loadLayer(
    store: *const gguf_store.Store,
    layer_idx: usize,
    recurrent_index: usize,
    full_index: usize,
) !LayerWeights {
    var name_buffer: [96]u8 = undefined;
    const attn_norm = try store.tensor(layerTensorName(&name_buffer, layer_idx, "attn_norm.weight"));

    const kind = blk: {
        var full_name_buffer: [96]u8 = undefined;
        const full_name = layerTensorName(&full_name_buffer, layer_idx, "attn_output.weight");
        if (store.tensor(full_name)) |full_out| {
            break :blk LayerKind{
                .full = .{
                    .q_gate = try store.tensor(layerTensorName(&full_name_buffer, layer_idx, "attn_q.weight")),
                    .q_norm = try store.tensor(layerTensorName(&full_name_buffer, layer_idx, "attn_q_norm.weight")),
                    .k = try store.tensor(layerTensorName(&full_name_buffer, layer_idx, "attn_k.weight")),
                    .k_norm = try store.tensor(layerTensorName(&full_name_buffer, layer_idx, "attn_k_norm.weight")),
                    .v = try store.tensor(layerTensorName(&full_name_buffer, layer_idx, "attn_v.weight")),
                    .out = full_out,
                },
            };
        } else |err| switch (err) {
            error.TensorNotFound => break :blk LayerKind{
                .recurrent = .{
                    .qkv = try store.tensor(layerTensorName(&full_name_buffer, layer_idx, "attn_qkv.weight")),
                    .gate = try store.tensor(layerTensorName(&full_name_buffer, layer_idx, "attn_gate.weight")),
                    .alpha = try store.tensor(layerTensorName(&full_name_buffer, layer_idx, "ssm_alpha.weight")),
                    .beta = try store.tensor(layerTensorName(&full_name_buffer, layer_idx, "ssm_beta.weight")),
                    .conv = try store.tensor(layerTensorName(&full_name_buffer, layer_idx, "ssm_conv1d.weight")),
                    .dt = try store.tensor(layerTensorName(&full_name_buffer, layer_idx, "ssm_dt.bias")),
                    .a = try store.tensor(layerTensorName(&full_name_buffer, layer_idx, "ssm_a")),
                    .ssm_norm = try store.tensor(layerTensorName(&full_name_buffer, layer_idx, "ssm_norm.weight")),
                    .out = try store.tensor(layerTensorName(&full_name_buffer, layer_idx, "ssm_out.weight")),
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
            .post_norm = try store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "post_attention_norm.weight")),
            .gate_inp = try store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_gate_inp.weight")),
            .up_exps = try store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_up_exps.weight")),
            .gate_exps = try store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_gate_exps.weight")),
            .down_exps = try store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_down_exps.weight")),
            .gate_inp_shexp = try store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_gate_inp_shexp.weight")),
            .up_shexp = try store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_up_shexp.weight")),
            .gate_shexp = try store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_gate_shexp.weight")),
            .down_shexp = try store.tensor(layerTensorName(&ffn_name_buffer, layer_idx, "ffn_down_shexp.weight")),
        },
        .cache_index = switch (kind) {
            .recurrent => recurrent_index,
            .full => full_index,
        },
    };
}

fn isFullLayer(store: *const gguf_store.Store, layer_idx: usize) !bool {
    var name_buffer: [96]u8 = undefined;
    const name = layerTensorName(&name_buffer, layer_idx, "attn_output.weight");
    _ = store.tensor(name) catch |err| switch (err) {
        error.TensorNotFound => return false,
        else => return err,
    };
    return true;
}

fn runFfn(
    hparams: HParams,
    scratch: *Scratch,
    weights: FfnWeights,
    input: []const f32,
    out_hidden: []f32,
) !void {
    const embd = hparams.n_embd;
    const n_expert_used = hparams.n_expert_used;
    const expert_ff = hparams.expert_ffn_len;
    const shared_ff = hparams.expert_shared_ffn_len;

    try projectAllRows(weights.gate_inp, input, scratch.router_logits);
    selectTopK(
        scratch.router_logits,
        scratch.selected_indices,
        scratch.selected_logits,
    );
    softmaxInto(
        scratch.selected_logits[0..n_expert_used],
        scratch.selected_weights[0..n_expert_used],
    );

    @memset(out_hidden, 0);
    for (0..n_expert_used) |slot| {
        const expert_idx = scratch.selected_indices[slot];
        const expert_weight = scratch.selected_weights[slot];

        try projectExpertUpGate(
            weights.up_exps,
            weights.gate_exps,
            expert_idx,
            input,
            expert_ff,
            scratch.expert0[0..expert_ff],
        );

        for (0..embd) |row_idx| {
            const down_row = expertRowIndex(weights.down_exps, row_idx, expert_idx);
            out_hidden[row_idx] += expert_weight * try weights.down_exps.dotRow(
                down_row,
                scratch.expert0[0..expert_ff],
            );
        }
    }

    const shared_gate = sigmoid(try weights.gate_inp_shexp.dotRow(0, input));
    try projectSharedExpert(
        weights.up_shexp,
        weights.gate_shexp,
        input,
        shared_ff,
        scratch.expert1[0..shared_ff],
    );
    for (0..embd) |row_idx| {
        out_hidden[row_idx] += shared_gate * try weights.down_shexp.dotRow(
            row_idx,
            scratch.expert1[0..shared_ff],
        );
    }
}

fn scanOutput(
    tensor: gguf_store.TensorView,
    hidden: []const f32,
    top_out: []OutputCandidate,
) !struct {
    argmax_token_id: usize,
    argmax_logit: f32,
    top_count: usize,
} {
    if (top_out.len == 0) {
        const best = try parallel_rows.argmaxRows(tensor, hidden);
        return .{
            .argmax_token_id = best.row_index,
            .argmax_logit = best.value,
            .top_count = 0,
        };
    }

    var argmax_token_id: usize = 0;
    var argmax_logit = -std.math.inf(f32);
    initCandidates(top_out);

    for (0..tensor.row_count) |token_idx| {
        const logit = try tensor.dotRow(token_idx, hidden);
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

fn kvSections4(model: *const gguf.File, key: []const u8) ![4]usize {
    const entry = model.findKey(key) orelse return error.MissingMetadata;
    if (entry.value != .array) return error.InvalidMetadataType;
    const array = entry.value.array;
    return switch (array.preview) {
        .ints => |values| blk: {
            if (values.len < 4) return error.InvalidMetadataValue;
            break :blk .{
                try castUsizeSigned(values[0]),
                try castUsizeSigned(values[1]),
                try castUsizeSigned(values[2]),
                try castUsizeSigned(values[3]),
            };
        },
        .uints => |values| blk: {
            if (values.len < 4) return error.InvalidMetadataValue;
            break :blk .{
                std.math.cast(usize, values[0]) orelse return error.ValueTooLarge,
                std.math.cast(usize, values[1]) orelse return error.ValueTooLarge,
                std.math.cast(usize, values[2]) orelse return error.ValueTooLarge,
                std.math.cast(usize, values[3]) orelse return error.ValueTooLarge,
            };
        },
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

fn layerTensorName(
    buffer: []u8,
    layer_idx: usize,
    suffix: []const u8,
) []const u8 {
    return std.fmt.bufPrint(buffer, "blk.{d}.{s}", .{ layer_idx, suffix }) catch unreachable;
}

fn loadWeightRow(tensor: gguf_store.TensorView, out: []f32) !void {
    if (tensor.row_count != 1 or tensor.row_len > out.len) {
        return error.InvalidWeightTensor;
    }
    try tensor.dequantizeRow(0, out[0..tensor.row_len]);
}

fn projectAllRows(
    tensor: gguf_store.TensorView,
    input: []const f32,
    out: []f32,
) !void {
    if (out.len < tensor.row_count) return error.OutputBufferTooSmall;
    for (0..tensor.row_count) |row_idx| {
        out[row_idx] = try tensor.dotRow(row_idx, input);
    }
}

fn projectQAndGate(
    tensor: gguf_store.TensorView,
    input: []const f32,
    num_heads: usize,
    head_dim: usize,
    q_out: []f32,
    gate_out: []f32,
) !void {
    const total = num_heads * head_dim;
    if (q_out.len < total or gate_out.len < total) return error.OutputBufferTooSmall;

    const stride = head_dim * 2;
    for (0..num_heads) |head_idx| {
        for (0..head_dim) |dim_idx| {
            const base = head_idx * stride + dim_idx;
            const out_idx = head_idx * head_dim + dim_idx;
            q_out[out_idx] = try tensor.dotRow(base, input);
            gate_out[out_idx] = try tensor.dotRow(base + head_dim, input);
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

fn applyTextImropeNeoxInPlace(
    values: []f32,
    position: usize,
    sections: [4]usize,
    freq_base: f32,
) void {
    const n_dims = values.len;
    std.debug.assert(n_dims % 2 == 0);
    const half = n_dims / 2;
    const sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    if (sect_dims == 0) return;

    const position_f = @as(f32, @floatFromInt(position));
    const n_dims_f = @as(f32, @floatFromInt(n_dims));
    const theta_scale = std.math.pow(f32, freq_base, -2.0 / n_dims_f);

    var theta_t = position_f;
    var theta_h = position_f;
    var theta_w = position_f;
    var theta_e: f32 = 0;

    for (0..half) |pair_idx| {
        const sector = pair_idx % sect_dims;
        var theta = theta_t;
        if (sector % 3 == 1 and sector < 3 * sections[1]) {
            theta = theta_h;
        } else if (sector % 3 == 2 and sector < 3 * sections[2]) {
            theta = theta_w;
        } else if (sector % 3 == 0 and sector < 3 * sections[0]) {
            theta = theta_t;
        } else {
            theta = theta_e;
        }

        const cos_theta = @as(f32, @floatCast(std.math.cos(theta)));
        const sin_theta = @as(f32, @floatCast(std.math.sin(theta)));
        const x0 = values[pair_idx];
        const x1 = values[half + pair_idx];
        values[pair_idx] = x0 * cos_theta - x1 * sin_theta;
        values[half + pair_idx] = x0 * sin_theta + x1 * cos_theta;

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }
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

fn softmaxInto(
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

fn softmaxSliceInPlace(values: []f32) void {
    if (values.len == 0) return;

    var max_value = values[0];
    for (values[1..]) |value| {
        max_value = @max(max_value, value);
    }

    var sum: f64 = 0;
    for (values) |*value| {
        value.* = @as(f32, @floatCast(std.math.exp(@as(f64, value.* - max_value))));
        sum += value.*;
    }

    const inv_sum = @as(f32, @floatCast(1.0 / sum));
    for (values) |*value| {
        value.* *= inv_sum;
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

fn dot(lhs: []const f32, rhs: []const f32) f32 {
    std.debug.assert(lhs.len == rhs.len);
    var sum: f32 = 0;
    for (lhs, rhs) |a, b| {
        sum += a * b;
    }
    return sum;
}

fn copyAdd(dst: []f32, lhs: []const f32, rhs: []const f32) void {
    std.debug.assert(dst.len == lhs.len and lhs.len == rhs.len);
    for (dst, lhs, rhs) |*out, a, b| {
        out.* = a + b;
    }
}

fn addInPlace(dst: []f32, src: []const f32) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*value, addend| {
        value.* += addend;
    }
}

fn sigmoid(value: f32) f32 {
    if (value >= 0) {
        const exp_neg = @as(f32, @floatCast(std.math.exp(@as(f64, -value))));
        return 1.0 / (1.0 + exp_neg);
    }
    const exp_pos = @as(f32, @floatCast(std.math.exp(@as(f64, value))));
    return exp_pos / (1.0 + exp_pos);
}

fn softplus(value: f32) f32 {
    if (value > 20) return value;
    if (value < -20) return @as(f32, @floatCast(std.math.exp(@as(f64, value))));
    return @as(f32, @floatCast(std.math.log1p(std.math.exp(@as(f64, value)))));
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

test "text IMRoPE keeps extra sector unrotated at pos zero" {
    var values = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    applyTextImropeNeoxInPlace(&values, 0, .{ 1, 1, 1, 1 }, 10000.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1), values[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 5), values[4], 0.0001);
}

test "softmax in place normalizes" {
    var values = [_]f32{ 3.0, 1.0, -2.0 };
    softmaxSliceInPlace(&values);

    var sum: f32 = 0;
    for (values) |value| sum += value;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.0001);
    try std.testing.expect(values[0] > values[1]);
    try std.testing.expect(values[1] > values[2]);
}
