const std = @import("std");
const cached_decode = @import("cached_decode.zig");
const gguf_store = @import("gguf_store.zig");
const metal_backend = @import("metal_backend.zig");
const metal_loader = @import("metal_loader.zig");

pub const ParityAgainst = enum {
    none,
    cpu,
    metal,
};

pub const ProfileSummary = struct {
    decode_backend: []const u8,
    metal_device_name: []const u8,
    metal_project_calls: usize,
    metal_project_hits: usize,
    metal_rowrange_calls: usize,
    metal_rowrange_hits: usize,
    cpu_fallback_calls: usize,
    metal_cache_entries: usize,
    metal_cache_bytes: usize,
};

pub const TraceBenchmarkResult = struct {
    elapsed_s: f64,
    steady_decode_tok_per_s: f64,
    first_decode_token_ms: f64,
    median_decode_token_ms: f64,
    p95_decode_token_ms: f64,
    checksum: f64,
    prompt_tokens: usize,
    timed_tokens: usize,
    argmax_sequence_hash: u64,
    topk_logit_hash: u64,
    argmax_mismatch_count: usize,
    max_topk_logit_delta: f32,
    mean_topk_logit_delta: f32,
    final_argmax_token_id: usize,
    final_argmax_logit: f32,
    top_count: usize,
    profile: ProfileSummary,
};

const TraceCapture = struct {
    argmax_ids: []usize,
    latencies_ns: []u64,
    final_candidates: []cached_decode.OutputCandidate,
    final_top_count: usize,
    final_argmax_token_id: usize,
    final_argmax_logit: f32,
    checksum: f64,
    total_elapsed_ns: u64,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    store: *const gguf_store.Store,
    loader: metal_loader.Loader,
    backend: ?*metal_backend.Backend,
    core: cached_decode.Engine,

    pub fn init(
        allocator: std.mem.Allocator,
        store: *const gguf_store.Store,
        enable_metal: bool,
        max_seq_len: usize,
    ) !Engine {
        var loader = try metal_loader.Loader.init(allocator, store);
        errdefer loader.deinit();

        var backend: ?*metal_backend.Backend = null;
        errdefer if (backend) |owned| {
            owned.deinit();
            allocator.destroy(owned);
        };
        if (enable_metal) {
            const owned = try allocator.create(metal_backend.Backend);
            errdefer allocator.destroy(owned);
            owned.* = try metal_backend.Backend.init(allocator, metal_backend.default_max_cache_bytes);
            backend = owned;
        }

        var engine = Engine{
            .allocator = allocator,
            .store = store,
            .loader = loader,
            .backend = backend,
            .core = undefined,
        };
        engine.core = try cached_decode.Engine.init(
            allocator,
            store,
            engine.backend,
            max_seq_len,
        );
        return engine;
    }

    pub fn deinit(self: *Engine) void {
        self.core.deinit();
        if (self.backend) |owned| {
            owned.deinit();
            self.allocator.destroy(owned);
        }
        self.loader.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *Engine) void {
        self.core.reset();
        if (self.backend) |owned| owned.resetStats();
    }

    pub fn decodeStep(
        self: *Engine,
        token_id: usize,
        top_out: []cached_decode.OutputCandidate,
    ) !cached_decode.RunResult {
        return self.core.step(token_id, top_out);
    }

    pub fn readLogits(self: *const Engine) ?[]const f32 {
        return self.core.output_logits;
    }

    pub fn profileSummary(self: *const Engine) ProfileSummary {
        if (self.backend) |owned| {
            const stats = owned.snapshotStats();
            return .{
                .decode_backend = "metal-cache",
                .metal_device_name = owned.deviceName(),
                .metal_project_calls = stats.metal_project_calls,
                .metal_project_hits = stats.metal_project_hits,
                .metal_rowrange_calls = stats.metal_rowrange_calls,
                .metal_rowrange_hits = stats.metal_rowrange_hits,
                .cpu_fallback_calls = stats.cpu_fallback_calls,
                .metal_cache_entries = stats.metal_cache_entries,
                .metal_cache_bytes = stats.metal_cache_bytes,
            };
        }

        return .{
            .decode_backend = "cpu",
            .metal_device_name = "cpu",
            .metal_project_calls = 0,
            .metal_project_hits = 0,
            .metal_rowrange_calls = 0,
            .metal_rowrange_hits = 0,
            .cpu_fallback_calls = 0,
            .metal_cache_entries = 0,
            .metal_cache_bytes = 0,
        };
    }

    pub fn benchmarkTrace(
        self: *Engine,
        token_seq: []const usize,
        prompt_tokens: usize,
        timed_tokens: usize,
        top_limit: usize,
        parity_against: ParityAgainst,
    ) !TraceBenchmarkResult {
        if (timed_tokens == 0) return error.InvalidBenchIters;
        if (prompt_tokens + timed_tokens > token_seq.len) return error.TokenSequenceTooShort;
        if (prompt_tokens + timed_tokens > self.core.max_seq_len) return error.DecodeCacheFull;

        const timed_capture = try self.runTrace(token_seq, prompt_tokens, timed_tokens, top_limit, true);
        defer {
            self.allocator.free(timed_capture.argmax_ids);
            self.allocator.free(timed_capture.latencies_ns);
            self.allocator.free(timed_capture.final_candidates);
        }

        var argmax_mismatch_count: usize = 0;
        var max_topk_logit_delta: f32 = 0;
        var mean_topk_logit_delta: f32 = 0;

        if (parity_against != .none) {
            var reference = try Engine.init(
                self.allocator,
                self.store,
                parity_against == .metal,
                self.core.max_seq_len,
            );
            defer reference.deinit();

            const parity_capture = try reference.runTrace(token_seq, prompt_tokens, timed_tokens, top_limit, false);
            defer {
                self.allocator.free(parity_capture.argmax_ids);
                self.allocator.free(parity_capture.latencies_ns);
                self.allocator.free(parity_capture.final_candidates);
            }

            argmax_mismatch_count = countArgmaxMismatches(timed_capture.argmax_ids, parity_capture.argmax_ids);
            const deltas = compareCandidateLogits(
                timed_capture.final_candidates[0..timed_capture.final_top_count],
                parity_capture.final_candidates[0..parity_capture.final_top_count],
            );
            max_topk_logit_delta = deltas.max_delta;
            mean_topk_logit_delta = deltas.mean_delta;
        }

        const first_decode_token_ms = nsToMs(timed_capture.latencies_ns[0]);
        const median_decode_token_ms = percentileNsMs(self.allocator, timed_capture.latencies_ns, 50) catch first_decode_token_ms;
        const p95_decode_token_ms = percentileNsMs(self.allocator, timed_capture.latencies_ns, 95) catch first_decode_token_ms;
        const elapsed_s = @as(f64, @floatFromInt(timed_capture.total_elapsed_ns)) / @as(f64, std.time.ns_per_s);

        return .{
            .elapsed_s = elapsed_s,
            .steady_decode_tok_per_s = @as(f64, @floatFromInt(timed_tokens)) / elapsed_s,
            .first_decode_token_ms = first_decode_token_ms,
            .median_decode_token_ms = median_decode_token_ms,
            .p95_decode_token_ms = p95_decode_token_ms,
            .checksum = timed_capture.checksum,
            .prompt_tokens = prompt_tokens,
            .timed_tokens = timed_tokens,
            .argmax_sequence_hash = hashArgmaxSequence(timed_capture.argmax_ids),
            .topk_logit_hash = hashCandidates(timed_capture.final_candidates[0..timed_capture.final_top_count]),
            .argmax_mismatch_count = argmax_mismatch_count,
            .max_topk_logit_delta = max_topk_logit_delta,
            .mean_topk_logit_delta = mean_topk_logit_delta,
            .final_argmax_token_id = timed_capture.final_argmax_token_id,
            .final_argmax_logit = timed_capture.final_argmax_logit,
            .top_count = timed_capture.final_top_count,
            .profile = self.profileSummary(),
        };
    }

    fn runTrace(
        self: *Engine,
        token_seq: []const usize,
        prompt_tokens: usize,
        timed_tokens: usize,
        top_limit: usize,
        measure_latency: bool,
    ) !TraceCapture {
        const candidate_count = @max(@as(usize, 1), top_limit);
        const argmax_ids = try self.allocator.alloc(usize, timed_tokens);
        errdefer self.allocator.free(argmax_ids);
        const latencies_ns = try self.allocator.alloc(u64, timed_tokens);
        errdefer self.allocator.free(latencies_ns);
        const final_candidates = try self.allocator.alloc(cached_decode.OutputCandidate, candidate_count);
        errdefer self.allocator.free(final_candidates);

        self.reset();
        var checksum: f64 = 0;
        var sink: [1]cached_decode.OutputCandidate = undefined;

        for (token_seq[0..prompt_tokens]) |token_id| {
            const result = try self.decodeStep(token_id, sink[0..0]);
            checksum += result.argmax_logit;
        }

        var total_elapsed_ns: u64 = 0;
        var final_top_count: usize = 0;
        var final_argmax_token_id: usize = 0;
        var final_argmax_logit: f32 = 0;

        for (0..timed_tokens) |idx| {
            const capture_slice = if (idx + 1 == timed_tokens) final_candidates else sink[0..0];
            const start_ns = if (measure_latency) try monotonicNowNs() else 0;
            const result = try self.decodeStep(token_seq[prompt_tokens + idx], capture_slice);
            const end_ns = if (measure_latency) try monotonicNowNs() else 0;

            if (measure_latency) {
                latencies_ns[idx] = end_ns - start_ns;
                total_elapsed_ns += latencies_ns[idx];
            } else {
                latencies_ns[idx] = 0;
            }

            checksum += result.argmax_logit;
            argmax_ids[idx] = result.argmax_token_id;
            final_top_count = result.top_count;
            final_argmax_token_id = result.argmax_token_id;
            final_argmax_logit = result.argmax_logit;
        }

        return .{
            .argmax_ids = argmax_ids,
            .latencies_ns = latencies_ns,
            .final_candidates = final_candidates,
            .final_top_count = final_top_count,
            .final_argmax_token_id = final_argmax_token_id,
            .final_argmax_logit = final_argmax_logit,
            .checksum = checksum,
            .total_elapsed_ns = total_elapsed_ns,
        };
    }
};

const LogitDeltaSummary = struct {
    max_delta: f32,
    mean_delta: f32,
};

fn countArgmaxMismatches(lhs: []const usize, rhs: []const usize) usize {
    std.debug.assert(lhs.len == rhs.len);
    var count: usize = 0;
    for (lhs, rhs) |a, b| {
        if (a != b) count += 1;
    }
    return count;
}

fn compareCandidateLogits(
    lhs: []const cached_decode.OutputCandidate,
    rhs: []const cached_decode.OutputCandidate,
) LogitDeltaSummary {
    var max_delta: f32 = 0;
    var total_delta: f64 = 0;
    var count: usize = 0;

    for (lhs) |candidate| {
        const rhs_logit = candidateLogit(rhs, candidate.token_id) orelse 0;
        const delta = @abs(candidate.logit - rhs_logit);
        max_delta = @max(max_delta, delta);
        total_delta += delta;
        count += 1;
    }
    for (rhs) |candidate| {
        if (candidateLogit(lhs, candidate.token_id) != null) continue;
        const delta = @abs(candidate.logit);
        max_delta = @max(max_delta, delta);
        total_delta += delta;
        count += 1;
    }

    return .{
        .max_delta = max_delta,
        .mean_delta = if (count == 0) 0 else @floatCast(total_delta / @as(f64, @floatFromInt(count))),
    };
}

fn candidateLogit(
    candidates: []const cached_decode.OutputCandidate,
    token_id: usize,
) ?f32 {
    for (candidates) |candidate| {
        if (candidate.token_id == token_id) return candidate.logit;
    }
    return null;
}

fn monotonicNowNs() !u64 {
    var ts: std.posix.timespec = undefined;
    if (std.c.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts) != 0) {
        return error.ClockGetTimeFailed;
    }

    const total_ns = @as(i128, ts.sec) * std.time.ns_per_s + @as(i128, ts.nsec);
    return std.math.cast(u64, total_ns) orelse error.ClockOutOfRange;
}

fn percentileNsMs(
    allocator: std.mem.Allocator,
    values_ns: []const u64,
    percentile: usize,
) !f64 {
    const copy = try allocator.dupe(u64, values_ns);
    defer allocator.free(copy);

    std.sort.heap(u64, copy, {}, std.sort.asc(u64));
    const idx = if (copy.len == 1)
        @as(usize, 0)
    else
        @min(copy.len - 1, ((copy.len - 1) * percentile) / 100);
    return nsToMs(copy[idx]);
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / @as(f64, std.time.ns_per_ms);
}

fn hashArgmaxSequence(values: []const usize) u64 {
    var hasher = std.hash.Wyhash.init(0);
    hasher.update(std.mem.sliceAsBytes(values));
    return hasher.final();
}

fn hashCandidates(candidates: []const cached_decode.OutputCandidate) u64 {
    var hasher = std.hash.Wyhash.init(0);
    for (candidates) |candidate| {
        hasher.update(std.mem.asBytes(&candidate.token_id));
        const bits: u32 = @bitCast(candidate.logit);
        hasher.update(std.mem.asBytes(&bits));
    }
    return hasher.final();
}
