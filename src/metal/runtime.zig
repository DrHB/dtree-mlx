const builtin = @import("builtin");

const runtime_impl = if (builtin.os.tag == .macos)
    @import("runtime_macos.zig")
else
    @import("runtime_stub.zig");

pub const Error = runtime_impl.Error;
pub const Report = runtime_impl.Report;
pub const BenchmarkResult = runtime_impl.BenchmarkResult;
pub const MatVecBenchmarkResult = runtime_impl.MatVecBenchmarkResult;
pub const DenseBuffer = runtime_impl.DenseBuffer;
pub const RawBuffer = runtime_impl.RawBuffer;
pub const DenseContext = runtime_impl.DenseContext;
pub const runBootstrap = runtime_impl.runBootstrap;
pub const runBenchmark = runtime_impl.runBenchmark;
pub const runMatVecBenchmark = runtime_impl.runMatVecBenchmark;
