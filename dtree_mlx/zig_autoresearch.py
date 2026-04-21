from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from .history import append_rows, run_metadata


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
SUMMARY_MD = EXPERIMENTS_DIR / "summary.md"
PLOT_SVG = EXPERIMENTS_DIR / "results.svg"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
ZIG_BINARY = REPO_ROOT / "zig-out" / "bin" / "dtree-mlx-zig"


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    title: str
    metric_key: str
    unit: str
    color: str
    command: list[str]
    bench_rows: int | None = None
    bench_iters: int | None = None
    bench_warmup: int | None = None
    decode_steps: int | None = None


SUITE_ORDER = [
    "logits_matvec",
    "blk0_qkv_projection",
    "full_token_pass",
    "cached_decode",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical pure-Zig benchmark suite and update experiment reports.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to the GGUF model to benchmark.",
    )
    parser.add_argument(
        "--profile",
        choices=["full", "decode", "micro"],
        default="full",
        help="Benchmark subset to run. full = micro + decode.",
    )
    parser.add_argument(
        "--token-id",
        type=int,
        default=42,
        help="Tokenizer-free synthetic token id used by the current pure-Zig harness.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Short free-form note describing the change being measured.",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional short label for the run. Falls back to notes, then the git subject.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse the existing zig binary instead of running `zig build` first.",
    )
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="Rebuild `experiments/summary.md` and `experiments/results.svg` from the CSV without benchmarking.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=RESULTS_CSV,
        help="CSV path for benchmark history.",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=SUMMARY_MD,
        help="Markdown summary path.",
    )
    parser.add_argument(
        "--plot-svg",
        type=Path,
        default=PLOT_SVG,
        help="SVG plot output path.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory for per-run raw JSON artifacts.",
    )
    return parser.parse_args(argv)


def build_suite_specs(model: str, token_id: int, profile: str) -> list[SuiteSpec]:
    binary = str(ZIG_BINARY)
    specs = OrderedDict(
        [
            (
                "logits_matvec",
                SuiteSpec(
                    name="logits_matvec",
                    title="Logits Head Matvec",
                    metric_key="full_tensor_matvecs_per_s",
                    unit="matvec/s",
                    color="#ff6b6b",
                    command=[
                        binary,
                        "--model",
                        model,
                        "--tensor",
                        "output.weight",
                        "--matvec",
                        "--bench",
                        "--bench-rows",
                        "248320",
                        "--bench-iters",
                        "1",
                        "--bench-warmup",
                        "0",
                    ],
                    bench_rows=248320,
                    bench_iters=1,
                    bench_warmup=0,
                ),
            ),
            (
                "blk0_qkv_projection",
                SuiteSpec(
                    name="blk0_qkv_projection",
                    title="Block 0 QKV Projection",
                    metric_key="full_projection_passes_per_s",
                    unit="projection/s",
                    color="#f7b801",
                    command=[
                        binary,
                        "--model",
                        model,
                        "--token-id",
                        str(token_id),
                        "--norm-tensor",
                        "blk.0.attn_norm.weight",
                        "--project-tensor",
                        "blk.0.attn_qkv.weight",
                        "--bench",
                        "--bench-rows",
                        "8192",
                        "--bench-iters",
                        "8",
                        "--bench-warmup",
                        "1",
                    ],
                    bench_rows=8192,
                    bench_iters=8,
                    bench_warmup=1,
                ),
            ),
            (
                "full_token_pass",
                SuiteSpec(
                    name="full_token_pass",
                    title="Fresh Full Token Pass",
                    metric_key="fresh_token_tok_per_s",
                    unit="tok/s",
                    color="#2ec4b6",
                    command=[
                        binary,
                        "--model",
                        model,
                        "--full-token-pass",
                        "--token-id",
                        str(token_id),
                        "--bench",
                        "--bench-iters",
                        "2",
                        "--bench-warmup",
                        "1",
                    ],
                    bench_iters=2,
                    bench_warmup=1,
                ),
            ),
            (
                "cached_decode",
                SuiteSpec(
                    name="cached_decode",
                    title="Cached Decode",
                    metric_key="cached_decode_tok_per_s",
                    unit="tok/s",
                    color="#3a86ff",
                    command=[
                        binary,
                        "--model",
                        model,
                        "--cached-decode",
                        "--token-id",
                        str(token_id),
                        "--bench",
                        "--bench-iters",
                        "2",
                        "--bench-warmup",
                        "1",
                    ],
                    bench_iters=2,
                    bench_warmup=1,
                ),
            ),
        ]
    )
    if profile == "micro":
        names = ["logits_matvec", "blk0_qkv_projection"]
    elif profile == "decode":
        names = ["full_token_pass", "cached_decode"]
    else:
        names = SUITE_ORDER
    return [specs[name] for name in names]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.results_csv = args.results_csv.resolve()
    args.summary_md = args.summary_md.resolve()
    args.plot_svg = args.plot_svg.resolve()
    args.runs_dir = args.runs_dir.resolve()

    if args.reports_only:
        rebuild_reports(args.results_csv, args.summary_md, args.plot_svg)
        print(f"updated reports from {args.results_csv}")
        return

    specs = build_suite_specs(args.model, args.token_id, args.profile)
    git_meta = current_git_metadata()
    label = args.label.strip() or args.notes.strip() or git_meta["git_subject"]
    meta = {
        **run_metadata("zig_autoresearch.py", experiment_tag=args.profile),
        **git_meta,
        "label": label,
        "notes": args.notes.strip(),
    }
    timestamp = meta["timestamp_utc"]
    run_id = make_run_id(timestamp, meta["git_short_commit"], args.profile, meta["git_dirty"])

    if not args.skip_build:
        run_checked(["zig", "build"], step_name="zig build")
    elif not ZIG_BINARY.exists():
        raise SystemExit(f"--skip-build was used but {ZIG_BINARY} does not exist.")

    run_rows: list[dict[str, Any]] = []
    artifact_steps: list[dict[str, Any]] = []
    for spec in specs:
        completed = run_checked(spec.command, step_name=spec.name)
        metrics = parse_key_value_output(completed.stdout)
        if spec.metric_key not in metrics:
            raise SystemExit(
                f"{spec.name} did not emit {spec.metric_key}. stdout was:\n{completed.stdout}"
            )
        row = {
            "timestamp_utc": timestamp,
            "run_id": run_id,
            "profile": args.profile,
            "label": label,
            "notes": args.notes.strip(),
            "suite": spec.name,
            "suite_title": spec.title,
            "metric_key": spec.metric_key,
            "metric_value": float(metrics[spec.metric_key]),
            "metric_unit": spec.unit,
            "elapsed_s": metric_number(metrics, "elapsed_s"),
            "checksum": metric_number(metrics, "checksum"),
            "model": args.model,
            "token_id": args.token_id,
            "bench_rows": spec.bench_rows,
            "bench_iters": spec.bench_iters,
            "bench_warmup": spec.bench_warmup,
            "decode_steps": spec.decode_steps,
            "command": spec.command,
            "metrics_json": metrics,
            "git_branch": meta["git_branch"],
            "git_commit": meta["git_commit"],
            "git_short_commit": meta["git_short_commit"],
            "git_dirty": meta["git_dirty"],
            "git_subject": meta["git_subject"],
        }
        run_rows.append(row)
        artifact_steps.append(
            {
                "suite": spec.name,
                "title": spec.title,
                "metric_key": spec.metric_key,
                "metric_value": row["metric_value"],
                "metric_unit": spec.unit,
                "command": spec.command,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
                "metrics": metrics,
            }
        )
        print(f"{spec.name}: {format_metric(row['metric_value'])} {spec.unit}")

    append_rows(args.results_csv, run_rows)
    write_run_artifact(
        args.runs_dir / f"{run_id}.json",
        {
            "run_id": run_id,
            "timestamp_utc": timestamp,
            "profile": args.profile,
            "label": label,
            "notes": args.notes.strip(),
            "model": args.model,
            "token_id": args.token_id,
            "metadata": meta,
            "steps": artifact_steps,
        },
    )
    rebuild_reports(args.results_csv, args.summary_md, args.plot_svg)

    print(f"results_csv: {args.results_csv}")
    print(f"summary_md: {args.summary_md}")
    print(f"plot_svg: {args.plot_svg}")


def run_checked(command: list[str], step_name: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"{step_name} failed with exit code {completed.returncode}\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def parse_key_value_output(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for raw_line in stdout.splitlines():
        if ":" not in raw_line:
            continue
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            continue
        metrics[key] = parse_scalar(raw_value)
    return metrics


def parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if raw.startswith("0") and raw not in {"0", "0.0"} and not raw.startswith("0."):
            raise ValueError
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def metric_number(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def current_git_metadata() -> dict[str, Any]:
    status = git_output(["status", "--short"])
    return {
        "git_branch": git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "git_short_commit": git_output(["rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(status),
        "git_subject": git_output(["show", "-s", "--format=%s", "HEAD"]),
    }


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def make_run_id(timestamp_utc: str, short_commit: str, profile: str, dirty: bool) -> str:
    stamp = timestamp_utc.replace("+00:00", "Z").replace("-", "").replace(":", "")
    dirty_suffix = "-dirty" if dirty else ""
    short = short_commit or "nogit"
    return f"{stamp}-{short}-{profile}{dirty_suffix}"


def write_run_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def rebuild_reports(results_csv: Path, summary_md: Path, plot_svg: Path) -> None:
    rows = load_result_rows(results_csv)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    plot_svg.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(render_summary(rows))
    plot_svg.write_text(render_svg(rows))


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [decode_row(row) for row in reader]


def decode_row(row: dict[str, str]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in row.items():
        if value is None or value == "":
            decoded[key] = ""
            continue
        if value.startswith("{") or value.startswith("["):
            try:
                decoded[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        decoded[key] = parse_scalar(value)
    return decoded


def render_summary(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "# Zig Experiment Summary\n\nNo experiment results yet.\n"

    ordered_rows = sorted(rows, key=lambda row: str(row["timestamp_utc"]))
    runs = collect_runs(ordered_rows)
    latest = runs[-1]
    latest_by_suite = latest["suites"]
    previous_by_suite = previous_suite_rows(ordered_rows, latest["run_id"])
    best_by_suite = best_suite_rows(ordered_rows)

    lines = [
        "# Zig Experiment Summary",
        "",
        "This file is generated from `experiments/results.csv`.",
        "",
        "## Latest Run",
        "",
        f"- Run: `{latest['run_id']}`",
        f"- Time: `{latest['timestamp_utc']}`",
        f"- Commit: `{latest['git_short_commit']}` on `{latest['git_branch']}`",
        f"- Dirty tree: `{latest['git_dirty']}`",
        f"- Subject: {latest['git_subject']}",
    ]
    if latest["label"]:
        lines.append(f"- Label: {latest['label']}")
    if latest["notes"]:
        lines.append(f"- Notes: {latest['notes']}")

    lines.extend(
        [
            "",
            "## Latest Metrics",
            "",
            "| Suite | Metric | Value | Delta vs previous same suite |",
            "|---|---|---:|---:|",
        ]
    )
    for suite_name in SUITE_ORDER:
        row = latest_by_suite.get(suite_name)
        if not row:
            continue
        previous = previous_by_suite.get(suite_name)
        delta = ""
        if previous:
            delta_value = float(row["metric_value"]) - float(previous["metric_value"])
            delta = signed_metric(delta_value)
        lines.append(
            "| {suite} | `{metric}` | {value} {unit} | {delta} |".format(
                suite=row["suite_title"],
                metric=row["metric_key"],
                value=format_metric(row["metric_value"]),
                unit=row["metric_unit"],
                delta=delta or "n/a",
            )
        )

    lines.extend(
        [
            "",
            "## Best So Far",
            "",
            "| Suite | Best | Commit | Time | Label |",
            "|---|---:|---|---|---|",
        ]
    )
    for suite_name in SUITE_ORDER:
        row = best_by_suite.get(suite_name)
        if not row:
            continue
        label = row.get("label") or row.get("notes") or row.get("git_subject") or ""
        lines.append(
            "| {suite} | {value} {unit} | `{commit}` | `{time}` | {label} |".format(
                suite=row["suite_title"],
                value=format_metric(row["metric_value"]),
                unit=row["metric_unit"],
                commit=row["git_short_commit"],
                time=row["timestamp_utc"],
                label=label,
            )
        )

    recent_runs = runs[-10:]
    lines.extend(
        [
            "",
            "## Recent Runs",
            "",
            "| Run | Commit | Label | Cached tok/s | Fresh tok/s | QKV proj/s | Logits matvec/s |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for run in reversed(recent_runs):
        suites = run["suites"]
        label = run["label"] or run["notes"] or run["git_subject"] or ""
        lines.append(
            "| `{run_id}` | `{commit}` | {label} | {cached} | {fresh} | {qkv} | {matvec} |".format(
                run_id=run["run_id"],
                commit=run["git_short_commit"],
                label=label,
                cached=table_metric(suites.get("cached_decode")),
                fresh=table_metric(suites.get("full_token_pass")),
                qkv=table_metric(suites.get("blk0_qkv_projection")),
                matvec=table_metric(suites.get("logits_matvec")),
            )
        )

    return "\n".join(lines) + "\n"


def collect_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in rows:
        run_id = str(row["run_id"])
        if run_id not in runs:
            runs[run_id] = {
                "run_id": run_id,
                "timestamp_utc": row["timestamp_utc"],
                "git_branch": row.get("git_branch", ""),
                "git_short_commit": row.get("git_short_commit", ""),
                "git_dirty": row.get("git_dirty", ""),
                "git_subject": row.get("git_subject", ""),
                "label": row.get("label", ""),
                "notes": row.get("notes", ""),
                "suites": {},
            }
        runs[run_id]["suites"][str(row["suite"])] = row
    return list(runs.values())


def previous_suite_rows(
    rows: list[dict[str, Any]],
    latest_run_id: str,
) -> dict[str, dict[str, Any]]:
    previous: dict[str, dict[str, Any]] = {}
    for row in rows:
        suite = str(row["suite"])
        if str(row["run_id"]) == latest_run_id:
            continue
        previous[suite] = row
    return previous


def best_suite_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        suite = str(row["suite"])
        if suite not in best or float(row["metric_value"]) > float(best[suite]["metric_value"]):
            best[suite] = row
    return best


def table_metric(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    return format_metric(row["metric_value"])


def render_svg(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return empty_svg("No experiment results yet.")

    grouped: dict[str, list[dict[str, Any]]] = {suite: [] for suite in SUITE_ORDER}
    for row in sorted(rows, key=lambda item: str(item["timestamp_utc"])):
        suite = str(row["suite"])
        if suite in grouped:
            grouped[suite].append(row)

    active_suites = [suite for suite in SUITE_ORDER if grouped[suite]]
    if not active_suites:
        return empty_svg("No supported experiment rows found.")

    panel_width = 520
    panel_height = 220
    cols = 2 if len(active_suites) > 1 else 1
    rows_count = math.ceil(len(active_suites) / cols)
    outer_margin = 28
    header_height = 88
    width = outer_margin * 2 + panel_width * cols + 24 * (cols - 1)
    height = header_height + outer_margin + rows_count * panel_height + 24 * (rows_count - 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#09111b"/>',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }",
        ".muted { fill: #7b8ba1; font-size: 12px; }",
        ".title { fill: #ecf3ff; font-size: 24px; font-weight: 700; }",
        ".panel-title { fill: #ecf3ff; font-size: 15px; font-weight: 700; }",
        ".panel-meta { fill: #a8b6ca; font-size: 12px; }",
        ".axis-label { fill: #7b8ba1; font-size: 11px; }",
        "</style>",
        '<text class="title" x="28" y="36">Pure-Zig Optimization History</text>',
        f'<text class="muted" x="28" y="60">Generated from experiments/results.csv on {escape(str(rows[-1]["timestamp_utc"]))}</text>',
    ]

    for index, suite_name in enumerate(active_suites):
        panel_x = outer_margin + (index % cols) * (panel_width + 24)
        panel_y = header_height + (index // cols) * (panel_height + 24)
        spec_row = grouped[suite_name][-1]
        color = suite_color(suite_name)
        parts.append(draw_panel(grouped[suite_name], panel_x, panel_y, panel_width, panel_height, color, spec_row))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def draw_panel(
    rows: list[dict[str, Any]],
    x: int,
    y: int,
    width: int,
    height: int,
    color: str,
    latest_row: dict[str, Any],
) -> str:
    left = x + 54
    right = x + width - 18
    top = y + 48
    bottom = y + height - 38
    values = [float(row["metric_value"]) for row in rows]
    ymin, ymax = value_bounds(values)

    line_points: list[str] = []
    circle_parts: list[str] = []
    point_count = len(rows)
    for idx, row in enumerate(rows):
        px = left if point_count == 1 else left + (right - left) * idx / (point_count - 1)
        py = bottom - ((float(row["metric_value"]) - ymin) / max(ymax - ymin, 1e-12)) * (bottom - top)
        line_points.append(f"{px:.2f},{py:.2f}")
        if idx == point_count - 1:
            radius = 4.5
            stroke = "#ffffff"
            stroke_width = 1.5
        else:
            radius = 3.0
            stroke = "#09111b"
            stroke_width = 1.0
        circle_parts.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius}" fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    y_ticks = 4
    grid_parts = []
    for tick_idx in range(y_ticks + 1):
        frac = tick_idx / y_ticks
        py = bottom - frac * (bottom - top)
        tick_value = ymin + frac * (ymax - ymin)
        grid_parts.append(f'<line x1="{left}" y1="{py:.2f}" x2="{right}" y2="{py:.2f}" stroke="#1d3148" stroke-width="1"/>')
        grid_parts.append(
            f'<text class="axis-label" x="{x + 8}" y="{py + 4:.2f}">{escape(format_metric(tick_value))}</text>'
        )

    latest_value = format_metric(latest_row["metric_value"])
    best_value = format_metric(max(values))
    latest_commit = escape(str(latest_row.get("git_short_commit", "")))
    latest_time = escape(str(latest_row.get("timestamp_utc", "")))
    unit = escape(str(latest_row.get("metric_unit", "")))
    title = escape(str(latest_row.get("suite_title", latest_row.get("suite", ""))))
    points = " ".join(line_points)

    return "\n".join(
        [
            "<g>",
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="#0f1b2b" stroke="#1d3148" stroke-width="1"/>',
            f'<text class="panel-title" x="{x + 18}" y="{y + 28}">{title}</text>',
            f'<text class="panel-meta" x="{x + 18}" y="{y + 44}">latest {latest_value} {unit}   best {best_value} {unit}</text>',
            f'<text class="panel-meta" x="{x + width - 18}" y="{y + 28}" text-anchor="end">{latest_commit}</text>',
            f'<text class="panel-meta" x="{x + width - 18}" y="{y + 44}" text-anchor="end">{latest_time}</text>',
            *grid_parts,
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>',
            *circle_parts,
            f'<text class="axis-label" x="{left}" y="{y + height - 12}">oldest</text>',
            f'<text class="axis-label" x="{right}" y="{y + height - 12}" text-anchor="end">latest</text>',
            "</g>",
        ]
    )


def suite_color(suite_name: str) -> str:
    colors = {
        "logits_matvec": "#ff6b6b",
        "blk0_qkv_projection": "#f7b801",
        "full_token_pass": "#2ec4b6",
        "cached_decode": "#3a86ff",
    }
    return colors[suite_name]


def value_bounds(values: list[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        if high >= 0:
            return 0.0, max(high * 1.2, 1.0)
        return min(low * 1.2, -1.0), 0.0
    pad = (high - low) * 0.12
    padded_low = low - pad
    padded_high = high + pad
    if low >= 0:
        padded_low = max(0.0, padded_low)
    if high <= 0:
        padded_high = min(0.0, padded_high)
    return padded_low, padded_high


def empty_svg(message: str) -> str:
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="860" height="220" viewBox="0 0 860 220">',
            '<rect width="100%" height="100%" fill="#09111b"/>',
            '<text x="32" y="52" fill="#ecf3ff" font-size="24" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" font-weight="700">Pure-Zig Optimization History</text>',
            f'<text x="32" y="94" fill="#7b8ba1" font-size="15" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">{escape(message)}</text>',
            "</svg>",
            "",
        ]
    )


def format_metric(value: Any) -> str:
    number = float(value)
    if abs(number) >= 100:
        return f"{number:.2f}"
    if abs(number) >= 10:
        return f"{number:.3f}"
    if abs(number) >= 1:
        return f"{number:.4f}"
    return f"{number:.5f}"


def signed_metric(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{format_metric(abs(value))}"


if __name__ == "__main__":
    main()
