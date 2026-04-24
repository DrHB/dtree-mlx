from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "bench_qwen36_llamacpp.py"
    namespace = runpy.run_path(str(script))
    namespace["main"]()
