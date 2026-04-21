#!/usr/bin/env bash
set -euo pipefail

variant="${1:-Q4_K_M}"
repo="${HF_QWEN36_GGUF_REPO:-batiai/Qwen3.6-35B-A3B-GGUF}"
workers="${HF_MAX_WORKERS:-4}"
filename="Qwen-Qwen3.6-35B-A3B-${variant}.gguf"
target_dir="${2:-models/qwen3.6-35b-a3b-${variant,,}}"

mkdir -p "${target_dir}"
hf download "${repo}" "${filename}" --local-dir "${target_dir}" --max-workers "${workers}"
