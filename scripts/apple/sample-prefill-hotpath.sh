#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_BENCH_BIN="${LLAMA_BENCH_BIN:-${ROOT_DIR}/build/bin/llama-bench}"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/llama.cpp-apple-prefill-samples/$(date +%Y%m%d-%H%M%S)}"

PID="${PID:-}"
SAMPLE_SECONDS="${SAMPLE_SECONDS:-5}"

HF_REPO="${HF_REPO:-lmstudio-community/Qwen3.5-4B-GGUF}"
QUANT="${QUANT:-Q4_K_M}"
PROMPT="${PROMPT:-8192}"
THREADS="${THREADS:-8}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
FLASH_ATTN="${FLASH_ATTN:-1}"
PRIO="${PRIO:--1}"
SPAWN_BENCH="${SPAWN_BENCH:-0}"

mkdir -p "${OUT_DIR}"

cleanup() {
    if [[ -n "${bench_pid:-}" ]]; then
        kill "${bench_pid}" >/dev/null 2>&1 || true
        wait "${bench_pid}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [[ "${SPAWN_BENCH}" == "1" ]]; then
    if [[ ! -x "${LLAMA_BENCH_BIN}" ]]; then
        echo "missing llama-bench binary: ${LLAMA_BENCH_BIN}" >&2
        exit 1
    fi

    "${LLAMA_BENCH_BIN}" \
        -hf "${HF_REPO}:${QUANT}" \
        -p "${PROMPT}" \
        -n 0 \
        -r 1 \
        -t "${THREADS}" \
        -ngl "${N_GPU_LAYERS}" \
        -b "${BATCH}" \
        -ub "${UBATCH}" \
        -fa "${FLASH_ATTN}" \
        --prio "${PRIO}" \
        --no-warmup \
        -o jsonl \
        > "${OUT_DIR}/bench.jsonl" \
        2> "${OUT_DIR}/bench.stderr.log" &
    bench_pid=$!
    PID="${bench_pid}"
    sleep 2
fi

if [[ -z "${PID}" ]]; then
    echo "set PID=<live pid> or SPAWN_BENCH=1" >&2
    exit 1
fi

sample_file="${OUT_DIR}/sample-pid-${PID}.txt"
sample "${PID}" "${SAMPLE_SECONDS}" -mayDie -file "${sample_file}" >/dev/null

python3 - "${sample_file}" "${OUT_DIR}/hotspots.txt" <<'PY'
from __future__ import annotations
import re
import sys
from collections import Counter
from pathlib import Path

sample_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
text = sample_path.read_text(errors="replace")

patterns = {
    "waitUntilCompleted": r"waitUntilCompleted",
    "ggml_metal_synchronize": r"ggml_metal_synchronize",
    "ggml_metal_graph_compute": r"ggml_metal_graph_compute",
    "ggml_metal_op_mul_mat": r"ggml_metal_op_mul_mat",
    "ggml_metal_op_unary": r"ggml_metal_op_unary",
    "ggml_metal_op_encode": r"ggml_metal_op_encode",
    "llama_context::decode": r"llama_context::decode",
    "kernel_mul_mm": r"kernel_mul_mm",
}

counts = Counter()
for name, pat in patterns.items():
    counts[name] = len(re.findall(pat, text))

with out_path.open("w") as f:
    for name, count in counts.most_common():
        f.write(f"{name},{count}\n")
PY

echo "sample written to ${sample_file}"
echo "hotspot summary written to ${OUT_DIR}/hotspots.txt"
