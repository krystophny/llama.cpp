#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_BENCH_BIN="${LLAMA_BENCH_BIN:-${ROOT_DIR}/build/bin/llama-bench}"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/llama.cpp-apple-prefill/$(date +%Y%m%d-%H%M%S)}"

HF_REPO="${HF_REPO:-lmstudio-community/Qwen3.5-4B-GGUF}"
QUANTS="${QUANTS:-Q8_0 Q6_K Q5_K_M Q4_K_M}"
PROMPTS="${PROMPTS:-1024 4096 8192 16384}"
FLASH_ATTN_VALUES="${FLASH_ATTN_VALUES:-1 0}"
BATCHES="${BATCHES:-2048 1024}"
UBATCHES="${UBATCHES:-512 256}"
REPETITIONS="${REPETITIONS:-3}"
THREADS="${THREADS:-8}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
PRIO="${PRIO:--1}"
VARIANT="${VARIANT:-baseline}"
TENSOR_MODE="${TENSOR_MODE:-default}"
LIST_DEVICES="${LIST_DEVICES:-1}"
NO_WARMUP="${NO_WARMUP:-1}"

if [[ ! -x "${LLAMA_BENCH_BIN}" ]]; then
    echo "missing llama-bench binary: ${LLAMA_BENCH_BIN}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}/raw"

case "${TENSOR_MODE}" in
    default)
        unset GGML_METAL_TENSOR_ENABLE || true
        unset GGML_METAL_TENSOR_DISABLE || true
        ;;
    enable)
        export GGML_METAL_TENSOR_ENABLE=1
        unset GGML_METAL_TENSOR_DISABLE || true
        ;;
    disable)
        export GGML_METAL_TENSOR_DISABLE=1
        unset GGML_METAL_TENSOR_ENABLE || true
        ;;
    *)
        echo "unknown TENSOR_MODE: ${TENSOR_MODE} (expected default|enable|disable)" >&2
        exit 1
        ;;
esac

{
    echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root_dir=${ROOT_DIR}"
    echo "llama_bench_bin=${LLAMA_BENCH_BIN}"
    echo "hf_repo=${HF_REPO}"
    echo "variant=${VARIANT}"
    echo "tensor_mode=${TENSOR_MODE}"
    echo "quants=${QUANTS}"
    echo "prompts=${PROMPTS}"
    echo "flash_attn_values=${FLASH_ATTN_VALUES}"
    echo "batches=${BATCHES}"
    echo "ubatches=${UBATCHES}"
    echo "repetitions=${REPETITIONS}"
    echo "threads=${THREADS}"
    echo "n_gpu_layers=${N_GPU_LAYERS}"
    echo "prio=${PRIO}"
    echo "git_head=$(git -C "${ROOT_DIR}" rev-parse HEAD)"
    echo "git_branch=$(git -C "${ROOT_DIR}" branch --show-current)"
    echo "uname=$(uname -a)"
    echo "sysctl_model=$(sysctl -n hw.model 2>/dev/null || true)"
    echo "sysctl_memsize=$(sysctl -n hw.memsize 2>/dev/null || true)"
} > "${OUT_DIR}/meta.env"

if [[ "${LIST_DEVICES}" == "1" ]]; then
    "${LLAMA_BENCH_BIN}" --list-devices > "${OUT_DIR}/devices.txt" 2>&1 || true
fi

for quant in ${QUANTS}; do
    for prompt in ${PROMPTS}; do
        for fa in ${FLASH_ATTN_VALUES}; do
            for batch in ${BATCHES}; do
                for ubatch in ${UBATCHES}; do
                    run_id="${VARIANT}__${quant}__p${prompt}__fa${fa}__b${batch}__ub${ubatch}"
                    stdout_file="${OUT_DIR}/raw/${run_id}.jsonl"
                    stderr_file="${OUT_DIR}/raw/${run_id}.stderr.log"

                    args=(
                        -hf "${HF_REPO}:${quant}"
                        -p "${prompt}"
                        -n 0
                        -r "${REPETITIONS}"
                        -t "${THREADS}"
                        -ngl "${N_GPU_LAYERS}"
                        -b "${batch}"
                        -ub "${ubatch}"
                        -fa "${fa}"
                        --prio "${PRIO}"
                        -o jsonl
                    )

                    if [[ "${NO_WARMUP}" == "1" ]]; then
                        args+=(--no-warmup)
                    fi

                    echo "==> ${run_id}"
                    "${LLAMA_BENCH_BIN}" "${args[@]}" > "${stdout_file}" 2> "${stderr_file}"
                done
            done
        done
    done
done

python3 "${ROOT_DIR}/scripts/apple/summarize-prefill-jsonl.py" \
    --input-dir "${OUT_DIR}/raw" \
    --csv "${OUT_DIR}/summary.csv" \
    --markdown "${OUT_DIR}/summary.md"

echo "results written to ${OUT_DIR}"
