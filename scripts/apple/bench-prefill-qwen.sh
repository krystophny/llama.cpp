#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_BENCH_BIN="${LLAMA_BENCH_BIN:-${ROOT_DIR}/build/bin/llama-bench}"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/llama.cpp-apple-prefill/$(date +%Y%m%d-%H%M%S)}"

MODEL_LABEL="${MODEL_LABEL:-qwen3.5-9b}"
HF_REPO="${HF_REPO:-lmstudio-community/Qwen3.5-9B-GGUF}"
MODEL_PATH="${MODEL_PATH:-}"
QUANTS="${QUANTS:-Q8_0 Q6_K Q5_K_M Q4_K_M}"
PROMPTS="${PROMPTS:-1024 4096 8192 16384}"
DEPTHS="${DEPTHS:-0}"
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
ALLOW_WITH_LIVE_FORTBENCH="${ALLOW_WITH_LIVE_FORTBENCH:-0}"
DRY_RUN="${DRY_RUN:-0}"

check_live_fortbench() {
    pgrep -af 'fortbench run-suite .*pilot-runnable-20-local-all|llama-server.*Qwen3\.5-122B-A10B|llama-server.*Qwen3\.5-35B-A3B' >/dev/null
}

if [[ ! -x "${LLAMA_BENCH_BIN}" ]]; then
    echo "missing llama-bench binary: ${LLAMA_BENCH_BIN}" >&2
    exit 1
fi

if [[ "${ALLOW_WITH_LIVE_FORTBENCH}" != "1" ]] && [[ "${DRY_RUN}" != "1" ]] && check_live_fortbench; then
    echo "refusing to spawn prefill benchmarks while live FortBench local production is active" >&2
    echo "set ALLOW_WITH_LIVE_FORTBENCH=1 to override intentionally" >&2
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
    echo "model_label=${MODEL_LABEL}"
    echo "hf_repo=${HF_REPO}"
    echo "model_path=${MODEL_PATH}"
    echo "variant=${VARIANT}"
    echo "tensor_mode=${TENSOR_MODE}"
    echo "quants=${QUANTS}"
    echo "prompts=${PROMPTS}"
    echo "depths=${DEPTHS}"
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
        for depth in ${DEPTHS}; do
            for fa in ${FLASH_ATTN_VALUES}; do
                for batch in ${BATCHES}; do
                    for ubatch in ${UBATCHES}; do
                        run_id="${MODEL_LABEL}__${VARIANT}__${quant}__p${prompt}__d${depth}__fa${fa}__b${batch}__ub${ubatch}"
                        stdout_file="${OUT_DIR}/raw/${run_id}.jsonl"
                        stderr_file="${OUT_DIR}/raw/${run_id}.stderr.log"

                        args=(
                            -p "${prompt}"
                            -n 0
                            -d "${depth}"
                            -r "${REPETITIONS}"
                            -t "${THREADS}"
                            -ngl "${N_GPU_LAYERS}"
                            -b "${batch}"
                            -ub "${ubatch}"
                            -fa "${fa}"
                            --prio "${PRIO}"
                            -o jsonl
                        )

                        if [[ -n "${MODEL_PATH}" ]]; then
                            args=(-m "${MODEL_PATH}" "${args[@]}")
                        else
                            args=(-hf "${HF_REPO}:${quant}" "${args[@]}")
                        fi

                        if [[ "${NO_WARMUP}" == "1" ]]; then
                            args+=(--no-warmup)
                        fi

                        echo "==> ${run_id}"
                        if [[ "${DRY_RUN}" == "1" ]]; then
                            printf '%q ' "${LLAMA_BENCH_BIN}" "${args[@]}" > "${stdout_file}"
                            printf '\n' >> "${stdout_file}"
                            : > "${stderr_file}"
                            continue
                        fi

                        "${LLAMA_BENCH_BIN}" "${args[@]}" > "${stdout_file}" 2> "${stderr_file}"
                    done
                done
            done
        done
    done
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dry run complete; commands written under ${OUT_DIR}/raw"
    exit 0
fi

python3 "${ROOT_DIR}/scripts/apple/summarize-prefill-jsonl.py" \
    --input-dir "${OUT_DIR}/raw" \
    --csv "${OUT_DIR}/summary.csv" \
    --markdown "${OUT_DIR}/summary.md"

echo "results written to ${OUT_DIR}"
