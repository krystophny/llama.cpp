#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${ROOT_DIR}/scripts/apple"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/llama.cpp-apple-prefill-suite/$(date +%Y%m%d-%H%M%S)}"

MODELS="${MODELS:-qwen3.5-9b qwen3.5-35b-a3b}"
VARIANTS="${VARIANTS:-baseline tensor-enable tensor-disable}"
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
ALLOW_WITH_LIVE_FORTBENCH="${ALLOW_WITH_LIVE_FORTBENCH:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_SAMPLES="${RUN_SAMPLES:-0}"
PROFILE_QUANT="${PROFILE_QUANT:-Q8_0}"
PROFILE_PROMPT="${PROFILE_PROMPT:-8192}"
PROFILE_DEPTH="${PROFILE_DEPTH:-0}"
PROFILE_FLASH_ATTN="${PROFILE_FLASH_ATTN:-1}"
PROFILE_BATCH="${PROFILE_BATCH:-2048}"
PROFILE_UBATCH="${PROFILE_UBATCH:-512}"

repo_for_model() {
    case "$1" in
        qwen3.5-9b)
            echo "lmstudio-community/Qwen3.5-9B-GGUF"
            ;;
        qwen3.5-35b-a3b)
            echo "lmstudio-community/Qwen3.5-35B-A3B-GGUF"
            ;;
        *)
            echo "unknown model: $1" >&2
            return 1
            ;;
    esac
}

tensor_mode_for_variant() {
    case "$1" in
        baseline)
            echo "default"
            ;;
        tensor-enable)
            echo "enable"
            ;;
        tensor-disable)
            echo "disable"
            ;;
        *)
            echo "unknown variant: $1" >&2
            return 1
            ;;
    esac
}

mkdir -p "${OUT_DIR}"

{
    echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root_dir=${ROOT_DIR}"
    echo "out_dir=${OUT_DIR}"
    echo "models=${MODELS}"
    echo "variants=${VARIANTS}"
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
    echo "run_samples=${RUN_SAMPLES}"
    echo "profile_quant=${PROFILE_QUANT}"
    echo "profile_prompt=${PROFILE_PROMPT}"
    echo "profile_depth=${PROFILE_DEPTH}"
    echo "profile_flash_attn=${PROFILE_FLASH_ATTN}"
    echo "profile_batch=${PROFILE_BATCH}"
    echo "profile_ubatch=${PROFILE_UBATCH}"
} > "${OUT_DIR}/suite.env"

for model in ${MODELS}; do
    hf_repo="$(repo_for_model "${model}")"
    for variant in ${VARIANTS}; do
        tensor_mode="$(tensor_mode_for_variant "${variant}")"
        run_dir="${OUT_DIR}/${model}/${variant}"
        mkdir -p "${run_dir}"

        MODEL_LABEL="${model}" \
        HF_REPO="${hf_repo}" \
        QUANTS="${QUANTS}" \
        PROMPTS="${PROMPTS}" \
        DEPTHS="${DEPTHS}" \
        FLASH_ATTN_VALUES="${FLASH_ATTN_VALUES}" \
        BATCHES="${BATCHES}" \
        UBATCHES="${UBATCHES}" \
        REPETITIONS="${REPETITIONS}" \
        THREADS="${THREADS}" \
        N_GPU_LAYERS="${N_GPU_LAYERS}" \
        PRIO="${PRIO}" \
        VARIANT="${variant}" \
        TENSOR_MODE="${tensor_mode}" \
        ALLOW_WITH_LIVE_FORTBENCH="${ALLOW_WITH_LIVE_FORTBENCH}" \
        DRY_RUN="${DRY_RUN}" \
        OUT_DIR="${run_dir}" \
        "${SCRIPT_DIR}/bench-prefill-qwen.sh"

        if [[ "${RUN_SAMPLES}" == "1" ]]; then
            sample_dir="${run_dir}/sample-${PROFILE_QUANT}-p${PROFILE_PROMPT}-d${PROFILE_DEPTH}"
            mkdir -p "${sample_dir}"
            HF_REPO="${hf_repo}" \
            QUANT="${PROFILE_QUANT}" \
            PROMPT="${PROFILE_PROMPT}" \
            DEPTH="${PROFILE_DEPTH}" \
            FLASH_ATTN="${PROFILE_FLASH_ATTN}" \
            BATCH="${PROFILE_BATCH}" \
            UBATCH="${PROFILE_UBATCH}" \
            THREADS="${THREADS}" \
            N_GPU_LAYERS="${N_GPU_LAYERS}" \
            PRIO="${PRIO}" \
            SPAWN_BENCH=1 \
            ALLOW_WITH_LIVE_FORTBENCH="${ALLOW_WITH_LIVE_FORTBENCH}" \
            OUT_DIR="${sample_dir}" \
            "${SCRIPT_DIR}/sample-prefill-hotpath.sh"
        fi
    done
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dry run complete; no summaries were generated"
    exit 0
fi

python3 "${SCRIPT_DIR}/compare-prefill-results.py" \
    --input-root "${OUT_DIR}" \
    --csv "${OUT_DIR}/comparison.csv" \
    --markdown "${OUT_DIR}/comparison.md"

echo "suite results written to ${OUT_DIR}"
