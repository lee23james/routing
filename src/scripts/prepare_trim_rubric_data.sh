#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${TRIM_RUBRIC_PREPARE_SOURCE_ONLY:-0}" != "1" ]; then
    cd "$SCRIPT_DIR/.."
fi

MATH_DRAFT_SERVER_URL="${MATH_DRAFT_SERVER_URL:-http://localhost:30001/v1}"
MATH_TARGET_SERVER_URL="${MATH_TARGET_SERVER_URL:-http://localhost:30000/v1}"
MATH_PRM_SERVER_URL="${MATH_PRM_SERVER_URL:-http://localhost:30002}"

AIME_DRAFT_SERVER_URL="${AIME_DRAFT_SERVER_URL:-http://localhost:30001/v1}"
AIME_TARGET_SERVER_URL="${AIME_TARGET_SERVER_URL:-http://localhost:30000/v1}"
AIME_PRM_SERVER_URL="${AIME_PRM_SERVER_URL:-http://localhost:30002}"

DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-/home/deepseek_VG/deepseek_VG/routing/routing/models/qwen3-1.7b}"
TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-/home/deepseek_VG/deepseek_VG/routing/routing/models/qwen3-14b}"
PRM_MODEL_NAME="${PRM_MODEL_NAME:-/home/deepseek_VG/deepseek_VG/routing/routing/models/qwen2.5-math-prm-7b}"

MAX_WORKERS="${MAX_WORKERS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
RUBRIC_TRAJECTORIES="${RUBRIC_TRAJECTORIES:-20}"
RUBRIC_CORR_THRESHOLD="${RUBRIC_CORR_THRESHOLD:-0.05}"
RUBRIC_STD_THRESHOLD="${RUBRIC_STD_THRESHOLD:-0.02}"

mkdir -p data/episodes data/rubrics logs

expected_episode_count() {
    local dataset="$1"
    case "$dataset" in
        math_train_1k)
            echo "${EXPECTED_MATH_TRAIN_1K_EPISODES:-1100}"
            ;;
        math500_test_100)
            echo "${EXPECTED_MATH500_TEST_100_EPISODES:-100}"
            ;;
        aime_train)
            echo "${EXPECTED_AIME_TRAIN_EPISODES:-451}"
            ;;
        aime_test)
            echo "${EXPECTED_AIME_TEST_EPISODES:-482}"
            ;;
        *)
            echo ""
            ;;
    esac
}

generate_dataset() {
    local dataset="$1"
    local draft_url="$2"
    local target_url="$3"
    local prm_url="$4"
    local out_file="data/episodes/${dataset}_episodes.jsonl"
    local expected_count
    local current_count

    if [ -f "$out_file" ] && [ "${FORCE:-0}" != "1" ]; then
        expected_count="$(expected_episode_count "$dataset")"
        current_count="$(wc -l < "$out_file")"
        if [ -n "$expected_count" ] && [ "$current_count" -ge "$expected_count" ]; then
            echo "[skip] episodes complete: $out_file ($current_count/$expected_count)"
            return 0
        fi
        if [ -n "$expected_count" ]; then
            echo "[resume] episodes partial: $out_file ($current_count/$expected_count)"
        else
            echo "[resume] episodes exist: $out_file ($current_count lines)"
        fi
    fi
    if [ -f "$out_file" ] && [ "${FORCE:-0}" = "1" ]; then
        rm -f "$out_file"
    fi

    echo "[generate] $dataset -> $out_file"
    if [ "${TRIM_RUBRIC_TEST_STUB_GENERATE:-0}" = "1" ]; then
        GENERATE_CALLED="${GENERATE_CALLED:-}${GENERATE_CALLED:+ }$dataset"
        export GENERATE_CALLED
        return 0
    fi

    PYTHONUNBUFFERED=1 python -u -m data.generate_episodes \
        --dataset "$dataset" \
        --output_dir data/episodes \
        --srm_server_url "$draft_url" \
        --lrm_server_url "$target_url" \
        --srm_model_name "$DRAFT_MODEL_NAME" \
        --lrm_model_name "$TARGET_MODEL_NAME" \
        --prm_server_url "$prm_url" \
        --prm_model_name "$PRM_MODEL_NAME" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --max_workers "$MAX_WORKERS" \
        2>&1 | tee "logs/generate_${dataset}.log"
}

learn_rubrics() {
    local dataset="$1"
    local ep_file="data/episodes/${dataset}_episodes.jsonl"
    local out_dir="data/rubrics/${dataset}"

    if [ ! -f "$ep_file" ]; then
        echo "[error] missing episodes for rubric discovery: $ep_file" >&2
        exit 1
    fi
    if [ -f "$out_dir/rubric_weights.json" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[skip] rubric weights exist: $out_dir/rubric_weights.json"
        return 0
    fi
    if [ -d "$out_dir" ] && [ "${FORCE:-0}" = "1" ]; then
        rm -f "$out_dir/rubric_weights.json" \
              "$out_dir/rubric_consistency.json" \
              "$out_dir/episode_rubric_scores.jsonl"
    fi

    echo "[rubric] $dataset -> $out_dir"
    PYTHONUNBUFFERED=1 python -u -m rubric.generate_rubrics \
        --episodes_path "$ep_file" \
        --output_dir "$out_dir" \
        --n_trajectories "$RUBRIC_TRAJECTORIES" \
        --corr_threshold "$RUBRIC_CORR_THRESHOLD" \
        --std_threshold "$RUBRIC_STD_THRESHOLD" \
        2>&1 | tee "logs/rubric_${dataset}.log"
}

main() {
    echo "Preparing TRIM-Rubric datasets"
    echo "  Math: draft=$MATH_DRAFT_SERVER_URL target=$MATH_TARGET_SERVER_URL prm=$MATH_PRM_SERVER_URL"
    echo "  AIME: draft=$AIME_DRAFT_SERVER_URL target=$AIME_TARGET_SERVER_URL prm=$AIME_PRM_SERVER_URL"

    generate_dataset "math_train_1k" "$MATH_DRAFT_SERVER_URL" "$MATH_TARGET_SERVER_URL" "$MATH_PRM_SERVER_URL"
    generate_dataset "math500_test_100" "$MATH_DRAFT_SERVER_URL" "$MATH_TARGET_SERVER_URL" "$MATH_PRM_SERVER_URL"
    generate_dataset "aime_train" "$AIME_DRAFT_SERVER_URL" "$AIME_TARGET_SERVER_URL" "$AIME_PRM_SERVER_URL"
    generate_dataset "aime_test" "$AIME_DRAFT_SERVER_URL" "$AIME_TARGET_SERVER_URL" "$AIME_PRM_SERVER_URL"

    learn_rubrics "math_train_1k"
    learn_rubrics "aime_train"

    echo "TRIM-Rubric data preparation complete."
}

if [ "${TRIM_RUBRIC_PREPARE_SOURCE_ONLY:-0}" != "1" ]; then
    main "$@"
fi
