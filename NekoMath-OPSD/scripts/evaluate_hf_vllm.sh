#!/usr/bin/env bash
# Pass@4 evaluation of an arbitrary HF model directory with 4 vLLM servers.
# Same config/test/params as scripts/evaluate_vllm_minicpm.sh (which only accepts
# a verl FSDP checkpoint directory), so the numbers stay comparable with the
# existing evaluations/rft and evaluations/opsd results.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${1:?usage: bash scripts/evaluate_hf_vllm.sh <model_dir> <out_dir> [test_file] [gpus]}"
OUT="${2:?usage: bash scripts/evaluate_hf_vllm.sh <model_dir> <out_dir> [test_file] [gpus]}"
TEST_FILE="${3:-$ROOT/data/evaluation/testset_zh_compact.json}"
GPUS="${4:-0,1}"
MODEL_NAME=small-nmath-minicpm
[[ -f "${MODEL_DIR}/config.json" ]] || { echo "not an HF model dir: ${MODEL_DIR}" >&2; exit 1; }
mkdir -p "${OUT}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PYTHON_BIN="${EVAL_PYTHON:-python}"
export VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
PORTS=(); for g in ${GPUS//,/ }; do PORTS+=("$((18000+g))"); done
for port in "${PORTS[@]}"; do
  if ss -ltn | grep -q ":${port} "; then
    echo "port ${port} is already occupied; refusing to start another evaluator" >&2; exit 1
  fi
done
PGIDS=()
for g in ${GPUS//,/ }; do
  port=$((18000+g))
  setsid env CUDA_VISIBLE_DEVICES="${g}" VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false \
    "${VLLM_BIN}" serve "${MODEL_DIR}" --served-model-name "${MODEL_NAME}" --host 127.0.0.1 --port "${port}" \
    --dtype bfloat16 --max-model-len 20480 --max-num-seqs 16 --gpu-memory-utilization 0.92 --trust-remote-code \
    >"${OUT}/vllm_${port}.log" 2>&1 &
  PGIDS+=("$!")
done
cleanup() {
  trap - EXIT INT TERM
  for pgid in "${PGIDS[@]}"; do kill -TERM -- "-${pgid}" 2>/dev/null || true; done
  for n in $(seq 1 30); do
    alive=0
    for pgid in "${PGIDS[@]}"; do kill -0 -- "-${pgid}" 2>/dev/null && alive=1; done
    (( alive == 0 )) && break
    sleep 1
  done
  for pgid in "${PGIDS[@]}"; do kill -KILL -- "-${pgid}" 2>/dev/null || true; done
  for port in "${PORTS[@]}"; do
    for n in $(seq 1 30); do ss -ltn | grep -q ":${port} " || break; sleep 1; done
  done
}
trap cleanup EXIT INT TERM
for port in "${PORTS[@]}"; do
  ready=0; for n in $(seq 1 240); do curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && { ready=1; break; }; sleep 2; done
  (( ready == 1 )) || { echo "vLLM port ${port} failed to start; inspect ${OUT}/vllm_${port}.log" >&2; exit 1; }
done
"${PYTHON_BIN}" "${ROOT}/scripts/evaluate_vllm_http.py" --model "${MODEL_NAME}" --system-prompt "${ROOT}/prompts/system.txt" --test-file "${TEST_FILE}" --output "${OUT}" \
  --ports "$(IFS=,; echo "${PORTS[*]}")" --batch-size 4 --max-tokens 16384 --top-p .95 --temperature 1.0 --seed 42
cleanup
