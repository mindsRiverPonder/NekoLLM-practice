#!/usr/bin/env bash
# Train -> export HF -> evaluate, in the environments selected by the user.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${COLDSTART_MODEL:?Set COLDSTART_MODEL}"
SEED="${SEED:-42}"
export SEED
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/opsd_seed$SEED}"
bash "$REPO_ROOT/scripts/train_opsd_core.sh" "$@"
python - "$OUTPUT_DIR/trainer_state.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1], encoding='utf-8')).get('complete') is True, 'training incomplete'
PY
python "$REPO_ROOT/scripts/export_opsd_hf.py" --base "$COLDSTART_MODEL" \
  --state "$OUTPUT_DIR/student.pt" --output "$OUTPUT_DIR/merged_hf"
# VLLM_BIN and EVAL_PYTHON may point to a separate vLLM environment.
bash "$REPO_ROOT/scripts/evaluate_hf_vllm.sh" "$OUTPUT_DIR/merged_hf" "$OUTPUT_DIR/evaluation" \
  "$REPO_ROOT/data/evaluation/testset_zh_compact.json" "${EVAL_GPUS:-0,1}"
