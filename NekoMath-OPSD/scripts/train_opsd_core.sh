#!/usr/bin/env bash
# Final trainer; the same implementation serves all three seeds.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${COLDSTART_MODEL:?Set COLDSTART_MODEL to the actual coldstart HF checkpoint}"
SEED="${SEED:-42}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED="$SEED" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" "$REPO_ROOT/scripts/train_opsd_core.py" \
  --model "$COLDSTART_MODEL" --system-prompt "$REPO_ROOT/prompts/system.txt" \
  --shared "${SHARED_DATA:-$REPO_ROOT/data/train/shared_core_len8192.jsonl}" \
  --output-dir "${OUTPUT_DIR:-$REPO_ROOT/outputs/opsd_seed$SEED}" --seed "$SEED" \
  --epochs "${EPOCHS:-2}" --max-new-tokens "${MAX_NEW_TOKENS:-8192}" \
  --max-length "${MAX_LENGTH:-10240}" --teacher-max-length "${TEACHER_MAX_LENGTH:-20480}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-4}" --global-batch-size "${GLOBAL_BATCH_SIZE:-8}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-0}" --lr "${LEARNING_RATE:-1e-6}" \
  --grad-clip "${GRAD_CLIP:-1.0}" --kl-clip "${KL_CLIP:-5}" --kl-chunk-tokens "${KL_CHUNK_TOKENS:-512}" \
  --pack-tokens "${PACK_TOKENS:-24576}" --rollout-buffer "${ROLLOUT_BUFFER:-3}" \
  --student-dtype bfloat16 --optimizer master --moment-dtype bfloat16 --master-dtype float32 \
  --save-every "${SAVE_EVERY:-50}" --teacher-device "${TEACHER_DEVICE:-cuda}" "$@"
