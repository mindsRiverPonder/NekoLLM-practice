#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${COLDSTART_MODEL:?Set COLDSTART_MODEL to the actual coldstart HF checkpoint}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUTPUT_DIR:-$REPO_ROOT/outputs/rft/$RUN_ID}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline WANDB_SILENT=true
export WANDB_PROJECT=neko-rft-vs-opsd WANDB_DIR="$OUT/wandb"
mkdir -p "$OUT" "$WANDB_DIR"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m verl.trainer.sft_trainer \
  data.train_files="${RFT_DATA:-$REPO_ROOT/data/train/rft_core.parquet}" data.val_files=null \
  data.messages_key=messages data.max_length="${MAX_LENGTH:-8192}" data.pad_mode=no_padding \
  data.train_batch_size="${GLOBAL_BATCH_SIZE:-8}" data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE:-1}" \
  data.use_dynamic_bsz=false data.num_workers=0 \
  model.path="$COLDSTART_MODEL" model.trust_remote_code=true model.use_remove_padding=true \
  model.lora_rank=0 +model.override_config.attn_implementation=sdpa \
  engine.model_dtype=bfloat16 engine.use_torch_compile=false \
  checkpoint.save_contents=[model,optimizer,extra] checkpoint.load_contents=[model,optimizer,extra] \
  trainer.total_epochs="${EPOCHS:-2}" trainer.seed=42 trainer.project_name=neko-rft-vs-opsd \
  trainer.experiment_name="rft_$RUN_ID" trainer.default_local_dir="$OUT" \
  trainer.n_gpus_per_node="$NPROC_PER_NODE" trainer.save_freq=after_each_epoch trainer.test_freq=-1 \
  trainer.resume_mode="${RESUME_MODE:-disable}" optim.lr="${LEARNING_RATE:-1e-6}" "$@" \
  2>&1 | tee "$OUT/train.log"
