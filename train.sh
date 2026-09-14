#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

main_dir=Peract

train_data_dir=zarr_datasets/peract/train.zarr/
eval_data_dir=zarr_datasets/peract/val.zarr/
train_instructions=instructions/peract/instructions.json
val_instructions=instructions/peract/instructions.json

dataset=Peract
num_workers=4
B=${BATCH_SIZE:-64}
B_val=16
chunk_size=1
memory_limit=8  # this means 8GB CPU RAM per worker per GPU,
# but it will never reach that, because these datasets are small
# reduce this if you can't allocate more than 96GB of CPU memory

# Training/testing arguments
val_freq=${VAL_FREQ:-4000}
diagnostic_interval=${DIAGNOSTIC_INTERVAL:-100}
val_batches=${VAL_BATCHES:--1}
validation_noise_repeats=${VALIDATION_NOISE_REPEATS:-3}
validation_probe_batches=${VALIDATION_PROBE_BATCHES:-4}
eval_only=false
lr=${LEARNING_RATE:-1e-4}
backbone_lr=1e-6  # doesn't matter when we don't finetune
lr_scheduler=cosine
wd=1e-4
train_iters=${TRAIN_ITERS:-300000}
use_compile=false # much faster, but sometimes unstable
use_ema=true
lv2_batch_size=1  # you can increase this and divide B equally, speed/accuracy tradeoff

# Model arguments, change (some of) these for new architectures
model_type=denoise3d
bimanual=false
keypose_only=true
pre_tokenize=true
workspace_normalizer_buffer=0.04

backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=4

C=120
num_attn_heads=8
num_vis_instr_attn_layers=2
num_history=3

num_shared_attn_layers=4
action_hidden_dim=256
action_num_blocks=6
# Exact JVP is always used by MeanFlow. This switch only controls whether its
# stop-gradient target is computed in smaller chunks to reduce peak VRAM.
jvp_chunking=${JVP_CHUNKING:-true}
jvp_microbatch_size=${JVP_MICROBATCH_SIZE:-8}
case "$jvp_chunking" in
    false|False|FALSE|0) jvp_microbatch_size=0 ;;
esac
guidance_scale=1.0
endpoint_loss_weight=0.25
ivc_loss_weight=0.5  # Extra (t,t) velocity supervision for samples with r != t.
condition_dropout_prob=0.0
gripper_transition_weight=2.0
gripper_closed_hold_weight=2.0
gripper_prediction_mode=direct
gripper_hold_prior_logit=2.0
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=${DENOISE_TIMESTEPS:-2}
denoise_model=meanflow

# Include the action-head/training recipe so an older Transformer checkpoint
# can never be resumed accidentally through the same directory.
run_log_dir=$model_type-$dataset-film_tcn_exact_jvp_v5_gripsequence-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-S$denoise_timesteps-jvp$jvp_microbatch_size-ema$use_ema
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
ngpus=1
master_port=$((20000 + RANDOM % 20000))

exec "${PYTHON_BIN:-python}" -u -m torch.distributed.run \
    --nproc_per_node "$ngpus" --master_port "$master_port" \
    main.py \
    --train_data_dir "$train_data_dir" \
    --eval_data_dir "$eval_data_dir" \
    --train_instructions "$train_instructions" \
    --val_instructions "$val_instructions" \
    --dataset "$dataset" \
    --num_workers "$num_workers" \
    --batch_size "$B" \
    --batch_size_val "$B_val" \
    --chunk_size "$chunk_size" \
    --memory_limit "$memory_limit" \
    --exp_log_dir "$main_dir" \
    --run_log_dir "$run_log_dir" \
    --checkpoint "$checkpoint" \
    --val_freq "$val_freq" \
    --diagnostic_interval "$diagnostic_interval" \
    --val_batches "$val_batches" \
    --validation_noise_repeats "$validation_noise_repeats" \
    --validation_probe_batches "$validation_probe_batches" \
    --eval_only "$eval_only" \
    --lr "$lr" \
    --backbone_lr "$backbone_lr" \
    --lr_scheduler "$lr_scheduler" \
    --wd "$wd" \
    --train_iters "$train_iters" \
    --use_compile "$use_compile" \
    --use_ema "$use_ema" \
    --lv2_batch_size "$lv2_batch_size" \
    --model_type "$model_type" \
    --bimanual "$bimanual" \
    --keypose_only "$keypose_only" \
    --pre_tokenize "$pre_tokenize" \
    --backbone "$backbone" \
    --finetune_backbone "$finetune_backbone" \
    --finetune_text_encoder "$finetune_text_encoder" \
    --fps_subsampling_factor "$fps_subsampling_factor" \
    --embedding_dim "$C" \
    --num_attn_heads "$num_attn_heads" \
    --num_vis_instr_attn_layers "$num_vis_instr_attn_layers" \
    --num_history "$num_history" \
    --num_shared_attn_layers "$num_shared_attn_layers" \
    --action_hidden_dim "$action_hidden_dim" \
    --action_num_blocks "$action_num_blocks" \
    --jvp_microbatch_size "$jvp_microbatch_size" \
    --guidance_scale "$guidance_scale" \
    --endpoint_loss_weight "$endpoint_loss_weight" \
    --ivc_loss_weight "$ivc_loss_weight" \
    --condition_dropout_prob "$condition_dropout_prob" \
    --gripper_transition_weight "$gripper_transition_weight" \
    --gripper_closed_hold_weight "$gripper_closed_hold_weight" \
    --gripper_prediction_mode "$gripper_prediction_mode" \
    --gripper_hold_prior_logit "$gripper_hold_prior_logit" \
    --workspace_normalizer_buffer "$workspace_normalizer_buffer" \
    --relative_action "$relative_action" \
    --rotation_format "$rotation_format" \
    --denoise_timesteps "$denoise_timesteps" \
    --denoise_model "$denoise_model"
