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
action_head=${ACTION_HEAD:-film_tcn}
flow_objective=${FLOW_OBJECTIVE:-meanflow}
default_offdiag_ratio=0.25
default_ivc_weight=0.5
default_flow_loss=l1
default_endpoint_weight=0.25
if [[ "$flow_objective" == "fm" ]]; then
    default_offdiag_ratio=0.0
    default_ivc_weight=0.0
fi
if [[ "$flow_objective" == "imf" ]]; then
    # Opt-in boundary-reuse iMF; isolate the objective from auxiliary losses.
    default_flow_loss=l2
    default_ivc_weight=0.0
    default_endpoint_weight=0.0
fi
attention_backend=${ATTENTION_BACKEND:-auto}
time_sampler=${TIME_SAMPLER:-logit_normal}
time_sampler_mean=${TIME_SAMPLER_MEAN:-0.0}
time_sampler_std=${TIME_SAMPLER_STD:-1.5}
meanflow_offdiag_ratio=${MEANFLOW_OFFDIAG_RATIO:-$default_offdiag_ratio}
flow_loss_type=${FLOW_LOSS_TYPE:-$default_flow_loss}
action_hidden_dim=256
action_num_blocks=6
# MeanFlow/iMF use exact JVP. This switch only chunks their detached target /
# prediction correction to reduce peak VRAM; it never enables finite differences.
jvp_chunking=${JVP_CHUNKING:-true}
jvp_microbatch_size=${JVP_MICROBATCH_SIZE:-8}
case "$jvp_chunking" in
    false|False|FALSE|0) jvp_microbatch_size=0 ;;
esac
guidance_scale=1.0
endpoint_loss_weight=${ENDPOINT_LOSS_WEIGHT:-$default_endpoint_weight}
ivc_loss_weight=${IVC_LOSS_WEIGHT:-$default_ivc_weight}
condition_dropout_prob=0.0
gripper_loss_type=${GRIPPER_LOSS_TYPE:-weighted_bce}
default_gripper_cost=2.0
if [[ "$gripper_loss_type" == "bce" ]]; then
    default_gripper_cost=0.0
fi
gripper_transition_weight=${GRIPPER_TRANSITION_WEIGHT:-$default_gripper_cost}
gripper_closed_hold_weight=${GRIPPER_CLOSED_HOLD_WEIGHT:-$default_gripper_cost}
gripper_prediction_mode=${GRIPPER_PREDICTION_MODE:-direct}
gripper_hold_prior_logit=${GRIPPER_HOLD_PRIOR_LOGIT:-2.0}
pose_position_weight=${POSE_POSITION_WEIGHT:-30.0}
pose_rotation_weight=${POSE_ROTATION_WEIGHT:-10.0}
gripper_loss_weight=${GRIPPER_LOSS_WEIGHT:-1.0}
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=${DENOISE_TIMESTEPS:-2}

# Include the action-head/training recipe so an older Transformer checkpoint
# can never be resumed accidentally through the same directory.
recipe_dir=$model_type-$dataset-${action_head}_config_v2_gripsequence-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$flow_objective-$attention_backend-$time_sampler-m$time_sampler_mean-s$time_sampler_std-off$meanflow_offdiag_ratio-$flow_loss_type-S$denoise_timesteps-jvp$jvp_microbatch_size-ema$use_ema
# No implicit resume. Every fresh/initialized launch gets a distinct directory.
# Set RUN_LOG_DIR to the original directory to explicitly resume there.
run_log_dir=${RUN_LOG_DIR:-$recipe_dir-g$gripper_prediction_mode-$gripper_loss_type-ep$endpoint_loss_weight-ivc$ivc_loss_weight-$(date -u +%Y%m%dT%H%M%S)-$$}
resume=${RESUME:-}
init_from=${INIT_FROM:-}
init_weights=${INIT_WEIGHTS:-raw}
if [[ -n "$resume" && -n "$init_from" ]]; then
    echo "RESUME and INIT_FROM are mutually exclusive." >&2
    exit 2
fi
checkpoint_args=()
if [[ -n "${SEED:-}" ]]; then
    checkpoint_args+=(--seed "$SEED")
fi
if [[ -n "$resume" ]]; then
    checkpoint_args+=(--resume "$resume")
fi
if [[ -n "$init_from" ]]; then
    checkpoint_args+=(--init_from "$init_from")
fi

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
    "${checkpoint_args[@]}" \
    --init_weights "$init_weights" \
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
    --action_head "$action_head" \
    --flow_objective "$flow_objective" \
    --attention_backend "$attention_backend" \
    --matmul_precision "${MATMUL_PRECISION:-legacy}" \
    --time_sampler "$time_sampler" \
    --time_sampler_mean "$time_sampler_mean" \
    --time_sampler_std "$time_sampler_std" \
    --meanflow_offdiag_ratio "$meanflow_offdiag_ratio" \
    --flow_loss_type "$flow_loss_type" \
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
    --gripper_loss_type "$gripper_loss_type" \
    --gripper_loss_weight "$gripper_loss_weight" \
    --pose_position_weight "$pose_position_weight" \
    --pose_rotation_weight "$pose_rotation_weight" \
    --workspace_normalizer_buffer "$workspace_normalizer_buffer" \
    --relative_action "$relative_action" \
    --rotation_format "$rotation_format" \
    --denoise_timesteps "$denoise_timesteps"
