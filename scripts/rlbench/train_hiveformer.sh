main_dir=hiveformer

DATA_PATH=/data/user_data/ngkanats

task=close_door

train_data_dir=$DATA_PATH/zarr_datasets/hiveformer/$task/train.zarr/
eval_data_dir=$DATA_PATH/zarr_datasets/hiveformer/$task/val.zarr/
train_instructions=instructions/hiveformer/$task.json
val_instructions=instructions/hiveformer/$task.json

dataset=HiveformerRLBench
num_workers=4
B=16
B_val=64
chunk_size=1
memory_limit=8

# Training/testing arguments, change these for HPT
val_freq=4000
eval_only=false
lr=1e-4
backbone_lr=1e-6  # doesn't matter when we don't finetune
lr_scheduler=constant
wd=1e-10
train_iters=100000  # this is a pessimistic estimate, convergence may be reached much earlier
use_compile=false
use_ema=false
lv2_batch_size=4  # we used 4 here for our results, different values could work better

# Model arguments, change (some of) these for new architectures
model_type=denoise3d
bimanual=false
keypose_only=false
pre_tokenize=true
workspace_normalizer_buffer=0.04

backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=4

C=144
num_attn_heads=8
num_vis_instr_attn_layers=2
num_history=3

num_shared_attn_layers=4
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=rectified_flow

run_log_dir=$task-$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

ngpus=1  # we actually used 1

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main.py \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --train_instructions $train_instructions \
    --val_instructions $val_instructions \
    --dataset $dataset \
    --num_workers $num_workers \
    --batch_size $B \
    --batch_size_val $B_val \
    --chunk_size $chunk_size \
    --memory_limit $memory_limit \
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir} \
    --checkpoint $checkpoint \
    --val_freq $val_freq \
    --eval_only $eval_only \
    --lr $lr \
    --backbone_lr $backbone_lr \
    --lr_scheduler $lr_scheduler \
    --wd $wd \
    --train_iters $train_iters \
    --use_compile $use_compile \
    --use_ema $use_ema \
    --lv2_batch_size $lv2_batch_size \
    --model_type $model_type \
    --bimanual $bimanual \
    --keypose_only $keypose_only \
    --pre_tokenize $pre_tokenize \
    --backbone $backbone \
    --finetune_backbone $finetune_backbone \
    --finetune_text_encoder $finetune_text_encoder \
    --fps_subsampling_factor $fps_subsampling_factor \
    --embedding_dim $C \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_history $num_history \
    --num_shared_attn_layers $num_shared_attn_layers \
    --workspace_normalizer_buffer $workspace_normalizer_buffer \
    --relative_action $relative_action \
    --rotation_format $rotation_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model
