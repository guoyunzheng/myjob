main_dir=Peract

DATA_PATH=/data/user_data/ngkanats

train_data_dir=zarr_datasets/peract/train.zarr/
eval_data_dir=zarr_datasets/peract/val.zarr/
train_instructions=instructions/peract/instructions.json
val_instructions=instructions/peract/instructions.json

dataset=Peract
num_workers=4
B=64 # we used 64 but you can use as low as 16 without much performance drop - it's much faster
B_val=16
chunk_size=1
memory_limit=8  # this means 8GB CPU RAM per worker per GPU,
# but it will never reach that, because these datasets are small
# reduce this if you can't allocate more than 96GB of CPU memory

# Training/testing arguments
val_freq=4000
eval_only=false
lr=1e-4
backbone_lr=1e-6  # doesn't matter when we don't finetune
lr_scheduler=constant
wd=1e-10
train_iters=300000
use_compile=true # much faster, but sometimes unstable
use_ema=false
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
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=meanflow

run_log_dir=$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

# 开启可扩展段以减少显存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 限制显存预占
export PYTORCH_ALLOC_CONF=max_split_size_mb:128
ngpus=1  # we used 4

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
