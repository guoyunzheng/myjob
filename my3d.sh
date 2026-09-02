#!/bin/bash

# =========================================================
# 1. 环境清理：取消ROS和Isaac Sim的全局变量，防止干扰
# =========================================================

unset ROS_MASTER_URI
unset ROS_HOSTNAME
unset ROS_IP
# 确保 Isaac Sim 的 Python 路径不干扰
PYTHONPATH=$(echo $PYTHONPATH | tr ':' '\n' | grep -v 'isaac-sim/kit/python' | tr '\n' ':')
export PYTHONPATH=${PYTHONPATH%:} # 移除末尾的冒号

# =========================================================
# 2. 激活 Conda 环境 (假设您的项目依赖一个特定的环境)
# =========================================================
# 请根据您的实际环境名进行修改
echo "Activating Conda environment..."
source /home/adminpc/anaconda3/etc/profile.d/conda.sh
conda activate 3dfa  # <-- **请替换成您实际的环境名**

# =========================================================
# 3. 设置 CoppeliaSim 和 CUDA 路径 (根据您的bashrc中的设置)
# =========================================================
echo "Setting CoppeliaSim and CUDA 11.8 paths..."
# CoppeliaSim
export COPPELIASIM_ROOT=/home/adminpc/gyz/3d/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
# 确保 CoppeliaSim 的库路径被添加到最前面，拥有最高优先级
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# 仅使用 CUDA 11.8，覆盖 bashrc 中可能设置的 12.1
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# =========================================================
# 4. 执行评估脚本
# =========================================================
echo "Starting RLBench evaluation..."

exp=peract
tasks=(
    place_shape_in_shape_sorter
    place_wine_at_rack_location
    reach_and_drag
    close_jar
    insert_onto_square_peg
    light_bulb_in
    meat_off_grill
    open_drawer
    push_buttons
    put_groceries_in_cupboard
    put_item_in_drawer
    put_money_in_safe
    slide_block_to_color_target
    stack_blocks
    stack_cups
    sweep_to_dustpan_of_size
    turn_tap
    place_cups
)

# Testing arguments
checkpoint=${CHECKPOINT:-best.pth}

max_tries=${MAX_TRIES:-2}
max_steps=${MAX_STEPS:-20}
headless=False
collision_checking=${COLLISION_CHECKING:-False}
seed=0

# Dataset arguments
data_dir=peract_raw/peract_test
dataset=Peract
image_size=256,256

# Model arguments
model_type=denoise3d
bimanual=false
prediction_len=1

backbone=clip
fps_subsampling_factor=4

embedding_dim=120
num_attn_heads=8
num_vis_instr_attn_layers=2
num_history=3

num_shared_attn_layers=4
action_hidden_dim=256
action_num_blocks=6
guidance_scale=${GUIDANCE_SCALE:-1.0}
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=${DENOISE_TIMESTEPS:-2}
denoise_model=meanflow
checkpoint_alias=${CHECKPOINT_ALIAS:-my_awesome_peract_model-s${denoise_timesteps}-g${guidance_scale}}

num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
    python online_evaluation_rlbench/evaluate_policy.py \
        --checkpoint $checkpoint \
        --task ${tasks[$i]} \
        --max_tries $max_tries \
        --max_steps $max_steps \
        --headless $headless \
        --collision_checking $collision_checking \
        --seed $seed \
        --data_dir $data_dir \
        --dataset $dataset \
        --image_size $image_size \
        --output_file eval_logs/$exp/$checkpoint_alias/seed$seed/${tasks[$i]}/eval.json  \
        --model_type $model_type \
        --bimanual $bimanual \
        --prediction_len $prediction_len \
        --backbone $backbone \
        --fps_subsampling_factor $fps_subsampling_factor \
        --embedding_dim $embedding_dim \
        --num_attn_heads $num_attn_heads \
        --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
        --num_history $num_history \
        --num_shared_attn_layers $num_shared_attn_layers \
        --action_hidden_dim $action_hidden_dim \
        --action_num_blocks $action_num_blocks \
        --guidance_scale $guidance_scale \
        --relative_action $relative_action \
        --rotation_format $rotation_format \
        --denoise_timesteps $denoise_timesteps \
        --denoise_model $denoise_model
done

python online_evaluation_rlbench/collect_results.py \
    --folder eval_logs/$exp/$checkpoint_alias/seed$seed/
