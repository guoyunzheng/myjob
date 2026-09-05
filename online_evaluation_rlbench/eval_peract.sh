exp=peract
tasks=(
    close_jar
    insert_onto_square_peg
    light_bulb_in
    meat_off_grill
    open_drawer
    place_shape_in_shape_sorter
    place_wine_at_rack_location
    push_buttons
    put_groceries_in_cupboard
    put_item_in_drawer
    put_money_in_safe
    reach_and_drag
    slide_block_to_color_target
    stack_blocks
    stack_cups
    sweep_to_dustpan_of_size
    turn_tap
    place_cups
)

# Testing arguments
checkpoint=peract.pth
checkpoint_alias=my_awesome_peract_model

max_tries=2
max_steps=20
headless=True
collision_checking=False #可能的问题。
gripper_open_threshold=0.75
gripper_close_threshold=0.25
require_pose_for_gripper_change=True
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
guidance_scale=1.0
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=2
denoise_model=meanflow

num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
    xvfb-run -a python online_evaluation_rlbench/evaluate_policy.py \
        --checkpoint $checkpoint \
        --task ${tasks[$i]} \
        --max_tries $max_tries \
        --max_steps $max_steps \
        --headless $headless \
        --collision_checking $collision_checking \
        --gripper_open_threshold $gripper_open_threshold \
        --gripper_close_threshold $gripper_close_threshold \
        --require_pose_for_gripper_change $require_pose_for_gripper_change \
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
        --gripper_prediction_mode auto \
        --relative_action $relative_action \
        --rotation_format $rotation_format \
        --denoise_timesteps $denoise_timesteps \
        --denoise_model $denoise_model
done

python online_evaluation_rlbench/collect_results.py \
    --folder eval_logs/$exp/$checkpoint_alias/seed$seed/
