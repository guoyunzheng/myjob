# Standalone script, use scripts/rlbench/hiveformer_datagen.sh
# to generate and package as well
DATA_PATH=hiveformer_raw/

seed=0
variation=0
variation_count=1
tasks=(
    close_door
)

num_tasks=${#tasks[@]}
for ((i=0; i<$num_tasks; i++)); do
     xvfb-run -a python data_generation/generate.py \
          --save_path ${DATA_PATH} \
          --image_size 256,256 --renderer opengl \
          --episodes_per_task 100 \
          --tasks ${tasks[$i]} --variations ${variation_count} --offset ${variation} \
          --processes 1 --seed ${seed}
done
