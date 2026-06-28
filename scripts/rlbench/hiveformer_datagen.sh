DATA_PATH=hiveformer_raw/
ZARR_PATH=zarr_datasets/hiveformer/

# First we generate and store raw demos
seed=0
variation=0
variation_count=1
# If you change this list of tasks here, then change
# data_processing/hiveformer_to_zarr.py, ln 221 to be consistent!
tasks=(
    close_door
)

# Train demos
num_tasks=${#tasks[@]}
for ((i=0; i<$num_tasks; i++)); do
     xvfb-run -a python data_generation/generate.py \
          --save_path ${DATA_PATH}/train \
          --image_size 256,256 --renderer opengl \
          --episodes_per_task 2 \
          --tasks ${tasks[$i]} --variations ${variation_count} --offset ${variation} \
          --processes 1 --seed 0
done
# Val demos (different seed!)
num_tasks=${#tasks[@]}
for ((i=0; i<$num_tasks; i++)); do
     xvfb-run -a python data_generation/generate.py \
          --save_path ${DATA_PATH}/val \
          --image_size 256,256 --renderer opengl \
          --episodes_per_task 2 \
          --tasks ${tasks[$i]} --variations ${variation_count} --offset ${variation} \
          --processes 1 --seed 1
done
# We do not need test episodes, we generate the seeds on the fly!
# Done here, let's package now

# Then we package to zarr for training
# ATTENTION: make sure to modify line 221 (tasks = ["close_door",])
# in data_processing/hiveformer_to_zarr.py
# if you want to run on more/other tasks
python data_processing/hiveformer_to_zarr.py \
    --root ${DATA_PATH} \
    --tgt ${ZARR_PATH} \
    --store_trajectory true \
    --split train
python data_processing/hiveformer_to_zarr.py \
    --root ${DATA_PATH} \
    --tgt ${ZARR_PATH} \
    --store_trajectory true \
    --split val
# You can safely delete the raw data after this step
