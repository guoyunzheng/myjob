DATA_PATH=peract_raw/
ZARR_PATH=zarr_datasets/peract/

# Save current directory
CURR_DIR=$(pwd)

# Good news, PerAct data is pre-generated!
# Download the packaged 3DDA data
# cd ${DATA_PATH}
# wget https://huggingface.co/katefgroup/3d_diffuser_actor/resolve/main/Peract_packaged.zip
# unzip Peract_packaged.zip
# rm Peract_packaged.zip

# # We need the test seeds - yes, nothing else is needed for the test set
# # Download them
# wget https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract_test.zip
# unzip peract_test.zip
# rm peract_test.zip
# mv ${DATA_PATH}/peract_test ${DATA_PATH}/test

# Return to current directory
cd "$CURR_DIR"

# Then we package to zarr for training

python data_processing/peract_to_zarr.py \
    --root ${DATA_PATH}/Peract_packaged/ \
    --tgt ${ZARR_PATH}
# You can safely delete the Peract_packaged folder now