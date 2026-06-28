DATA_PATH=peract2_raw/
ZARR_PATH=zarr_datasets/peract2/

# Ultra-fast, just download our data!
cd ${ZARR_PATH}
wget https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2.zip
unzip peract2.zip
rm -rf peract2.zip

# Download the test seeds
CURR_DIR=$(pwd)
cd ${DATA_PATH}
wget https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2_test.zip
unzip peract2_test.zip
rm peract2_test.zip
cd "$CURR_DIR"
# Good to go!
