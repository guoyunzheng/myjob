#!/usr/bin/env bash

# Run train.sh in a transient
# systemd user service with a cgroup memory limit.
set -euo pipefail

unit_name="${UNIT_NAME:-gyz-train}"
memory_high="${MEMORY_HIGH:-100G}"
memory_max="${MEMORY_MAX:-120G}"
memory_swap_max="${MEMORY_SWAP_MAX:-1G}"
conda_env="${CONDA_ENV:-/home/adminpc/anaconda3/envs/3dfa}"

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
project_dir="$(dirname "$script_path")"

# The first invocation creates the constrained service. The service invokes
# this same file again with GYZ_LIMITED_CHILD=1 and enters the training branch.
if [[ "${GYZ_LIMITED_CHILD:-0}" != "1" ]]; then
    if systemctl --user is-active --quiet "${unit_name}.service"; then
        echo "${unit_name}.service is already running." >&2
        echo "Use: systemctl --user status ${unit_name}.service" >&2
        exit 1
    fi

    echo "Starting ${unit_name}.service"
    echo "MemoryHigh=${memory_high}, MemoryMax=${memory_max}, MemorySwapMax=${memory_swap_max}"

    exec systemd-run --user \
        --unit="$unit_name" \
        --collect \
        -p "WorkingDirectory=$project_dir" \
        -p MemoryAccounting=yes \
        -p "MemoryHigh=$memory_high" \
        -p "MemoryMax=$memory_max" \
        -p "MemorySwapMax=$memory_swap_max" \
        -p OOMPolicy=kill \
        -p KillMode=control-group \
        /usr/bin/env \
        GYZ_LIMITED_CHILD=1 \
        CONDA_ENV="$conda_env" \
        RESUME="${RESUME:-}" \
        INIT_FROM="${INIT_FROM:-}" \
        INIT_WEIGHTS="${INIT_WEIGHTS:-raw}" \
        RUN_LOG_DIR="${RUN_LOG_DIR:-}" \
        ACTION_HEAD="${ACTION_HEAD:-film_tcn}" \
        FLOW_OBJECTIVE="${FLOW_OBJECTIVE:-meanflow}" \
        ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}" \
        MATMUL_PRECISION="${MATMUL_PRECISION:-legacy}" \
        SEED="${SEED:-}" \
        TIME_SAMPLER="${TIME_SAMPLER:-logit_normal}" \
        TIME_SAMPLER_MEAN="${TIME_SAMPLER_MEAN:-0.0}" \
        TIME_SAMPLER_STD="${TIME_SAMPLER_STD:-1.5}" \
        MEANFLOW_OFFDIAG_RATIO="${MEANFLOW_OFFDIAG_RATIO:-}" \
        ENDPOINT_LOSS_WEIGHT="${ENDPOINT_LOSS_WEIGHT:-}" \
        IVC_LOSS_WEIGHT="${IVC_LOSS_WEIGHT:-}" \
        FLOW_LOSS_TYPE="${FLOW_LOSS_TYPE:-}" \
        GRIPPER_PREDICTION_MODE="${GRIPPER_PREDICTION_MODE:-direct}" \
        GRIPPER_LOSS_TYPE="${GRIPPER_LOSS_TYPE:-weighted_bce}" \
        GRIPPER_TRANSITION_WEIGHT="${GRIPPER_TRANSITION_WEIGHT:-}" \
        GRIPPER_CLOSED_HOLD_WEIGHT="${GRIPPER_CLOSED_HOLD_WEIGHT:-}" \
        GRIPPER_HOLD_PRIOR_LOGIT="${GRIPPER_HOLD_PRIOR_LOGIT:-2.0}" \
        GRIPPER_LOSS_WEIGHT="${GRIPPER_LOSS_WEIGHT:-1.0}" \
        POSE_POSITION_WEIGHT="${POSE_POSITION_WEIGHT:-30.0}" \
        POSE_ROTATION_WEIGHT="${POSE_ROTATION_WEIGHT:-10.0}" \
        JVP_CHUNKING="${JVP_CHUNKING:-false}" \
        JVP_MICROBATCH_SIZE="${JVP_MICROBATCH_SIZE:-8}" \
        BATCH_SIZE="${BATCH_SIZE:-128}" \
        LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
        TRAIN_ITERS="${TRAIN_ITERS:-300000}" \
        VAL_FREQ="${VAL_FREQ:-2000}" \
        DIAGNOSTIC_INTERVAL="${DIAGNOSTIC_INTERVAL:-100}" \
        VAL_BATCHES="${VAL_BATCHES:--1}" \
        VALIDATION_NOISE_REPEATS="${VALIDATION_NOISE_REPEATS:-3}" \
        VALIDATION_PROBE_BATCHES="${VALIDATION_PROBE_BATCHES:-4}" \
        DENOISE_TIMESTEPS="${DENOISE_TIMESTEPS:-5}" \
        PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}" \
        TRAIN_PROGRESS_INTERVAL="${TRAIN_PROGRESS_INTERVAL:-100}" \
        /bin/bash "$script_path"
fi

python_bin="$conda_env/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "Conda Python not found or not executable: $python_bin" >&2
    exit 127
fi

export PATH="$conda_env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

cd "$project_dir"

echo "Python: $python_bin"
echo "Project: $project_dir"
"$python_bin" -c \
    'import sys, torch, zarr; print(f"Interpreter: {sys.executable}"); print(f"torch={torch.__version__}, zarr={zarr.__version__}")'

progress_interval="${TRAIN_PROGRESS_INTERVAL:-100}"

if [[ ! "$progress_interval" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRAIN_PROGRESS_INTERVAL must be a positive integer: $progress_interval" >&2
    exit 2
fi

echo "Journal progress interval: approximately every $progress_interval steps"

# tqdm redraws progress with carriage returns, while journald records complete
# newline-delimited messages. The awk filter converts both separators to
# newlines, keeps ordinary output, and retains roughly one tqdm update per
# selected step interval. pipefail preserves torchrun's failure status.
# Keep service-specific defaults here; train.sh owns the shared recipe.
BATCH_SIZE="${BATCH_SIZE:-128}" VAL_FREQ="${VAL_FREQ:-2000}" \
    LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
    TRAIN_ITERS="${TRAIN_ITERS:-300000}" \
    DENOISE_TIMESTEPS="${DENOISE_TIMESTEPS:-5}" \
    JVP_CHUNKING="${JVP_CHUNKING:-false}" PYTHON_BIN="$python_bin" \
    /bin/bash "$project_dir/train.sh" 2>&1 | \
    stdbuf -oL awk -v interval="$progress_interval" '
        BEGIN {
            RS = "\r|\n"
            ORS = "\n"
            last_bucket = -1
            last_eval_bucket = -1
            step_offset = 0
        }
        length($0) == 0 { next }
        {
            # Capture the checkpoint step so the remaining-range tqdm counter
            # can be displayed as the absolute training step.
            if (match($0, /\(step [0-9]+\)/)) {
                checkpoint_step = substr($0, RSTART, RLENGTH)
                gsub(/[^0-9]/, "", checkpoint_step)
                step_offset = checkpoint_step + 0
                print
                fflush()
                next
            }

            # tqdm includes a current/total token such as 18123/150000.
            # Keep the first update in each interval-sized step bucket.
            if (match($0, /[0-9]+\/[0-9]+/)) {
                token = substr($0, RSTART, RLENGTH)
                split(token, counts, "/")
                current = counts[1] + 0
                total = counts[2] + 0
                bucket = int(current / interval)
                if (bucket == last_bucket && current != total) {
                    next
                }
                last_bucket = bucket
                absolute_token = (step_offset + current) "/" (step_offset + total)
                sub(token, absolute_token)
            } else if (match($0, /[0-9]+it/)) {
                # Evaluation tqdm instances have no known total (for example,
                # "123it"). Rate-limit these records in the same way.
                eval_token = substr($0, RSTART, RLENGTH)
                eval_current = eval_token + 0
                eval_bucket = int(eval_current / interval)
                if (eval_bucket == last_eval_bucket) {
                    next
                }
                last_eval_bucket = eval_bucket
            }
            print
            fflush()
        }
    '
