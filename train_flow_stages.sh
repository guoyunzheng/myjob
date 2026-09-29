#!/usr/bin/env bash
# Reuse one FM checkpoint across independent MeanFlow experiments.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ $# -ne 1 ]]; then
    echo "Usage: bash train_flow_stages.sh fm|meanflow" >&2
    echo "MeanFlow requires INIT_FROM=/path/to/FM/step240000.pth (or RESUME for an existing MF run)." >&2
    exit 2
fi

export SEED=${SEED:-0}
case "$1" in
    fm)
        export FLOW_OBJECTIVE=fm
        export TRAIN_ITERS=${TRAIN_ITERS:-300000}
        export MEANFLOW_OFFDIAG_RATIO=0
        export ENDPOINT_LOSS_WEIGHT=0
        export IVC_LOSS_WEIGHT=0
        export MILESTONE_CKPT_STEPS=${MILESTONE_CKPT_STEPS:-240000,300000}
        export RUN_LOG_DIR=${RUN_LOG_DIR:-fm_pretrain_300k}
        ;;
    meanflow)
        if [[ -z "${INIT_FROM:-}" && -z "${RESUME:-}" ]]; then
            echo "Set INIT_FROM to the fixed FM step240000.pth checkpoint." >&2
            exit 2
        fi
        export FLOW_OBJECTIVE=meanflow
        export TRAIN_ITERS=${TRAIN_ITERS:-60000}
        export MEANFLOW_OFFDIAG_RATIO=${MEANFLOW_OFFDIAG_RATIO:-0.75}
        export ENDPOINT_LOSS_WEIGHT=${ENDPOINT_LOSS_WEIGHT:-0}
        export IVC_LOSS_WEIGHT=${IVC_LOSS_WEIGHT:-0.5}
        export MILESTONE_CKPT_STEPS=${MILESTONE_CKPT_STEPS:-60000}
        export RUN_LOG_DIR=${RUN_LOG_DIR:-mf_from_fm240k_off075}
        ;;
    *)
        echo "Unknown mode '$1'; choose fm or meanflow." >&2
        exit 2
        ;;
esac

exec bash train.sh
