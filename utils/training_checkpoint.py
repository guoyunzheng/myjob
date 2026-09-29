"""Versioned training checkpoints: strict resume versus weights-only initialization."""

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import random
import subprocess
import uuid

import numpy as np
import torch

from .checkpoint_utils import load_model_state_strict, validate_model_state


CHECKPOINT_VERSION = 2
# Changing logging destinations/frequency does not change the learned function.
# Validation recipe remains checked because best_loss depends on it.
OPERATIONAL_FIELDS = {
    "resume", "init_from", "init_weights", "checkpoint", "eval_only",
    "base_log_dir", "exp_log_dir", "run_log_dir", "log_dir", "local_rank",
    "diagnostic_interval", "interm_ckpt_freq", "milestone_ckpt_steps",
}
ARCHITECTURE_FIELDS = (
    "model_type", "backbone", "embedding_dim", "num_attn_heads",
    "num_vis_instr_attn_layers", "num_history", "bimanual",
    "num_shared_attn_layers", "action_head", "action_hidden_dim",
    "action_num_blocks", "rotation_format", "relative_action",
    "gripper_prediction_mode", "gripper_hold_prior_logit", "fps_subsampling_factor", "keypose_only",
    "custom_img_size",
    "input_contract_version",
)


def normalize_checkpoint_arguments(args, parser):
    if args.resume and args.init_from:
        parser.error("--resume and --init_from are mutually exclusive.")
    if args.checkpoint:
        if args.resume or args.init_from:
            parser.error("Do not combine legacy --checkpoint with --resume/--init_from.")
        if not args.eval_only:
            args.resume = args.checkpoint  # Strict alias, never a best-effort resume.
    if args.eval_only and (args.resume or args.init_from or not args.checkpoint):
        parser.error("--eval_only requires --checkpoint, not --resume/--init_from.")
    if args.init_weights != "raw" and not args.init_from:
        parser.error("--init_weights=ema requires --init_from.")
    return args


def config_snapshot(args):
    return {key: str(value) if isinstance(value, os.PathLike) else value
            for key, value in vars(args).items() if key not in {"local_rank", "log_dir"}}


def read_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Refusing to start from scratch.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping.")
    return checkpoint


def select_weights(checkpoint, selection="raw"):
    if selection == "ema":
        state = checkpoint.get("ema_weight")
        if state is None:
            raise ValueError("EMA weights requested but checkpoint has no ema_weight.")
        return state
    if selection != "raw":
        raise ValueError("Weight selection must be raw or ema.")
    return checkpoint.get("weight", checkpoint)


def validate_normalizer(state):
    entries = [value for key, value in state.items()
               if key.removeprefix("module.") == "workspace_normalizer"]
    if len(entries) != 1:
        raise ValueError("Weights must contain exactly one workspace_normalizer.")
    bounds = entries[0]
    if (not torch.is_tensor(bounds) or bounds.ndim != 2 or bounds.shape[0] != 2
            or not torch.isfinite(bounds).all() or not torch.all(bounds[1] > bounds[0])):
        raise ValueError("workspace_normalizer must contain finite, increasing bounds.")
    return bounds.detach().cpu().clone()


def validate_init_config(checkpoint, config, fields=ARCHITECTURE_FIELDS):
    # Objective/optimizer settings MAY change for a new experiment. Same-shaped
    # architecture semantics (heads, rotation conventions, etc.) may not.
    saved = checkpoint.get("config") or {}
    differences = [key for key in fields
                   if key in saved and saved[key] != config.get(key)]
    if differences:
        raise ValueError(f"Initialization architecture mismatch: {differences}. "
                         "Partial architecture transfer is not implemented.")
    if not saved:
        print("WARNING: weights-only file has no config; only keys, shapes and normalizer can be checked.")
    elif "input_contract_version" not in saved and saved.get("num_history") in (1, 2):
        print("WARNING: legacy checkpoint has no input contract version and num_history=1/2; "
              "old training may have selected the oldest states. New inputs retain the current "
              "state. Treat this as a new input-distribution experiment, not equivalent evaluation.")


def validate_evaluation_config(checkpoint, config):
    # Unlike weight initialization, evaluation cannot switch the learned field
    # semantics merely because the two heads happen to have identical shapes.
    validate_init_config(checkpoint, config,
                         fields=ARCHITECTURE_FIELDS + ("flow_objective", "denoise_model"))


def validate_resume(checkpoint, config, world_size):
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Strict resume requires a version-2 full training checkpoint. "
                         "Legacy checkpoints lack full state; use --init_from for a new run.")
    required = ("weight", "optimizer", "lr_scheduler", "scaler", "ema_state",
                "iter", "best_loss", "config", "run_metadata", "rank_states",
                "workspace_normalizer", "world_size", "steps_per_epoch", "scaler_enabled")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Incomplete resume checkpoint: missing {missing}.")
    for key in ("weight", "optimizer", "lr_scheduler", "scaler", "ema_state", "config", "run_metadata"):
        if not isinstance(checkpoint[key], Mapping):
            raise ValueError(f"Invalid resume state: {key} must be a mapping.")
    saved = checkpoint["config"]
    keys = (set(saved) | set(config)) - OPERATIONAL_FIELDS
    differences = [f"{key}: saved={saved.get(key)!r}, requested={config.get(key)!r}"
                   for key in sorted(keys)
                   if key not in saved or key not in config or saved[key] != config[key]]
    if differences:
        raise ValueError("Strict resume config mismatch:\n" + "\n".join(differences)
                         + "\nUse --init_from if this is a new experiment.")
    if checkpoint["world_size"] != world_size or len(checkpoint["rank_states"]) != world_size:
        raise ValueError("Strict resume requires the same distributed world_size.")
    if type(checkpoint["steps_per_epoch"]) is not int or checkpoint["steps_per_epoch"] < 1:
        raise ValueError("Invalid steps_per_epoch in checkpoint.")
    if not {"state", "param_groups"} <= checkpoint["optimizer"].keys():
        raise ValueError("Incomplete optimizer state.")
    if "last_epoch" not in checkpoint["lr_scheduler"]:
        raise ValueError("Incomplete learning-rate scheduler state.")
    if checkpoint["scaler_enabled"] and not checkpoint["scaler"]:
        raise ValueError("Enabled AMP scaler is missing its state.")
    step = checkpoint["iter"]
    if type(step) is not int or step < 0 or step > config["train_iters"]:
        raise ValueError("Invalid checkpoint iteration.")
    if checkpoint.get("validation_version") != 2:
        raise ValueError("Strict resume validation metric version mismatch.")
    if config["use_ema"] and checkpoint.get("ema_weight") is None:
        raise ValueError("EMA-enabled resume requires ema_weight.")
    bounds = validate_normalizer(checkpoint["weight"])
    if not torch.equal(bounds, checkpoint["workspace_normalizer"]):
        raise ValueError("Checkpoint normalizer metadata disagrees with model weights.")
    if config["use_ema"] and not torch.equal(bounds, validate_normalizer(checkpoint["ema_weight"])):
        raise ValueError("EMA and raw workspace normalizers disagree.")
    if not checkpoint["run_metadata"].get("run_id"):
        raise ValueError("Resume checkpoint has no run_id.")
    for state in checkpoint["rank_states"]:
        if not isinstance(state, Mapping) or not {"rng", "loader_epoch_generator"} <= state.keys():
            raise ValueError("Resume checkpoint is missing per-rank RNG/loader state.")
        if not isinstance(state["rng"], Mapping) or not {"python", "numpy", "torch", "cuda"} <= state["rng"].keys():
            raise ValueError("Incomplete RNG state.")


def capture_rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG state cannot be restored without CUDA.")
        torch.cuda.set_rng_state(state["cuda"].cpu())


def git_identity(project_dir):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(project_dir), *args],
                                       stderr=subprocess.DEVNULL).decode("utf-8").strip()
    try:
        # Include uncommitted and untracked project source, but not data/weights.
        files = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
        digest = hashlib.sha256()
        roots = {"modeling", "utils", "datasets", "online_evaluation_rlbench", "experiments"}
        for name in sorted(set(files)):
            path = Path(project_dir) / name
            if (name == "main.py" or (name.split("/")[0] in roots and name.endswith(".py"))) and path.is_file():
                digest.update(name.encode("utf-8"))
                digest.update(path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode("utf-8"))
        return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                "dirty": bool(git("status", "--porcelain")), "source_sha256": digest.hexdigest()}
    except (OSError, subprocess.CalledProcessError, UnicodeError):
        return {"commit": None, "branch": None, "dirty": None, "source_sha256": None}


def make_run_metadata(config, source=None, mode="scratch", source_path=None):
    previous = (source or {}).get("run_metadata") or {}
    return {
        "run_id": previous["run_id"] if mode == "resume" else uuid.uuid4().hex,
        "session_id": uuid.uuid4().hex,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode, "source_checkpoint": str(source_path) if source_path else None,
        "parent_run_id": previous.get("run_id") if mode == "init" else previous.get("parent_run_id"),
        "parent_session_id": previous.get("session_id"),
        "git": git_identity(Path(__file__).resolve().parents[1]),
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "cuda": torch.version.cuda, "numpy": str(np.__version__)},
        "config": config,
    }


def check_output_directory(log_dir, resume_path=None):
    log_dir = Path(log_dir)
    artifacts = (list(log_dir.glob("*.pth")) + list(log_dir.glob("run_*.json"))
                 + list(log_dir.glob("events.out.tfevents.*")))
    if artifacts:
        last = log_dir / "last.pth"
        if resume_path is None or not last.is_file() or Path(resume_path).resolve() != last.resolve():
            raise ValueError("Output directory already contains a run. Use a fresh run_log_dir "
                             "or explicitly --resume that directory's last.pth.")


def write_run_manifest(log_dir, metadata):
    path = Path(log_dir) / f"run_{metadata['session_id']}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)


def initialize_weights(model, ema_model, checkpoint, selection="raw"):
    state = select_weights(checkpoint, selection)
    validate_normalizer(state)
    validate_model_state(model, state)
    validate_model_state(ema_model, state)
    load_model_state_strict(model, state)
    # Both branches must start from the selected weights, not the pre-load copy.
    load_model_state_strict(ema_model, state)


def restore_training_state(checkpoint, model, ema_model, optimizer, scheduler, scaler, ema):
    if checkpoint["scaler_enabled"] != scaler.is_enabled():
        raise ValueError("Resume AMP scaler enabled state differs from the runtime.")
    validate_model_state(model, checkpoint["weight"])
    ema_weights = checkpoint.get("ema_weight")
    if ema_weights is None:
        ema_weights = checkpoint["weight"]
    validate_model_state(ema_model, ema_weights)
    load_model_state_strict(model, checkpoint["weight"])
    load_model_state_strict(ema_model, ema_weights)
    # Scheduler was constructed before optimizer restore; do NOT replay step().
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["lr_scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    ema.load_state_dict(checkpoint["ema_state"])
    return checkpoint["iter"], checkpoint["best_loss"]


def build_training_checkpoint(model, ema_model, optimizer, scheduler, scaler, ema,
                              *, config, run_metadata, step, best_loss, rank_states,
                              steps_per_epoch):
    state = model.state_dict()
    return {
        "checkpoint_version": CHECKPOINT_VERSION, "validation_version": 2,
        "weight": state, "ema_weight": ema_model.state_dict() if config["use_ema"] else None,
        "optimizer": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "ema_state": ema.state_dict(),
        "scaler_enabled": scaler.is_enabled(),
        "iter": step, "best_loss": best_loss, "config": config,
        "workspace_normalizer": validate_normalizer(state),
        "run_metadata": run_metadata, "world_size": len(rank_states),
        "rank_states": rank_states, "steps_per_epoch": steps_per_epoch,
    }


def atomic_save_checkpoint(checkpoint, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
