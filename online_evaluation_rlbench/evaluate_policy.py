"""Online evaluation script on RLBench."""

import argparse
import random
from pathlib import Path
import json
import os

import torch
import numpy as np

from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from utils.common_utils import str2bool, str_none, round_floats
from utils.checkpoint_utils import load_model_state_strict


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Tuples: (name, type, default)
    arguments = [
        # Testing arguments
        ('checkpoint', str_none, None),
        ('task', str, "close_jar"),
        ('max_tries', int, 10),
        ('max_steps', int, 25),
        ('headless', str2bool, False),
        ('collision_checking', str2bool, False),
        ('gripper_open_threshold', float, 0.75),
        ('gripper_close_threshold', float, 0.25),
        ('require_pose_for_gripper_open', str2bool, True),
        ('seed', int, 0),
        # Dataset arguments
        ('data_dir', Path, Path(__file__).parent / "demos"),
        ('dataset', str, "Peract"),
        ('image_size', str, "256,256"),
        # Logging arguments
        ('output_file', Path, Path(__file__).parent / "eval.json"),
        # Model arguments: general policy type
        ('model_type', str, 'denoise3d'),
        ('bimanual', str2bool, False),
        ('prediction_len', int, 1),
        # Model arguments: encoder
        ('backbone', str, "clip"),
        ('fps_subsampling_factor', int, 5),
        # Model arguments: encoder and head
        ('embedding_dim', int, 144),
        ('num_attn_heads', int, 9),
        ('num_vis_instr_attn_layers', int, 2),
        ('num_history', int, 0),
        # Model arguments: head
        ('num_shared_attn_layers', int, 4),
        ('action_hidden_dim', int, 256),
        ('action_num_blocks', int, 6),
        ('guidance_scale', float, 1.0),
        ('gripper_prediction_mode', str, 'auto'),
        ('relative_action', str2bool, False),
        ('rotation_format', str, 'quat_xyzw'),
        ('denoise_timesteps', int, 2),
        ('denoise_model', str, "meanflow")
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def load_models(args):
    print("Loading model from", args.checkpoint, flush=True)

    model_dict = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    checkpoint_config = model_dict.get("config") or {}
    trained_gripper_mode = checkpoint_config.get(
        "gripper_prediction_mode", "legacy_denoise"
    )
    gripper_mode = (
        trained_gripper_mode
        if args.gripper_prediction_mode == "auto"
        else args.gripper_prediction_mode
    )
    if gripper_mode != trained_gripper_mode:
        raise ValueError(
            "Checkpoint gripper mode mismatch: trained with "
            f"{trained_gripper_mode!r}, requested {gripper_mode!r}."
        )

    model_class = fetch_model_class(args.model_type)
    model_kwargs = dict(
        backbone=args.backbone,
        num_vis_instr_attn_layers=args.num_vis_instr_attn_layers,
        fps_subsampling_factor=args.fps_subsampling_factor,
        embedding_dim=args.embedding_dim,
        num_attn_heads=args.num_attn_heads,
        nhist=args.num_history,
        nhand=2 if args.bimanual else 1,
        num_shared_attn_layers=args.num_shared_attn_layers,
        relative=args.relative_action,
        rotation_format=args.rotation_format,
        denoise_timesteps=args.denoise_timesteps,
        denoise_model=args.denoise_model
    )
    if args.model_type == "denoise3d":
        model_kwargs.update(
            action_hidden_dim=args.action_hidden_dim,
            action_num_blocks=args.action_num_blocks,
            guidance_scale=args.guidance_scale,
            gripper_prediction_mode=gripper_mode,
            gripper_hold_prior_logit=float(
                checkpoint_config.get("gripper_hold_prior_logit", 2.0)
            ),
        )
    model = model_class(**model_kwargs)

    if checkpoint_config:
        if checkpoint_config.get("denoise_model") != args.denoise_model:
            raise ValueError(
                "Checkpoint denoise_model mismatch: trained with "
                f"{checkpoint_config.get('denoise_model')!r}, requested "
                f"{args.denoise_model!r}."
            )
        trained_steps = checkpoint_config.get("denoise_timesteps")
        if trained_steps is not None and trained_steps != args.denoise_timesteps:
            print(
                "WARNING: inference step count differs from checkpoint validation: "
                f"trained/validated={trained_steps}, requested={args.denoise_timesteps}."
            )
        trained_guidance = checkpoint_config.get("guidance_scale")
        if (
            trained_guidance is not None
            and float(trained_guidance) != args.guidance_scale
        ):
            print(
                "WARNING: inference guidance differs from checkpoint recipe: "
                f"trained/validated={trained_guidance}, "
                f"requested={args.guidance_scale}."
            )

    selected_weight = (
        model_dict["ema_weight"]
        if model_dict.get("ema_weight") is not None
        else model_dict["weight"]
    )
    weight_name = "EMA" if model_dict.get("ema_weight") is not None else "raw"
    load_model_state_strict(
        model,
        selected_weight,
        str(args.checkpoint),
        allowed_missing_prefixes=(
            "encoder.proprio_state_encoder.",
            "condition_pooler.relevance_score.",
            "condition_pooler.spatial_moment_projection.",
            "condition_pooler.secondary_relevance_score.",
            "condition_pooler.secondary_slot_projection.",
        ),
    )
    print(
        f"Loaded {weight_name} weights strictly "
        f"(training step {model_dict.get('iter', 'unknown')}, "
        f"gripper mode {gripper_mode})."
    )
    model.eval()

    return model.cuda()


if __name__ == "__main__":
    # Arguments
    args = parse_arguments()
    print("Arguments:")
    print(args)
    print("-" * 100)

    # Save results here
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Bimanual vs single-arm utils
    if args.bimanual:
        from online_evaluation_rlbench.utils_with_bimanual_rlbench import RLBenchEnv, Actioner
    elif "peract" in args.dataset.lower():
        from online_evaluation_rlbench.utils_with_rlbench import RLBenchEnv, Actioner
    else:
        from online_evaluation_rlbench.utils_with_hiveformer_rlbench import RLBenchEnv, Actioner

    # Dataset class (for getting cameras and tasks/variations)
    dataset_class = fetch_dataset_class(args.dataset)

    # Load models
    model = load_models(args)
    # print(model.workspace_normalizer)

    # Evaluate - reload environment for each task (crashes otherwise)
    task_success_rates = {}
    for task_str in [args.task]:

        # Seeds - re-seed for each task
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # Load RLBench environment
        env_kwargs = dict(
            data_path=args.data_dir,
            task_str=task_str,
            image_size=[int(x) for x in args.image_size.split(",")],
            apply_rgb=True,
            apply_pc=True,
            headless=bool(args.headless),
            apply_cameras=dataset_class.cameras,
            collision_checking=bool(args.collision_checking),
        )
        if not args.bimanual and "peract" in args.dataset.lower():
            env_kwargs.update(
                gripper_open_threshold=args.gripper_open_threshold,
                gripper_close_threshold=args.gripper_close_threshold,
                require_pose_for_gripper_open=(
                    args.require_pose_for_gripper_open
                ),
            )
        env = RLBenchEnv(**env_kwargs)

        # Actioner (runs the policy online)
        actioner = Actioner(model, backbone=args.backbone)

        # Evaluate
        var_success_rates = env.evaluate_task_on_multiple_variations(
            task_str,
            max_steps=args.max_steps,
            actioner=actioner,
            max_tries=args.max_tries,
            prediction_len=args.prediction_len,
            num_history=args.num_history
        )
        print()
        print(
            f"{task_str} variation success rates:",
            round_floats(var_success_rates)
        )
        print(
            f"{task_str} mean success rate:",
            round_floats(var_success_rates["mean"])
        )

        task_success_rates[task_str] = var_success_rates
        with open(args.output_file, "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)
