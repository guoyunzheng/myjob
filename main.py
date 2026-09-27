"""Main script for training and testing."""

import argparse
import os
from pathlib import Path
import sys

import torch

from modeling.flow_config import add_flow_arguments, normalize_flow_arguments, configure_action_precision
from modeling.loss_config import add_loss_arguments, normalize_loss_arguments
from utils.reproducibility import seed_training
from utils.common_utils import str2bool, str_none
from utils.training_checkpoint import normalize_checkpoint_arguments


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Tuples: (name, type, default)
    arguments = [
        # Dataset/loader arguments
        ('train_data_dir', Path, ''),
        ('eval_data_dir', Path, ''),
        ('train_instructions', Path, ''),
        ('val_instructions', Path, ''),
        ('dataset', str, "Peract"),
        ('num_workers', int, 4),
        ('batch_size', int, 64),
        ('batch_size_val', int, 64),
        ('chunk_size', int, 1),
        ('memory_limit', float, 8),  # cache limit in GB
        # Logging arguments
        ('base_log_dir', Path, Path(__file__).parent / "train_logs"),
        ('exp_log_dir', Path, "exp"),
        ('run_log_dir', Path, "run"),
        # Training and testing arguments
        ('val_freq', int, 4000),
        ('diagnostic_interval', int, 100),
        ('val_batches', int, -1),  # -1 validates the complete validation set
        ('validation_noise_repeats', int, 3),
        ('validation_probe_batches', int, 4),
        ('interm_ckpt_freq', int, 1000000),
        ('eval_only', str2bool, False),
        ('lr', float, 1e-4),
        ('backbone_lr', float, 1e-4),
        ('lr_scheduler', str, "constant"),
        ('wd', float, 5e-3),
        ('train_iters', int, 600000),
        ('seed', int, None),  # Opt-in: preserve historical unseeded launches.
        ('matmul_precision', str, 'legacy'),
        ('use_compile', str2bool, False),
        ('use_ema', str2bool, False),
        ('lv2_batch_size', int, 1),
        # Model arguments: general policy type
        ('model_type', str, 'denoise3d'),
        ('bimanual', str2bool, False),
        ('keypose_only', str2bool, True),
        ('pre_tokenize', str2bool, True),
        ('custom_img_size', int, None),
        ('workspace_normalizer_buffer', float, 0.04),
        # Model arguments: encoder
        ('backbone', str, "clip"),
        ('finetune_backbone', str2bool, False),
        ('finetune_text_encoder', str2bool, False),
        ('fps_subsampling_factor', int, 5),
        # Model arguments: encoder and head
        ('embedding_dim', int, 120),  # divisible by num_attn_heads
        ('num_attn_heads', int, 8),
        ('num_vis_instr_attn_layers', int, 3),
        ('num_history', int, 1),
        # Model arguments: head
        ('num_shared_attn_layers', int, 4),
        ('action_hidden_dim', int, 256),
        ('action_num_blocks', int, 6),
        ('jvp_microbatch_size', int, 8),
        ('guidance_scale', float, 1.0),
        ('endpoint_loss_weight', float, 0.25),
        ('ivc_loss_weight', float, 0.0),
        ('condition_dropout_prob', float, 0.0),
        ('gripper_transition_weight', float, None),
        ('gripper_closed_hold_weight', float, None),
        ('gripper_prediction_mode', str, 'direct'),
        ('gripper_hold_prior_logit', float, 2.0),
        ('relative_action', str2bool, False),
        ('rotation_format', str, 'quat_xyzw'),
        ('denoise_timesteps', int, 2),
        ('denoise_model', str, None)  # Legacy scheduler alias; prefer flow_objective.
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    add_flow_arguments(parser)
    add_loss_arguments(parser)
    loading = parser.add_argument_group('Checkpoint loading (no implicit resume)')
    loading.add_argument('--checkpoint', type=str_none, default=None,
                         help='Evaluation weights; during training, legacy alias for strict --resume.')
    loading.add_argument('--resume', type=str_none, default=None,
                         help='Restore a full version-2 training checkpoint with matching config.')
    loading.add_argument('--init_from', type=str_none, default=None,
                         help='Initialize matching model weights only; start a new run at step 0.')
    loading.add_argument('--init_weights', choices=('raw', 'ema'), default='raw',
                         help='Which weights to use with --init_from (default: raw).')
    args = normalize_flow_arguments(parser.parse_args(argv), parser)
    args = normalize_loss_arguments(args, parser)
    args = normalize_checkpoint_arguments(args, parser)
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error('seed must be in [0, 2**32).')
    if args.matmul_precision not in ('legacy', 'ieee'):
        parser.error('matmul_precision must be legacy or ieee.')
    if args.diagnostic_interval < 1 or args.validation_noise_repeats < 1:
        parser.error(
            'diagnostic_interval and validation_noise_repeats must be positive'
        )
    if args.validation_probe_batches < 0:
        parser.error('validation_probe_batches must be non-negative')
    if args.val_batches != -1 and args.val_batches < 1:
        parser.error('val_batches must be -1 or positive')
    return args


def suppress_output_on_non_main():
    if int(os.environ.get("RANK", 0)) != 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Arguments
    args = parse_arguments()
    from datasets import fetch_dataset_class
    from modeling.policy import fetch_model_class
    from utils.trainers import fetch_train_tester
    print("Arguments:")
    print(args)
    print("-" * 100)

    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count:", torch.cuda.device_count())
    args.local_rank = int(os.environ["LOCAL_RANK"])
    suppress_output_on_non_main()

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    configure_action_precision(args.action_head, args.matmul_precision)
    seed_training(args.seed, torch.distributed.get_rank())

    # Select dataset and model classes
    dataset_class = fetch_dataset_class(args.dataset)
    model_class = fetch_model_class(args.model_type)

    # Run
    TrainTester = fetch_train_tester(args.dataset)
    train_tester = TrainTester(args, dataset_class, model_class)
    train_tester.main()

    # Safe program termination
    if torch.distributed.is_initialized():
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
