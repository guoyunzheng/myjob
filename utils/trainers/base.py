from copy import deepcopy
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from modeling.encoder.text import fetch_tokenizers
from ..common_utils import count_parameters
from ..depth2cloud import fetch_depth2cloud
from ..data_preprocessors import fetch_data_preprocessor
from ..ema import EMA
from ..training_checkpoint import (
    read_checkpoint, config_snapshot, validate_resume, validate_init_config, validate_evaluation_config,
    make_run_metadata, check_output_directory, write_run_manifest,
    initialize_weights, restore_training_state, capture_rng_state,
    restore_rng_state, build_training_checkpoint, atomic_save_checkpoint,
)
from ..schedulers import fetch_scheduler
from .utils import compute_task_balanced_selection
from .validation import ActionValidation, noise_sensitivity


class BaseTrainTester:
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args, dataset_cls, model_cls):
        """Initialize."""
        self.args = args
        self.dataset_cls = dataset_cls
        self.model_cls = model_cls

        # Fail before datasets, normalizer scans, model allocations or log writes.
        self.config = config_snapshot(args)
        self.load_mode = "resume" if args.resume else "init" if args.init_from else "scratch"
        source_path = args.resume or args.init_from or args.checkpoint
        self.source_checkpoint = read_checkpoint(source_path) if source_path else None
        if args.eval_only:
            self.load_mode = "eval"
        if self.load_mode == "resume":
            validate_resume(self.source_checkpoint, self.config, dist.get_world_size())
        elif args.eval_only:
            validate_evaluation_config(self.source_checkpoint, self.config)
        elif self.source_checkpoint is not None:
            validate_init_config(self.source_checkpoint, self.config)
        if not args.eval_only:
            check_output_directory(args.log_dir, args.resume)
        metadata = [make_run_metadata(self.config, self.source_checkpoint,
                                      self.load_mode, source_path) if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(metadata, src=0)
        self.run_metadata = metadata[0]
        if self.load_mode == "resume":
            old_hash = self.source_checkpoint["run_metadata"].get("git", {}).get("source_sha256")
            new_hash = self.run_metadata["git"]["source_sha256"]
            if old_hash and new_hash and old_hash != new_hash:
                raise ValueError("Strict resume source-code fingerprint mismatch. "
                                 "Use the original code or --init_from for a new experiment.")
            if not old_hash or not new_hash:
                print("WARNING: source fingerprint unavailable; code identity cannot be verified.")

        self.preprocessor = fetch_data_preprocessor(self.args.dataset)(
            self.args.keypose_only,
            self.args.num_history,
            custom_imsize=self.args.custom_img_size,
            depth2cloud=fetch_depth2cloud(self.args.dataset)
        )

    def get_datasets(self):
        """Initialize datasets."""
        # Initialize datasets with arguments
        train_dataset = self.dataset_cls(
            root=self.args.train_data_dir,
            instructions=self.args.train_instructions,
            relative_action=self.args.relative_action,
            mem_limit=self.args.memory_limit,
            chunk_size=self.args.chunk_size,
            deterministic_instructions=False,
        )
        val_dataset = self.dataset_cls(
            root=self.args.eval_data_dir,
            instructions=self.args.val_instructions,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1,
            chunk_size=self.args.chunk_size,
            deterministic_instructions=True,
        )
        return train_dataset, val_dataset

    def get_loaders(self):
        """Initialize data loaders."""
        from utils.reproducibility import data_seed
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Datasets
        train_dataset, val_dataset = self.get_datasets()
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(data_seed(self.args.seed))
        self.train_generator = g
        train_sampler = DistributedSampler(train_dataset, drop_last=True, seed=data_seed(self.args.seed))
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size // self.args.chunk_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            worker_init_fn=seed_worker,
            collate_fn=base_collate_fn,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            generator=g,
            prefetch_factor=4,
            persistent_workers=True
        )
        # No sampler for val!
        if dist.get_rank() == 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size_val // self.args.chunk_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=base_collate_fn,
                pin_memory=True,
                sampler=None,
                drop_last=False,
                prefetch_factor=4,
                persistent_workers=True
            )
        else:
            val_loader = None
        self.unique_train_samples = len(train_dataset.annos["action"])
        self.validation_samples = len(val_dataset.annos["action"])
        self.global_batch_size = self.args.batch_size * dist.get_world_size()
        if dist.get_rank() == 0:
            validation_samples = len(val_dataset.annos["action"])
            print(
                f"Dataset: {self.unique_train_samples} unique train samples, "
                f"{validation_samples} validation samples, global batch "
                f"{self.global_batch_size}."
            )
        return train_loader, val_loader, train_sampler

    def get_model(self):
        """Initialize the model."""
        from modeling.flow_config import flow_model_kwargs
        from modeling.loss_config import extra_loss_model_kwargs
        # Initialize model with arguments
        model_kwargs = dict(
            backbone=self.args.backbone,
            finetune_backbone=self.args.finetune_backbone,
            finetune_text_encoder=self.args.finetune_text_encoder,
            num_vis_instr_attn_layers=self.args.num_vis_instr_attn_layers,
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            embedding_dim=self.args.embedding_dim,
            num_attn_heads=self.args.num_attn_heads,
            nhist=self.args.num_history,
            nhand=2 if self.args.bimanual else 1,
            num_shared_attn_layers=self.args.num_shared_attn_layers,
            relative=self.args.relative_action,
            rotation_format=self.args.rotation_format,
            denoise_timesteps=self.args.denoise_timesteps,
            denoise_model=self.args.denoise_model,
            lv2_batch_size=self.args.lv2_batch_size,
        )
        if self.args.model_type == "denoise3d":
            model_kwargs.update(flow_model_kwargs(self.args))
            model_kwargs.update(extra_loss_model_kwargs(self.args))
            model_kwargs.update(
                action_hidden_dim=self.args.action_hidden_dim,
                action_num_blocks=self.args.action_num_blocks,
                jvp_microbatch_size=self.args.jvp_microbatch_size,
                guidance_scale=self.args.guidance_scale,
                endpoint_loss_weight=self.args.endpoint_loss_weight,
                ivc_loss_weight=self.args.ivc_loss_weight,
                condition_dropout_prob=self.args.condition_dropout_prob,
                gripper_transition_weight=self.args.gripper_transition_weight,
                gripper_closed_hold_weight=self.args.gripper_closed_hold_weight,
                gripper_prediction_mode=self.args.gripper_prediction_mode,
                gripper_hold_prior_logit=self.args.gripper_hold_prior_logit,
            )
        _model = self.model_cls(**model_kwargs)

        # Print basic modules' parameters
        if dist.get_rank() == 0:
            count_parameters(_model)

        # Useful for some models to ensure parameters are contiguous
        for name, param in _model.named_parameters():
            if param.requires_grad and param.ndim > 1 and not param.is_contiguous():
                print(f"Fixing layout for: {name}")
                param.data = param.contiguous()

        return _model

    @torch.no_grad()
    def get_workspace_normalizer(self, ndims=3):
        print("Computing workspace normalizer...")

        # Initialize datasets with arguments
        train_dataset = self.dataset_cls(
            root=self.args.train_data_dir,
            instructions=self.args.train_instructions,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1,
            actions_only=True,
            chunk_size=self.args.chunk_size
        )

        data_loader = DataLoader(
            train_dataset,
            batch_size=max(self.args.batch_size, 64) // self.args.chunk_size,
            collate_fn=actions_collate_fn,
            shuffle=False,
            num_workers=self.args.num_workers
        )

        # Loop and compute action min-max
        min_, max_ = torch.ones(ndims) * 10000, -torch.ones(ndims) * 10000
        for sample in tqdm(data_loader):
            action = sample["action"][..., :ndims].reshape([-1, ndims])
            min_ = torch.min(min_, action.min(0).values)
            max_ = torch.max(max_, action.max(0).values)

        min_ = min_ - self.args.workspace_normalizer_buffer
        max_ = max_ + self.args.workspace_normalizer_buffer

        return nn.Parameter(torch.stack([min_, max_]), requires_grad=False)

    def get_optimizer(self, model):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.args.lr},
            {"params": [], "weight_decay": self.args.wd, "lr": self.args.lr}
        ]
        if self.args.finetune_backbone:
            optimizer_grouped_parameters.append({
                "params": [], "weight_decay": self.args.wd,
                "lr": self.args.backbone_lr
            })

        # Collect names of all norm parameters
        norm_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LocalResponseNorm,
            torch.nn.RMSNorm
        )
        norm_param_names = set()
        for module_name, module in model.named_modules():
            if isinstance(module, norm_types):
                for param_name, _ in module.named_parameters(recurse=False):
                    norm_param_names.add(f"{module_name}.{param_name}")

        # Now split parameters based on name
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in norm_param_names or name.endswith(".bias"):
                optimizer_grouped_parameters[0]["params"].append(param)
            elif self.args.finetune_backbone and 'backbone' in name:
                optimizer_grouped_parameters[2]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.95)
        )
        return optimizer

    def main(self):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, val_loader, train_sampler = self.get_loaders()
        dataset_counts = {
            "train_samples": self.unique_train_samples,
            "validation_samples": self.validation_samples,
        }
        if self.load_mode == "resume":
            saved_counts = self.source_checkpoint["run_metadata"].get("dataset_counts")
            if saved_counts is not None and saved_counts != dataset_counts:
                raise ValueError("Strict resume dataset sample counts changed.")
        self.run_metadata["dataset_counts"] = dataset_counts

        # Get model
        model = self.get_model()
        self.run_metadata["parameter_counts"] = {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "action_head": sum(p.numel() for p in model.prediction_head.parameters())
            if hasattr(model, "prediction_head") else None,
        }
        self.tokenizer = fetch_tokenizers(self.args.backbone)
        if self.source_checkpoint is None:
            normalizer = self.get_workspace_normalizer()
            model.workspace_normalizer.copy_(normalizer)
            dist.barrier(device_ids=[torch.cuda.current_device()])

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda()
        # Build the optimizer only after parameters are on their final device.
        optimizer = self.get_optimizer(model)
        lr_scheduler = fetch_scheduler(
            self.args.lr_scheduler, optimizer, self.args.train_iters
        )
        scaler = torch.GradScaler()
        # make sure to compile before DDP!
        if self.args.use_compile:
            model.compute_loss = torch.compile(model.compute_loss, fullgraph=True)
        model = DistributedDataParallel(
            model, device_ids=[self.args.local_rank],
            broadcast_buffers=False, find_unused_parameters=True
        )

        # Initialize EMA copy
        ema_model = deepcopy(model)
        self.ema = EMA()

        # Check for a checkpoint
        start_iter, best_loss = 0, None
        self.steps_per_epoch = len(train_loader)
        if self.steps_per_epoch < 1 and not self.args.eval_only:
            raise ValueError("Training loader has no complete batches.")
        if self.load_mode == "resume":
            if self.source_checkpoint["steps_per_epoch"] != self.steps_per_epoch:
                raise ValueError("Strict resume dataset/loader length changed.")
            start_iter, best_loss = restore_training_state(
                self.source_checkpoint, model, ema_model, optimizer, lr_scheduler, scaler, self.ema
            )
        elif self.source_checkpoint is not None:
            selection = ("ema" if self.args.use_ema else "raw") if self.args.eval_only else self.args.init_weights
            initialize_weights(model, ema_model, self.source_checkpoint, selection)
        print(f"Checkpoint mode={self.load_mode}, run_id={self.run_metadata['run_id']}, step={start_iter}")
        print(model.module.workspace_normalizer)

        # Eval only
        if self.args.eval_only:
            if dist.get_rank() == 0:
                print("Test evaluation.......")
                model.eval()
                self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    val_loader, step_id=-1,
                    val_iters=-1
                )
            dist.barrier(device_ids=[torch.cuda.current_device()])
            return ema_model if self.args.use_ema else model

        if dist.get_rank() == 0:
            self.run_metadata["workspace_normalizer"] = model.module.workspace_normalizer.detach().cpu().tolist()
            write_run_manifest(self.args.log_dir, self.run_metadata)
            self.writer = SummaryWriter(log_dir=self.args.log_dir, purge_step=start_iter or None)
            self.writer.add_text("run/config", str(self.config), start_iter)
            self.writer.add_text("run/identity", str(self.run_metadata), start_iter)
            self.writer.add_scalar("run/unique_train_samples", self.unique_train_samples, start_iter)
            self.writer.add_scalar("run/global_batch_size", self.global_batch_size, start_iter)
            self.writer.add_scalar("run/validation_samples", self.validation_samples, start_iter)

        # Step the sampler to the currect "epoch"
        samples_per_epoch = len(train_loader)
        epoch = start_iter // samples_per_epoch + 1
        train_sampler.set_epoch(epoch)  # ensures new batches are sampled

        # Training loop
        model.train()
        rank_state = None
        if self.load_mode == "resume":
            rank_state = self.source_checkpoint["rank_states"][dist.get_rank()]
            self.train_generator.set_state(rank_state["loader_epoch_generator"].cpu())
        self.loader_epoch_generator_state = self.train_generator.get_state()
        iter_loader = iter(train_loader)
        if rank_state is not None:
            # Replay the consumed batch indices, then restore training-process RNG.
            # Persistent worker/prefetch RNG is not serializable: this is NOT a
            # promise of bitwise-identical augmentations after a restart.
            for _ in range(start_iter % samples_per_epoch):
                next(iter_loader)
            restore_rng_state(rank_state["rng"])
            print("Restored training state and batch cursor; worker-prefetch replay is not bitwise guaranteed.")
        self.source_checkpoint = None
        for step_id in trange(start_iter, self.args.train_iters):
            try:
                sample = next(iter_loader)
            except StopIteration:
                # when the iterator is exhausted, we need to reset it
                # and increment the epoch
                epoch += 1
                train_sampler.set_epoch(epoch)
                self.loader_epoch_generator_state = self.train_generator.get_state()
                iter_loader = iter(train_loader)
                sample = next(iter_loader)

            self.train_one_step(
                model, optimizer, scaler, lr_scheduler, sample, step_id
            )
            self.ema.step(model, ema_model, self.args.use_ema, step_id)

            save_due = (step_id + 1) % self.args.val_freq == 0 or step_id + 1 == self.args.train_iters
            new_loss = None
            if save_due and dist.get_rank() == 0:
                print("Train evaluation.......")
                model.eval()
                self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    train_loader, step_id,
                    val_iters=10,
                    split='train'
                )
                print("Test evaluation.......")
                new_loss = self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    val_loader, step_id,
                    val_iters=self.args.val_batches
                )
                model.train()
            if save_due:
                state = {
                    "rng": capture_rng_state(),
                    "loader_epoch_generator": (
                        self.train_generator.get_state() if (step_id + 1) % samples_per_epoch == 0
                        else self.loader_epoch_generator_state
                    ),
                }
                rank_states = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
                dist.gather_object(state, rank_states, dst=0)
            if save_due and dist.get_rank() == 0:
                best_loss = self.save_checkpoint(
                    model, ema_model, optimizer, lr_scheduler, scaler, step_id,
                    new_loss, best_loss, rank_states
                )
            dist.barrier(device_ids=[torch.cuda.current_device()])

        return ema_model if self.args.use_ema else model

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        pass  # implement in children

    def _model_forward(self, model, sample, training=True):
        action, action_mask, rgbs, rgb2d, pcds, instr, prop = self.prepare_batch(
            sample, augment=training
        )
        if self.args.pre_tokenize:
            instr = self.tokenizer(instr).cuda(non_blocking=True)
        # The encoder keeps BF16/Flash Attention. The compact action head and
        # exact JVP switch back to FP32 inside the policy where millimeter-scale
        # numerical precision matters.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                action, action_mask, rgbs, rgb2d, pcds, instr, prop,
                run_inference=not training
            )
        return out  # loss if training, else action

    def train_one_step(
        self, model, optimizer, scaler, lr_scheduler, sample, step_id=0
    ):
        """Run a single training step."""
        optimizer.zero_grad(set_to_none=True)
        log_step = (
            step_id == 0
            or (step_id + 1) % self.args.diagnostic_interval == 0
        )
        actor = model.module if hasattr(model, "module") else model
        actor.collect_training_diagnostics = log_step and not self.args.use_compile

        # Forward pass
        loss = self._model_forward(model, sample)
        finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(
                f"Non-finite training loss at step {step_id + 1}."
            )

        # Backward pass
        scaler.scale(loss).backward()

        # Clip gradients
        scaler.unscale_(optimizer)
        group_norms = {}
        if log_step:
            grouped = {}
            for name, parameter in actor.named_parameters():
                if parameter.grad is not None:
                    grouped.setdefault(name.split(".")[0], []).append(
                        parameter.grad.detach().float().norm().square()
                    )
            group_norms = {
                f"grad_norm/{name}": torch.stack(norms).sum().sqrt()
                for name, norms in grouped.items()
            }
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        # Update
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        # Step the lr scheduler
        skipped = scaler.get_scale() < previous_scale
        if not skipped:
            lr_scheduler.step()

        if log_step:
            metrics = {
                "loss_total": loss.detach(),
                "grad_norm_before_clip": grad_norm,
                "gradient_clipped": (grad_norm > 10.0).float(),
                "optimizer_step_skipped": float(skipped),
                "amp_scale": scaler.get_scale(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "unique_data_passes": (
                    (step_id + 1) * self.global_batch_size
                    / self.unique_train_samples
                ),
                **group_norms,
            }
            metrics.update({
                f"loss/{name}": value
                for name, value in getattr(
                    actor, "loss_diagnostics", {}
                ).items()
            })
            metrics.update({
                f"flow/{name}": value
                for name, value in getattr(actor, "flow_diagnostics", {}).items()
            })
            world_size = dist.get_world_size()
            for name, value in metrics.items():
                value = torch.as_tensor(
                    value, device=loss.device
                ).detach().clone().float()
                dist.reduce(value, dst=0)
                if dist.get_rank() == 0:
                    self.writer.add_scalar(
                        f"training/{name}", value / world_size, step_id + 1
                    )

    @torch.inference_mode()
    def evaluate_nsteps(self, model, loader, step_id, val_iters, split='val'):
        """Run a given number of evaluation steps."""
        accumulator = ActionValidation()
        probes = {}
        device = next(model.parameters()).device
        model.eval()
        # Validation runs only on rank zero; bypass DDP forward synchronization.
        model = model.module if hasattr(model, "module") else model

        cuda_devices = [device.index] if device.type == "cuda" else []
        # Compare checkpoints with the same validation noise without perturbing
        # the RNG stream used by subsequent training batches.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(0)
            for i, sample in tqdm(enumerate(loader)):
                if i == val_iters:
                    break

                pred_action = self._model_forward(model, sample, training=False)
                gt_action = sample["action"].cuda(non_blocking=True)
                if self.args.relative_action:
                    pred_action = relative_to_absolute(
                        pred_action[:, :, 0],
                        sample["proprioception"].cuda(non_blocking=True)[:, :, 0]
                    )
                    gt_action = relative_to_absolute(
                        gt_action[:, :, 0],
                        sample["proprioception"].cuda(non_blocking=True)[:, :, 0]
                    )

                current_pose = sample["proprioception"].to(
                    device, non_blocking=True
                )[:, -1:]
                if pred_action.ndim == 3:
                    current_pose = current_pose[:, :, 0]
                accumulator.update(
                    pred_action, gt_action, current_pose, sample["task"]
                )
                if split == "val" and i < self.args.validation_probe_batches:
                    predictions = [pred_action]
                    # Extra trials do not alter later validation samples' RNG.
                    with torch.random.fork_rng(devices=cuda_devices):
                        torch.manual_seed(1000 + i)
                        for _ in range(
                            self.args.validation_noise_repeats - 1
                        ):
                            repeated = self._model_forward(
                                model, sample, training=False
                            )
                            if self.args.relative_action:
                                repeated = relative_to_absolute(
                                    repeated[:, :, 0], current_pose
                                )
                            predictions.append(repeated)
                    for name, value in noise_sensitivity(
                        predictions
                    ).items():
                        probes.setdefault(name, []).append(value.item())

        # Log all statistics
        values = accumulator.summarize(split)
        values.update({
            f"{split}-probe/{name}": float(np.mean(items))
            for name, items in probes.items()
        })
        if not accumulator.task_samples:
            raise RuntimeError(f"No samples were evaluated for {split}.")
        print(
            f"{split}: evaluated {sum(accumulator.task_samples.values())} "
            f"samples across {len(accumulator.task_samples)} tasks; "
            "these are offline checks, not simulator success rates."
        )
        if dist.get_rank() == 0:
            if step_id > -1:
                for key, val in values.items():
                    self.writer.add_scalar(key, val, step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                if key.startswith((
                    f"{split}-losses/mean/",
                    f"{split}-diagnostics/",
                    f"{split}-probe/",
                )):
                    print(f"{key}: {value:.03f}")

        # Select checkpoints by a task-balanced metric. A global sample mean is
        # easily dominated by coarse, high-success tasks and can improve while
        # insertion/stacking tasks remain unusable.
        task_scores = {
            key.split("/")[1]: value for key, value in values.items()
            if key.startswith(f'{split}-loss/')
            and key.endswith('/traj_score')
        }
        if not task_scores:
            raise RuntimeError(
                f"No per-task trajectory scores were produced for {split}."
            )
        (
            macro_score,
            worst_quartile_score,
            selection_score,
        ) = compute_task_balanced_selection(
            task_scores.values(),
            worst_fraction=0.25,
        )
        if dist.get_rank() == 0:
            print(
                "Checkpoint selection: "
                f"macro={macro_score:.4f}, "
                f"worst_quartile={worst_quartile_score:.4f}, "
                f"combined={selection_score:.4f}"
            )
            worst_tasks = sorted(
                task_scores.items(), key=lambda item: item[1], reverse=True
            )[:5]
            print(
                "Worst validation tasks: "
                + ", ".join(
                    f"{task}={score:.4f}" for task, score in worst_tasks
                )
            )
            if step_id > -1:
                self.writer.add_scalar(
                    f'{split}-losses/selection/macro_score',
                    macro_score,
                    step_id,
                )
                self.writer.add_scalar(
                    f'{split}-losses/selection/worst_quartile_score',
                    worst_quartile_score,
                    step_id,
                )
                self.writer.add_scalar(
                    f'{split}-losses/selection/combined_score',
                    selection_score,
                    step_id,
                )
        return selection_score

    def save_checkpoint(self, model, ema_model, optimizer, lr_scheduler, scaler,
                        step_id, new_loss, best_loss, rank_states):
        """Save complete, versioned training state with atomic replacement."""
        is_best = best_loss is None or new_loss <= best_loss
        if is_best:
            best_loss = new_loss
        checkpoint = build_training_checkpoint(
            model, ema_model, optimizer, lr_scheduler, scaler, self.ema,
            config=self.config, run_metadata=self.run_metadata, step=step_id + 1,
            best_loss=best_loss, rank_states=rank_states, steps_per_epoch=self.steps_per_epoch,
        )
        if is_best:
            atomic_save_checkpoint(checkpoint, self.args.log_dir / "best.pth")

        # Last checkpoint (always saved)
        atomic_save_checkpoint(checkpoint, self.args.log_dir / "last.pth")

        # Save intermediate checkpoints
        if (step_id + 1) % self.args.interm_ckpt_freq == 0:
            atomic_save_checkpoint(checkpoint, self.args.log_dir / f"interm{step_id + 1}.pth")

        return best_loss


def base_collate_fn(batch):
    """Custom collate_fn, measured to be faster than default."""
    _dict = {}

    # Values for these come as lists
    list_keys = ["task", "instr"]
    for key in list_keys:
        if key not in batch[0].keys():
            continue
        _dict[key] = []
        for item in batch:
            _dict[key].extend(item[key])

    # Treat rest as tensors
    _dict.update({
        k_: (
            torch.cat([item[k_] for item in batch])
            if batch[0][k_] is not None else None
        )
        for k_ in batch[0].keys() if k_ not in list_keys
    })

    return _dict


def actions_collate_fn(batch):
    return {"action": torch.cat([item["action"] for item in batch])}


def relative_to_absolute(action, proprio):
    # action (B, T, 8), proprio (B, 1, 7)
    pos = proprio[..., :3] + action[..., :3].cumsum(1)

    orn = proprio[..., 3:6] + action[..., 3:6].cumsum(1)
    orn = (orn + torch.pi) % (2 * torch.pi) - torch.pi

    return torch.cat([pos, orn, action[..., 6:]], -1)
