"""Step-8 four-cell recipe, explicit single-job launch, and coverage-checked report.

Planning never starts training. Launch is a preview unless --execute is given.
Generate manifests on the machine where they will run (paths are absolute).
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CELLS = (("film_tcn", "fm"), ("film_tcn", "meanflow"),
         ("transformer", "fm"), ("transformer", "meanflow"))


def canonical(value):
    return json.loads(json.dumps(value, default=str))


def flags(config):
    result = []
    for key, value in config.items():
        if value is not None:
            result.extend((f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)))
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def build_suite(name, tasks, gripper_profile, seeds=(0, 1, 2), eval_seeds=(0, 1, 2),
                eval_steps=(1, 2, 5), batch_size=64, train_iters=300000, lr=1e-4,
                project_dir=ROOT, demo_dir=None):
    from main import parse_arguments as parse_train
    from online_evaluation_rlbench.evaluate_policy import parse_arguments as parse_eval
    from utils.training_checkpoint import git_identity

    root = Path(project_dir).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Suite name must contain only letters, digits, underscore or hyphen.")
    if not tasks or len(set(tasks)) != len(tasks) or any(not re.fullmatch(r"[a-z0-9_]+", t) for t in tasks):
        raise ValueError("Provide unique RLBench task names (lowercase letters, digits, underscore).")
    for label, values, lower, upper in (("seeds", seeds, 0, 2**32-1), ("eval_seeds", eval_seeds, 0, 2**32-1),
                                       ("eval_steps", eval_steps, 1, 10000)):
        if not values or len(set(values)) != len(values) or any(type(v) is not int or not lower <= v <= upper for v in values):
            raise ValueError(f"{label} must contain unique integers in [{lower}, {upper}].")
    if batch_size < 1 or train_iters < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("batch_size/train_iters/lr must be positive and finite.")
    if gripper_profile not in ("compat", "plain"):
        raise ValueError("Explicitly choose gripper_profile=compat or plain; neither is an established winner.")
    common = dict(
        train_data_dir=root / "zarr_datasets/peract/train.zarr",
        eval_data_dir=root / "zarr_datasets/peract/val.zarr",
        train_instructions=root / "instructions/peract/instructions.json",
        val_instructions=root / "instructions/peract/instructions.json",
        dataset="Peract", model_type="denoise3d", batch_size=batch_size, batch_size_val=16,
        num_workers=4, chunk_size=1, memory_limit=8, train_iters=train_iters,
        lr=lr, backbone_lr=1e-6, lr_scheduler="cosine", wd=1e-4, use_ema=True, use_compile=False,
        lv2_batch_size=1, num_history=3, bimanual=False, keypose_only=True,
        pre_tokenize=True, backbone="clip", finetune_backbone=False, finetune_text_encoder=False,
        fps_subsampling_factor=4, embedding_dim=120, num_attn_heads=8,
        num_vis_instr_attn_layers=2, action_hidden_dim=256, action_num_blocks=6,
        attention_backend="math", matmul_precision="ieee", time_sampler="logit_normal",
        time_sampler_mean=0., time_sampler_std=1.5, flow_loss_type="l1",
        pose_position_weight=30., pose_rotation_weight=10., gripper_loss_weight=1.,
        endpoint_loss_weight=0., ivc_loss_weight=0., condition_dropout_prob=0.,
        gripper_prediction_mode="direct", gripper_loss_type="bce" if gripper_profile == "plain" else "weighted_bce",
        gripper_hold_prior_logit=0. if gripper_profile == "plain" else 2.,
        gripper_transition_weight=0. if gripper_profile == "plain" else 2.,
        gripper_closed_hold_weight=0. if gripper_profile == "plain" else 2.,
        guidance_scale=1., jvp_microbatch_size=8, denoise_timesteps=2,
        relative_action=False, rotation_format="quat_xyzw", workspace_normalizer_buffer=.04,
        val_freq=4000, val_batches=-1, validation_noise_repeats=3, validation_probe_batches=4,
        base_log_dir=root / "train_logs", exp_log_dir=name,
    )
    jobs = []
    for head, objective in CELLS:
        for seed in seeds:
            job_id = f"{head}-{objective}-s{seed}"
            train_flags = dict(common, action_head=head, flow_objective=objective, seed=seed,
                               meanflow_offdiag_ratio=.25 if objective == "meanflow" else 0., run_log_dir=job_id)
            argv = flags(train_flags)
            resolved = canonical(vars(parse_train(argv)))
            checkpoint = root / "train_logs" / name / job_id / "last.pth"
            evaluations = []
            for steps in eval_steps:
                for evaluation_seed in eval_seeds:
                    for task in tasks:
                        evaluation_id = f"{job_id}-{task}-n{steps}-e{evaluation_seed}"
                        output = root / "eval_logs" / name / (evaluation_id + ".json")
                        evaluation = {key: train_flags[key] for key in (
                            "model_type", "dataset", "bimanual", "backbone", "fps_subsampling_factor",
                            "embedding_dim", "num_attn_heads", "num_vis_instr_attn_layers", "num_history",
                            "action_hidden_dim", "action_num_blocks", "action_head", "flow_objective",
                            "attention_backend", "matmul_precision", "time_sampler", "time_sampler_mean",
                            "time_sampler_std", "meanflow_offdiag_ratio", "flow_loss_type",
                            "gripper_prediction_mode", "relative_action", "rotation_format", "guidance_scale")}
                        evaluation.update(checkpoint=checkpoint, task=task, seed=evaluation_seed,
                                          denoise_timesteps=steps, prediction_len=1, headless=True,
                                          max_steps=25, max_tries=10, collision_checking=False,
                                          gripper_command_mode="hysteresis", gripper_open_threshold=.75,
                                          gripper_close_threshold=.25, require_pose_for_gripper_open=True,
                                          data_dir=Path(demo_dir).resolve() if demo_dir else root / "online_evaluation_rlbench/demos",
                                          output_file=output)
                        eval_argv = flags(evaluation)
                        evaluations.append(dict(id=evaluation_id, output=str(output), argv=eval_argv,
                                                config=canonical(vars(parse_eval(eval_argv)))))
            jobs.append(dict(id=job_id, action_head=head, flow_objective=objective, seed=seed,
                             argv=argv, config=resolved, checkpoint=str(checkpoint), evaluations=evaluations))
    spec = canonical(dict(name=name, tasks=list(tasks), gripper_profile=gripper_profile, seeds=list(seeds),
                          eval_seeds=list(eval_seeds), eval_steps=list(eval_steps), batch_size=batch_size,
                          train_iters=train_iters, lr=lr, project_dir=str(root),
                          demo_dir=str(Path(demo_dir).resolve()) if demo_dir else None))
    suite = dict(version=1, spec=spec, source_sha256=git_identity(root)["source_sha256"], jobs=jobs,
                 notes=["Fixed-update, not equal-parameter or equal-wall-time comparison.",
                        "MF samples sorted pairs; its t marginal differs from FM single draws.",
                        "Endpoint/IVC disabled in all cells; gripper profile explicit, not validated as optimal.",
                        "Same integer seed does not imply identical initialization or noise across architectures.",
                        "Seeds and IEEE settings do not guarantee bitwise CUDA determinism.",
                        "Evaluation uses final-step last.pth EMA, matched tasks/episodes/seeds/NFE and execution rules."])
    suite["manifest_sha256"] = digest(suite)
    return suite


def load_suite(path):
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = suite.pop("manifest_sha256", None)
    if supplied != digest(suite):
        raise ValueError("Manifest changed or is incomplete; regenerate instead of editing individual cells.")
    suite["manifest_sha256"] = supplied
    # Regenerate the typed recipe: do not execute arbitrary commands from JSON.
    rebuilt = build_suite(**suite["spec"])
    if suite["jobs"] != rebuilt["jobs"]:
        raise ValueError("Manifest jobs disagree with the shared recipe.")
    return suite


def select_command(suite, job_id, evaluation_id=None):
    job = next((j for j in suite["jobs"] if j["id"] == job_id), None)
    if job is None:
        raise ValueError(f"Unknown training job: {job_id}")
    if evaluation_id is None:
        return job, None, [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "1",
                           "main.py", *job["argv"]]
    evaluation = next((e for e in job["evaluations"] if e["id"] == evaluation_id), None)
    if evaluation is None:
        raise ValueError(f"Unknown evaluation job: {evaluation_id}")
    return job, evaluation, [sys.executable, "-m", "online_evaluation_rlbench.evaluate_policy", *evaluation["argv"]]


def launch(suite, job_id, evaluation_id=None, execute=False):
    from utils.training_checkpoint import git_identity, read_checkpoint
    job, evaluation, command = select_command(suite, job_id, evaluation_id)
    print(shlex.join(command))
    if not execute:
        return  # No file writes, imports of models/datasets, or subprocess launch.
    root = Path(suite["spec"]["project_dir"])
    if not suite["source_sha256"] or git_identity(root)["source_sha256"] != suite["source_sha256"]:
        raise ValueError("Source changed after planning; regenerate the suite before executing.")
    if evaluation:
        for path in (evaluation["output"], evaluation["output"] + ".config.json"):
            if Path(path).exists():
                raise ValueError("Evaluation output already exists; refusing to overwrite.")
        checkpoint = read_checkpoint(job["checkpoint"])
        verify_checkpoint(suite, job, checkpoint)
        del checkpoint
        if not Path(evaluation["config"]["data_dir"]).is_dir():
            raise ValueError("Evaluation demo directory is missing.")
    else:
        directory = Path(job["checkpoint"]).parent
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Training directory is not empty; explicit strict resume is a separate workflow.")
        for name in ("train_data_dir", "eval_data_dir", "train_instructions", "val_instructions"):
            if not Path(job["config"][name]).exists():
                raise ValueError(f"Missing training input: {name}")
    env = dict(os.environ, PYTHONHASHSEED=str(evaluation["config"]["seed"] if evaluation else job["seed"]))
    subprocess.run(command, cwd=root, env=env, check=True)


def verify_checkpoint(suite, job, checkpoint):
    if checkpoint.get("iter") != suite["spec"]["train_iters"]:
        raise ValueError("Checkpoint is not at the predeclared final training step.")
    if checkpoint.get("ema_weight") is None:
        raise ValueError("Matrix evaluation requires the declared EMA weights.")
    config = checkpoint.get("config", {})
    if any(config.get(key) != value for key, value in job["config"].items()):
        raise ValueError("Checkpoint training recipe does not match this matrix job.")
    if checkpoint.get("run_metadata", {}).get("git", {}).get("source_sha256") != suite["source_sha256"]:
        raise ValueError("Checkpoint source differs from the planned source.")


def report(suite):
    """Only produce aggregates when every expected result has compatible provenance."""
    missing, rejected, records = [], [], []
    reference_coverage, reference_bounds, job_run_ids = {}, None, {}
    reference_dataset, reference_runtime, parameter_counts = None, None, {}
    for job in suite["jobs"]:
        for evaluation in job["evaluations"]:
            path = Path(evaluation["output"])
            sidecar = Path(str(path) + ".config.json")
            if not path.is_file() or not sidecar.is_file():
                missing.append(evaluation["id"])
                continue
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                if meta.get("evaluation_config") != evaluation["config"]:
                    raise ValueError("evaluation settings differ")
                if meta.get("evaluation_source_sha256") != suite["source_sha256"]:
                    raise ValueError("evaluation source differs")
                source = meta.get("checkpoint", {})
                if (source.get("iteration") != suite["spec"]["train_iters"] or source.get("selected_weights") != "EMA"
                    or source.get("path") != job["checkpoint"] or not source.get("run_id")
                    or not suite["source_sha256"] or source.get("source_sha256") != suite["source_sha256"]):
                    raise ValueError("checkpoint provenance differs or is missing")
                if job["id"] in job_run_ids and source["run_id"] != job_run_ids[job["id"]]:
                    raise ValueError("results mix separate training runs for the same job")
                if any(source.get("training_config", {}).get(k) != v for k, v in job["config"].items()):
                    raise ValueError("training recipe differs")
                dataset = source.get("dataset_counts")
                runtime = source.get("runtime")
                if (not isinstance(dataset, dict) or set(dataset) != {"train_samples", "validation_samples"}
                    or any(type(v) is not int or v <= 0 for v in dataset.values())):
                    raise ValueError("missing or invalid training dataset counts")
                if not runtime or (reference_runtime is not None and runtime != reference_runtime):
                    raise ValueError("training runtime differs or is missing")
                if reference_dataset is not None and dataset != reference_dataset:
                    raise ValueError("training dataset counts differ")
                params = source.get("parameter_counts")
                if not isinstance(params, dict) or any(type(params.get(k)) is not int or params[k] <= 0 for k in ("total", "trainable", "action_head")):
                    raise ValueError("missing parameter counts")
                if job["action_head"] in parameter_counts and params != parameter_counts[job["action_head"]]:
                    raise ValueError("parameter counts differ within the same architecture")
                bounds = source.get("workspace_normalizer")
                if (not isinstance(bounds, list) or len(bounds) != 2 or any(len(row) != 3 for row in bounds)
                    or not all(math.isfinite(v) for row in bounds for v in row)
                    or not all(a < b for a, b in zip(*bounds))):
                    raise ValueError("invalid workspace bounds")
                if reference_bounds is not None and bounds != reference_bounds:
                    raise ValueError("workspace normalizer differs across cells")
                task = evaluation["config"]["task"]
                coverage = meta.get("coverage", {}).get(task)
                if not isinstance(coverage, dict) or not coverage:
                    raise ValueError("missing episode coverage")
                counts = {}
                identities = {}
                for variation, count in coverage.items():
                    n, success = count["episodes"], count["successes"]
                    if type(n) is not int or type(success) is not int or n <= 0 or not 0 <= success <= n:
                        raise ValueError("invalid success/episode counts")
                    counts[variation] = n
                    ids = count.get("episode_ids")
                    if not isinstance(ids, list) or len(ids) != n or len(set(ids)) != n or any(not isinstance(i, str) for i in ids):
                        raise ValueError("missing or duplicate episode identities")
                    identities[variation] = ids
                if task in reference_coverage and identities != reference_coverage[task]:
                    raise ValueError("task variation/episode coverage differs across cells")
                if set(result) != {task} or set(result[task]) != set(coverage) | {"mean"}:
                    raise ValueError("score keys differ from declared coverage")
                expected = sum(v["successes"] for v in coverage.values()) / sum(counts.values())
                values = [(result[task]["mean"], expected)] + [(result[task][key], c["successes"]/c["episodes"]) for key, c in coverage.items()]
                # Existing result JSON rounds to two decimals; aggregate the
                # integer counts, not these display-rounded rates.
                if any(not math.isfinite(a) or not 0 <= a <= 1 or abs(a-b) > .0050001 for a, b in values):
                    raise ValueError("scores disagree with success counts")
                reference_coverage[task], reference_bounds = identities, bounds
                reference_dataset, reference_runtime = dataset, runtime
                parameter_counts[job["action_head"]] = params
                job_run_ids[job["id"]] = source["run_id"]
                records.append(dict(job=job["id"], head=job["action_head"], objective=job["flow_objective"],
                                    train_seed=job["seed"], steps=evaluation["config"]["denoise_timesteps"],
                                    task=task, success_rate=expected))
            except (ValueError, KeyError, TypeError, AttributeError, IndexError, OSError) as error:
                rejected.append(dict(id=evaluation["id"], reason=str(error)))
    aggregates = []
    if not missing and not rejected:
        for head, objective in CELLS:
            for steps in suite["spec"]["eval_steps"]:
                seed_means = [statistics.mean(r["success_rate"] for r in records if
                              (r["head"], r["objective"], r["steps"], r["train_seed"]) == (head, objective, steps, seed))
                              for seed in suite["spec"]["seeds"]]
                aggregates.append(dict(action_head=head, objective=objective, nfe=steps,
                                       macro_success_rate=statistics.mean(seed_means),
                                       std_across_training_seeds=statistics.stdev(seed_means) if len(seed_means) > 1 else None,
                                       training_seeds=len(seed_means)))
    return dict(status="complete" if not missing and not rejected else "incomplete", missing=missing, rejected=rejected,
                accepted_results=len(records), aggregates=aggregates, parameter_counts=parameter_counts,
                note="No aggregate or winner is inferred from an incomplete matrix. Matched NFE is not matched latency.")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--name", required=True)
    plan.add_argument("--tasks", nargs="+", required=True)
    plan.add_argument("--gripper-profile", choices=("compat", "plain"), required=True)
    plan.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    plan.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    plan.add_argument("--eval-steps", type=int, nargs="+", default=[1, 2, 5])
    plan.add_argument("--batch-size", type=int, default=64)
    plan.add_argument("--train-iters", type=int, default=300000)
    plan.add_argument("--lr", type=float, default=1e-4)
    plan.add_argument("--demo-dir", type=Path)
    plan.add_argument("--output", type=Path)
    run = commands.add_parser("launch")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--job", required=True)
    run.add_argument("--evaluation")
    run.add_argument("--execute", action="store_true")
    summary = commands.add_parser("report")
    summary.add_argument("--manifest", type=Path, required=True)
    summary.add_argument("--output", type=Path)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "plan":
            output = args.pop("output")
            result = build_suite(**args)
        elif command == "launch":
            launch(load_suite(args["manifest"]), args["job"], args["evaluation"], args["execute"])
            return
        else:
            output = args["output"]
            result = report(load_suite(args["manifest"]))
        if output:
            write_json(output, result)
            print(f"Wrote {output}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
