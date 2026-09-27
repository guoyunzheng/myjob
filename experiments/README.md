# Step 8: four-cell flow comparison

Step 9's opt-in iMF objective is documented in [IMF.md](IMF.md). It does not change this four-cell matrix or automatically launch additional runs.

This prepares **FiLM-TCN/FM, FiLM-TCN/MeanFlow, Transformer/FM, Transformer/MeanFlow**.
It does not establish which model is better and does not start training by default.
Generate the manifest on the Linux training machine: paths and source identity belong to that checkout.

## Fixed comparison recipe

- Same data paths, history=3, action representation, encoder configuration, batch, LR schedule, updates, EMA and validation settings.
- Three training seeds (0/1/2), three evaluation seeds (0/1/2), and equal inference budgets (1/2/5 head calls by default).
- No endpoint or IVC auxiliary losses in **any** cell. Keep base pose L1 weights 30/10.
- FP32 GEMM and cuDNN TF32 disabled for both heads; encoder BF16 unchanged.
- Choose a gripper profile explicitly: `compat` retains direct + weighted BCE + prior 2; `plain` uses direct + ordinary BCE + prior 0.
  Neither is a validated winner. The user's two September 5 gripper experiments reduced success rates.
- All cells use the same default hysteresis and opening gate at evaluation. Their independent ablations belong to step 7.
- Final-update `last.pth` **EMA**, not whichever checkpoint or inference step happens to score highest afterwards.
- Shared update and NFE budgets are not equal parameter counts, wall time, latency, or memory. Parameter counts are recorded separately.
- MeanFlow uses sorted time pairs and 25% off-diagonal samples; FM uses a single time draw. Their time marginals differ.
  Thus the objective comparison also includes this declared sampling recipe, not a claim of isolating only the JVP term.
- Fixed seeds control Python/NumPy/Torch initialization/noise and sampler/worker generators. They do not make CUDA bitwise deterministic,
  nor make differently shaped architectures consume identical random numbers. Existing validation noise seeds remain shared and fixed.

## Plan and inspect (no training)

```bash
python experiments/flow_matrix.py plan \
  --name step8_compat --gripper-profile compat \
  --tasks close_jar open_drawer \
  --demo-dir /absolute/path/to/peract_test \
  --output tmp/step8_compat.json

python experiments/flow_matrix.py launch \
  --manifest tmp/step8_compat.json --job film_tcn-fm-s0
```

Replace the example tasks with the **predeclared complete task set** for your experiment. Training still uses the entire configured
Peract Zarr dataset; `--tasks` selects online evaluation tasks, not a training-data filter.
Planning writes only the requested JSON and refuses to replace an existing manifest. Without `--output`, it prints JSON.
`launch` without `--execute` prints a command only. It never silently runs all cells or falls back to a smaller batch on OOM.

The plan accepts `--batch-size`, `--train-iters`, `--lr`, `--seeds`, `--eval-seeds`, and `--eval-steps`.
Change common settings by generating a **new named suite**, not by hand-editing individual jobs.
Use a short, small-batch suite first to check the complete CLIP/CUDA environment; do not mix its results with the full-budget suite.

## Execute one explicitly selected job later

```bash
# Starts one potentially long training job; add --execute only when ready.
python experiments/flow_matrix.py launch \
  --manifest tmp/step8_compat.json --job film_tcn-fm-s0 --execute

# Starts one simulation evaluation after the declared final checkpoint exists.
python experiments/flow_matrix.py launch \
  --manifest tmp/step8_compat.json --job film_tcn-fm-s0 \
  --evaluation film_tcn-fm-s0-close_jar-n1-e0 --execute
```

Jobs use the current Python environment, one GPU via torchrun, and `PYTHONHASHSEED` in each child process.
No environment installation, Git mutation, automatic resume, implicit initialization, or background services are performed.
Existing run directories/results, changed source, missing input paths, incomplete checkpoints, wrong recipes, or absent EMA fail explicitly.
Resume a stopped job only via the existing strict `main.py --resume` workflow with the same recipe; never reuse its partial checkpoint as final.
Do not use `best.pth` or mix independently restarted runs under one matrix job.

## Report

```bash
python experiments/flow_matrix.py report \
  --manifest tmp/step8_compat.json --output tmp/step8_compat_report.json
```

The reporter needs every expected score JSON and its `.config.json` sidecar. It checks training/evaluation settings, source hashes,
run identity, final update, EMA, workspace normalizer, runtime versions, dataset sample counts, and ordered variation/episode identities.
Zero-episode or skipped variations are rejected. Success rates are recomputed from integer counts because old display JSON rounds to 2 decimals.

Complete reports give task-macro success rate at each NFE, then mean and sample standard deviation across training seeds.
This is not a confidence interval; repeated evaluations reuse demos and are not independent new training runs.
An incomplete or inconsistent matrix has no aggregate or winner. Parameter counts do not constitute latency or peak-memory measurements.

Keep datasets immutable: matching paths, sample counts and episode identities **does not hash all data contents**. Keep hardware, dependencies,
evaluation/demo version, simulator configuration and training data snapshot fixed, and archive their external identifiers with the experiment.
Current tests use CPU and synthetic results; full CLIP/GPU/RLBench runs and success-rate conclusions remain outstanding.
