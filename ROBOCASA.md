# Predict 2.5 policy in RoboCasa

This workspace uses **`nvidia/Cosmos-Predict2.5-2B/robot/policy/robocasa`**, the
RoboCasa robot policy. NVIDIA reports 71.1% average success versus Predict 2's
67.1%, on 24 tasks with 50 training demonstrations per task.
See the [official recipe and results](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/predict2/cosmos_policy/post_training.html).

The implementation lives in `cosmos-predict2.5/`. The original root
`cosmos_policy/` implementation is Predict 2 and cannot load this policy by
changing only the checkpoint argument.

## Install and reuse the simulator

This workspace is already installed and validated: SLURM job `210050` completed
the one-trial `TurnOffMicrowave` evaluation successfully, and job `210201`
validated real Reason1 inference and disk-cache reuse for an uncached instruction.
Installation and download
commands below are for rebuilding; to run more evaluations, use **Submit one trial**.

Follow [SETUP.md](SETUP.md), or run from the workspace root:

```bash
bash bin/setup_predict2p5.sh
source bin/inference_env.sh
```

This creates a separate CUDA 13/Python 3.10 environment and installs the existing
RoboCasa fork. Keep `robocasa-cosmos-policy/`, its kitchen assets, and
`robocasa/macros_private.py`; both policy generations use them.

If assets or macros have not been installed yet:

```bash
"$PYTHON_BIN" robocasa-cosmos-policy/robocasa/scripts/download_kitchen_assets.py
"$PYTHON_BIN" robocasa-cosmos-policy/robocasa/scripts/setup_macros.py
```

The downloader prompts for confirmation and downloads several GB; run it in a
terminal and answer `y`. The macros script can also prompt for an existing config.
Skip these steps if your assets/macros are already complete. EGL variables are
set before simulator imports by the environment script and rollout launcher.

## Cache inference artifacts

```bash
source bin/inference_env.sh
"$PYTHON_BIN" bin/download_predict2p5.py --with-policy --with-reason1
```

The helper downloads:

* `nvidia/Cosmos-Predict2.5-2B/robot/policy/robocasa/model.pt`.
* `success_only/reason1_embeddings.pkl` from the RoboCasa dataset repository,
  copied to `.cache/predict2.5/reason1_embeddings.pkl`.
* `robocasa_dataset_statistics.json` from the original policy repository,
  copied to `.cache/predict2.5/robocasa_dataset_statistics.json`. These are shared
  action/proprio scaling statistics, not Predict 2 policy weights.
* The shared Wan tokenizer from `nvidia/Cosmos-Predict2.5-2B/tokenizer.pth`.
  The setup applies [patches/predict2p5-tokenizer.patch](patches/predict2p5-tokenizer.patch)
  to replace the old repository URL. Both files have SHA-256
  `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`,
  verified from Hugging Face LFS metadata. No tokenizer weights change.
* The public `nvidia/Cosmos-Reason1-7B` encoder and tokenizer, pinned to revision
  `3210bec0495fdc7a8d3dbb8d58da5711eab4b423` (approximately 15.45 GiB).
  Setup applies `patches/predict2p5-reason1.patch` to register the tokenizer's
  public mapping and save cache misses in atomic individual files.

The official Reason1 cache is approximately 25 GB. Upstream preloads all its
embeddings onto the GPU; the 96 GB SLURM memory request provides room for both
CPU/GPU copies on GB10 unified memory and the policy. The helper keeps a writable
local copy as well as the download cache, requiring roughly 50 GB of disk for
embeddings alone.

Training trajectories are unnecessary for evaluation. Predict 2.5 uses **Reason1
embeddings**, replacing T5. Cached benchmark instructions avoid loading the full
Reason1 encoder; novel instructions need matching embeddings. The CLI argument
is still named `--t5_text_embeddings_path`.
Generated embeddings are reused from `.cache/predict2.5/reason1_embeddings.pkl.extra/`;
concurrent evaluations do not rewrite the shared 25 GB base file.

## Submit one trial

From the workspace root:

```bash
source bin/inference_env.sh
bash slurm/gpu-job.sh --time=01:00:00 --mem=96G -- \
  bin/run_robocasa_predict2p5.py \
  --task_name TurnOffMicrowave --num_trials_per_task 1
```

[bin/run_robocasa_predict2p5.py](bin/run_robocasa_predict2p5.py) changes into the
upstream checkout, invokes its evaluator, and supplies the full recipe defaults.
The SLURM launcher honors `PYTHON_BIN` from the environment script.
CLI arguments override the helper's defaults.

| Setting | Predict 2.5 value |
| --- | --- |
| Config | `cosmos_predict2p5_2b_480p_robocasa_50_demos_per_task__inference_no_s3` |
| Checkpoint | `nvidia/Cosmos-Predict2.5-2B/robot/policy/robocasa` |
| Text embeddings | `reason1` |
| Rectified-flow shift | `5` |
| Action chunk / executed steps | `32` / `16` |
| Action denoising steps | `5` |
| Cameras | One wrist, two third-person |
| Proprio/actions | Normalize proprio, unnormalize actions |
| Images | JPEG compression and vertical flip |
| Seed / deterministic | `195` / `True` |
| Planning / parallel inference | Disabled |

The `no_s3` config disables object-store callbacks and uses cached text embeddings.
Uncached task instructions use the public Reason1 encoder/tokenizer; cache them
before evaluation with `bin/download_predict2p5.py --with-policy --with-reason1`
using `$PYTHON_BIN` after sourcing `bin/inference_env.sh`.
Both regular and `no_s3` configs exist in the pinned source.

Monitor the returned job ID:

```bash
bash slurm/monitor-job.sh JOB_ID
cat /spatial-storage/raymond/logs/spatial-gpu-JOB_ID.err
```

Videos and results go to `results/robocasa-predict2.5/`. A completed trial validates
rendering, model loading, and action generation together; it is not a benchmark
success-rate measurement. Adjust the initial one-hour/96 GB allocation after
measuring runs.

Also validate a task with an instruction absent from the downloaded cache:

```bash
bash slurm/gpu-job.sh --time=01:00:00 --mem=96G -- \
  bin/run_robocasa_predict2p5.py --task_name PnPCabToCounter \
  --num_trials_per_task 1 --seed 195 --run_id_note reason1-cache-miss-check
```

The first seed-195 episode requests "pick the candle from the cabinet and place
it on the counter", which exposed the missing Reason1 tokenizer mapping in job
210103. A microwave-only smoke test does not exercise this fallback.

Once one trial works, use 50 trials per task and compare seeds 195, 196, and 197
for the published protocol:

```bash
bash slurm/gpu-job.sh --time=04:00:00 --mem=96G -- \
  bin/run_robocasa_predict2p5.py --task_name TurnOffMicrowave \
  --num_trials_per_task 50 --seed 195 --run_id_note predict2p5-50trials-seed195
```

Four hours is an example allocation, not an established GB10 runtime.

### Submit the complete benchmark

```bash
bash bin/submit_robocasa_full_eval.sh
```

This submits 72 jobs: 24 tasks, 50 trials per task, and seeds 195, 196, and 197.
Each job explicitly requests one node, one task, one GPU, four CPUs, 96 GiB of
memory, and a 24-hour time limit. The successful GB10 smoke run used four CPUs
and peaked at 43.77 GiB RSS. The longer allocation allows for 50 episodes, task
horizons up to 1,000 steps, and renderer retries; full task runtimes remain
estimates until measured. Jobs release resources when finished, even if they
finish before the limit. A longer time request can increase queue wait times.

Results are separated by task and seed under
`results/robocasa-predict2.5/full-eval/`. The script uses the same policy settings
as the verified one-trial evaluator, with planning and parallel inference disabled.

Monitor the latest submitted batch until every job finishes:

```bash
bash bin/monitor_robocasa_full_eval.sh
```

The table shows each job's task, seed, state, elapsed time, exit code, and node
or pending reason. It refreshes every ten seconds; Ctrl-C detaches without
cancelling jobs. The monitor returns the prompt when all jobs finish, with a
nonzero exit status if a job failed or submission stopped early. Use `--once`
for a single snapshot, or `--interval 30` to refresh less often.

The submission script records accepted job IDs in
`results/robocasa-predict2.5/full-eval/submissions/<batch>/jobs.tsv` and prints
the exact monitoring command for that batch. Pass this manifest as an argument
to monitor an older batch. If no manifest exists, the monitor recovers this
workspace's full-evaluation jobs submitted today from SLURM accounting. For
jobs submitted before tracking was added, explicit recovery is also available:

```bash
bash bin/monitor_robocasa_full_eval.sh --discover --since 2026-10-07
```

Recovery includes all matching submissions since the given date, including
completed jobs. To stream one job's logs, use `bash slurm/monitor-job.sh JOB_ID`.

After the batch finishes, calculate the benchmark numbers:

```bash
python3 bin/summarize_robocasa_full_eval.py
```

This reads the latest batch's job IDs and their SLURM stdout logs. It prints
the overall success rate across 3,600 episodes, each seed's score across 24
tasks, the sample standard deviation of the three seed scores, and per-task
scores. With 50 episodes in every task/seed combination, the pooled success
rate equals the mean of the task and seed success rates. Success counts are
used directly rather than averaging the evaluator's rounded percentages.

Reports are saved beside that batch's `jobs.tsv`: `summary.txt` for reading,
`summary.json` for further analysis, `results.csv` for all 72 evaluations, and
`task_results.csv` for the 24 task scores. Pass a specific `jobs.tsv` to summarize
an older batch. The script uses that batch's job logs, so reruns in the same
task/seed output folders cannot mix results. No GPU allocation is needed.

Missing runs, jobs without the launcher's `JOB_OK` marker, and runs with fewer
than 50 episodes are listed as incomplete. The final score is withheld and
the script exits nonzero until all 72 runs are complete. Duplicate task/seed
entries are rejected; choose a manifest containing exactly one benchmark batch.

## Direct upstream evaluation

The helper runs the equivalent of this command from **inside**
`cosmos-predict2.5/` (remaining recipe flags are in the helper):

```bash
source ../bin/inference_env.sh
"$PYTHON_BIN" -m cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
  --config cosmos_predict2p5_2b_480p_robocasa_50_demos_per_task__inference_no_s3 \
  --config_file cosmos_predict2/_src/predict2/cosmos_policy/config/config.py \
  --ckpt_path nvidia/Cosmos-Predict2.5-2B/robot/policy/robocasa \
  --dataset_stats_path "$PWD/../.cache/predict2.5/robocasa_dataset_statistics.json" \
  --t5_text_embeddings_path "$PWD/../.cache/predict2.5/reason1_embeddings.pkl" \
  --text_embeddings_kind reason1 --shift 5 \
  --task_name TurnOffMicrowave --num_trials_per_task 1
```

## Training (optional)

Use the [official Predict 2.5 training recipe](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/predict2/cosmos_policy/post_training.html#training-instructions)
and `scripts.train` in the **new checkout**. The experiment is
`cosmos_predict2p5_2b_480p_robocasa_50_demos_per_task_no_s3`, using config
`cosmos_predict2/_src/predict2/cosmos_policy/config/config.py`.
Do not use the old `cosmos_policy.scripts.train` command.

Training needs the full `nvidia/RoboCasa-Cosmos-Policy` dataset including Reason1
embeddings and a Predict 2.5 **base pretrained** checkpoint. The `no_s3` training
config retains an internal checkpoint path; override `checkpoint.load_path`
with the downloaded local base `.pt` path outside NVIDIA. The RoboCasa policy
weights above are for evaluation or deliberate continuation of policy training.

The reference run uses global batch size 800 (25 per GPU across 32 H100s), 45K
steps, learning rate `2e-4`, and decay around 42.8K steps. With eight GPUs,
gradient accumulation of four preserves the batch size. This ARM workspace
migration validates inference, not multi-node training.

## Predict 2 cleanup

Follow [PREDICT2_MIGRATION.md](PREDICT2_MIGRATION.md). Retain the simulator,
shared caches, tokenizer, and local scaling statistics; remove obsolete
environments and policy weights after the new rollout works.
