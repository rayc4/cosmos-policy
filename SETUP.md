# Predict 2.5 workspace setup

This workspace now runs the RoboCasa **Predict 2.5 robot policy** from the official
[nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)
repository. The original `cosmos_policy/` source and root `pyproject.toml` describe
Predict 2 and are retained as historical source; they do not install the new runtime.
Use the commands below instead of `uv sync` in this directory.

## ARM / GB10 installation

Run from `/spatial-storage/raymond/projects/cosmos-policy`:

```bash
bash bin/setup_predict2p5.sh
source bin/inference_env.sh
```

The script creates `cosmos-predict2.5/.venv`, using Python 3.10, CUDA 13, PyTorch
2.9.1, and the upstream locked RoboCasa dependencies. A reproducible patch loads
the identical Wan tokenizer from the authorized 2.5 repository, removing a
separate Predict 2 access requirement. The source revision is in
[bin/predict2p5-revision.txt](bin/predict2p5-revision.txt). It installs the existing
`robocasa-cosmos-policy/` clone into the new environment. The root `.venv` remains
the old Predict 2 environment.

NVIDIA's current [setup guide](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/setup.md)
requires CUDA 13 for DGX Spark and Jetson AGX. The previous CUDA 12.8 workspace
recipe is superseded. Keep Python 3.10 for the selected ARM extension wheels.

`bin/inference_env.sh` selects the new interpreter through `PYTHON_BIN`, adds its
command-line tools to `PATH` (the tokenizer loader invokes `uvx`), and shares the
workspace Hugging Face/uv/Python caches, sets EGL rendering, and adds the new
environment's CUDA library directories to `LD_LIBRARY_PATH`. Source it in each
new shell/job. It does not activate an environment or change working directory.

The CUDA 13 ARM Decord wheel needs FFmpeg 6 (`libavcodec.so.60`), unlike the
old CUDA 12.8 wheel. The setup script builds it into `.cache/ffmpeg6/` when missing.
To rebuild it manually:

```bash
bash bin/build_arm_ffmpeg.sh
source bin/inference_env.sh
```

RoboCasa's older NumPy/Numba/Protobuf pins conflict with the policy environment.
[bin/robocasa-arm-overrides.txt](bin/robocasa-arm-overrides.txt) preserves the new
locked versions while installing the simulator, including upstream's cuDNN 9.21
override of Torch's older declared pin. Metadata checks can report these
intentional conflicts and Megatron Core's `numpy<2` requirement. Runtime validation
is required. Upstream also declares `cosmos-oss==0.1.0` while its workspace
package reports 1.5.0, and the cuSPARSELt ARM wheel uses an `sbsa` metadata tag
that `uv pip check` flags; its actual shared library is verified ARM aarch64.
These metadata issues are inherited from the pinned upstream dependency set.
A later `uv sync --project cosmos-predict2.5` removes the editable
simulator installation; rerun `bash bin/setup_predict2p5.sh` afterward.

## Download policy artifacts

```bash
source bin/inference_env.sh
"$PYTHON_BIN" bin/download_predict2p5.py --with-policy --with-reason1
"$PYTHON_BIN" bin/check_predict2p5_assets.py
```

This downloads the RoboCasa policy weights, shared Wan tokenizer,
normalization statistics, precomputed Reason1 embeddings, and the pinned public
Reason1 7B encoder/tokenizer. It does not download training trajectories or the
generic video-generation model. Reason1 is needed when a randomized task
instruction is absent from the precomputed embeddings. `--reason1-only` caches
just this encoder/tokenizer when the other assets are already installed.

Setup applies `patches/predict2p5-reason1.patch`, which adds the tokenizer's
missing public checkpoint mapping using the same repository and revision as
upstream's shared checkpoint registry. The full-evaluation submitter checks all
local assets and patches before submitting any jobs.
New instruction embeddings are saved as atomic individual files under
`.cache/predict2.5/reason1_embeddings.pkl.extra/`; parallel jobs reuse these
without rewriting the 25 GB base embedding cache.
`bin/inference_env.sh` sets `COSMOS_REASON1_DIR` to the pinned local snapshot;
the runtime patch honors this for model and processor loading, including in
`HF_HUB_OFFLINE=1` mode.

If Hugging Face rejects a download, authenticate with
`cosmos-predict2.5/.venv/bin/hf auth login` and accept the relevant repository's
license/access terms. Credentials remain in the shared cache.
Kitchen assets and simulator macros are covered in [ROBOCASA.md](ROBOCASA.md).

## SLURM validation and evaluation

To check the actual Reason1 forward pass and disk-cache reuse without loading
the 25 GB base embeddings or the policy model:

```bash
source bin/inference_env.sh
bash slurm/gpu-job.sh --cpus-per-task=4 --mem=40G --time=00:20:00 -- \
  bin/check_reason1_inference.py
```

This checks the candle instruction that caused job 210103 to fail. Use the
uncached-instruction rollout in [ROBOCASA.md](ROBOCASA.md) to validate the policy
and encoder together with the full 96 GiB allocation.
Job **210201** passed this encoder-only check completely offline on October 7,
2026, finishing in 2 minutes 57 seconds with exit code 0.

From the workspace root:

```bash
source bin/inference_env.sh
bash slurm/gpu-job.sh --time=00:15:00 --mem=32G -- bin/check_inference.py
```

The shared launcher sources `bin/inference_env.sh` and honors `PYTHON_BIN`, selecting
the new environment. The checker tests imports, video decoding, GPU extension
imports, and a CUDA tensor operation without downloading weights. Its filename
is unchanged, but it now imports Predict 2.5.

For a complete one-trial rollout:

```bash
bash slurm/gpu-job.sh --time=01:00:00 --mem=96G -- \
  bin/run_robocasa_predict2p5.py
```

Follow either returned job ID:

```bash
bash slurm/monitor-job.sh JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,NodeList
```

`monitor-job.sh` replays and streams stdout/stderr until the job finishes, then
returns the job's exit status to your shell. It waits quietly for logs while a
job is pending and uses accounting for completed jobs. Ctrl-C detaches without
cancelling the job. For completed jobs with custom log paths, pass
`--stdout /path/to/job.out --stderr /path/to/job.err`. Set `POLL_INTERVAL` to
change the default five-second polling interval.

SLURM logs are under `/spatial-storage/raymond/logs/spatial-gpu-JOB_ID.out` and
`.err`; rollouts go to `results/robocasa-predict2.5/`. These initial allocations
are starting values, not measured policy memory/runtime requirements.
Keep SLURM's `CUDA_VISIBLE_DEVICES`; render and run inference inside the GPU job.

For an interactive GPU shell:

```bash
srun -p gb10 -J cosmos-predict2p5 -N 1 -n 1 -c 4 \
  --mem=96G --gres=gpu:1 --time=01:00:00 --chdir="$PWD" --pty bash -l
source bin/inference_env.sh
"$PYTHON_BIN" bin/check_inference.py
"$PYTHON_BIN" bin/run_robocasa_predict2p5.py
```

## Connecting another framework

Run with the new interpreter and `cosmos-predict2.5/` as the working directory.
Import `get_model`, `load_dataset_stats`, `init_t5_text_embeddings_cache`, and
`get_action` from
`cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils`.
Use the RoboCasa evaluator's `PolicyEvalConfig` and the defaults in
[bin/run_robocasa_predict2p5.py](bin/run_robocasa_predict2p5.py). Initialize the cache
with `embeddings_kind="reason1"`; its function/argument retain historical T5 names.
Load the model once and reuse it across observations.

The old ALOHA HTTP server is a Predict 2 interface. This migration targets RoboCasa;
use its Python API for a matching robot/camera/action layout. A different robot
generally needs its own compatible fine-tuned checkpoint.

## Docker alternative

Build inside the **Predict 2.5 checkout**, following its
[setup guide](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/setup.md).
For Blackwell, upstream uses `docker/nightly.Dockerfile`. The root `docker/`
belongs to Predict 2. The commands above use native ARM and SLURM.

## Remove the old setup

See [PREDICT2_MIGRATION.md](PREDICT2_MIGRATION.md) for cleanup paths and shared
artifacts that the new policy still needs.
