# Predict 2 to Predict 2.5 migration

The active RoboCasa runtime is now `cosmos-predict2.5/.venv`, with CUDA 13,
PyTorch 2.9.1, Reason1 embeddings, and the official 2.5 RoboCasa policy.
The upstream source revision is in `bin/predict2p5-revision.txt`.
[SETUP.md](SETUP.md) and [ROBOCASA.md](ROBOCASA.md) contain the new commands.

## What changed

* `bin/inference_env.sh` selects the new interpreter and CUDA library directories.
* `bin/setup_predict2p5.sh` installs pinned upstream dependencies and the existing
  simulator in the new environment.
* `bin/download_predict2p5.py --with-policy --with-reason1` caches the required
  policy, metadata, and Reason1 encoder/tokenizer for uncached instructions.
* `bin/check_predict2p5_assets.py` verifies all local inference assets and patches
  before full-benchmark submission.
* `patches/predict2p5-tokenizer.patch` points the upstream tokenizer config to
  the identical file in the 2.5 repository, removing a separate Predict 2
  license/cache dependency.
* `patches/predict2p5-reason1.patch` registers the omitted public Reason1 tokenizer
  mapping; the policy already registers the corresponding encoder weights. It
  also saves newly computed embeddings in separate atomic files so parallel
  jobs preserve the shared base cache and one another's additions.
* `bin/run_robocasa_predict2p5.py` uses the new evaluator/config, Reason1
  embeddings, and shift 5. It changes into the upstream checkout for asset paths.
* `bin/check_inference.py` validates the new imports, video decoding, and GPU.
* Root `pyproject.toml`, `uv.lock`, `cosmos_policy/`, and `docker/` are historical
  Predict 2 source. New launchers do not install/import them.
* `LIBERO.md` and `ALOHA.md` are retained as legacy Predict 2 recipes. This
  migration targets RoboCasa; 2.5 also has a separate LIBERO policy, but the
  published 2.5 recipe does not provide an ALOHA checkpoint.

No old environment, cached weights, simulator assets, or rollout results were
deleted during migration.

## Undo old environment settings

Close any shell with the old environment activated, or run `deactivate` and
start a fresh shell. Remove old shell-startup lines or job exports that:

* Activate `/spatial-storage/raymond/projects/cosmos-policy/.venv/bin/activate`.
* Set `PYTHON_BIN` to the root `.venv/bin/python`.
* Add the root `.venv/.../nvidia/cuda_runtime/lib` to `LD_LIBRARY_PATH`.
* Submit `cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py` with a
  `cosmos_predict2_2b_...` config or Predict 2 policy checkpoint.

Replace them with `source bin/inference_env.sh` from this workspace and
`bin/run_robocasa_predict2p5.py`. Fresh shells avoid retaining old CUDA library
paths when the new script extends `LD_LIBRARY_PATH`.

## Remove the old environment

After the new GPU check and one-trial rollout pass, stop any old jobs/processes
using the root environment. From the workspace root, this removes **only** the
old environment (about 13 GB before migration):

```bash
rm -rf -- /spatial-storage/raymond/projects/cosmos-policy/.venv
```

Keep `cosmos-predict2.5/.venv`. Never run the old root `uv sync` recipe afterward:
it recreates the old environment. Use `bash bin/setup_predict2p5.sh`.

The old `.cache/robocasa-core-constraints.txt` is obsolete and can be removed:

```bash
rm -f -- /spatial-storage/raymond/projects/cosmos-policy/.cache/robocasa-core-constraints.txt
```

Keep `.cache/predict2.5-core-constraints.txt`, used by the new setup.
The old `.cache/ffmpeg/` (FFmpeg 4) and `.cache/ffmpeg-build.log` can also be
removed once no Predict 2 process needs them; Predict 2.5 uses `.cache/ffmpeg6/`.

## Remove obsolete checkpoints

The original workspace Hugging Face cache contained no policy weight directories
when migration began, so there may be no old weights to remove here.
Other caches used by earlier jobs may contain them.

Inspect these possible cache locations:

```bash
find .cache/huggingface /spatial-storage/raymond/cache/huggingface \
  -maxdepth 2 -type d -name 'models--nvidia--Cosmos-Policy-*-Predict2-2B'
```

The new helper copies normalization statistics to
`.cache/predict2.5/robocasa_dataset_statistics.json`. Once that file exists, the
old `models--nvidia--Cosmos-Policy-RoboCasa-Predict2-2B` cache directory can be
removed, including its old policy weights/T5 embeddings. The downloader may
recreate a tiny statistics-only cache under that name when rerun.

Old LIBERO/ALOHA policy caches are removable only if you no longer use their
legacy recipes. Delete only the specific obsolete model directory you identify;
do not clear the entire Hugging Face cache.

**Keep these shared/new artifacts:**

| Artifact | Why retain it |
| --- | --- |
| `models--nvidia--Cosmos-Predict2.5-2B` | New RoboCasa policy and identical shared tokenizer |
| `models--nvidia--Cosmos-Reason1-7B` | Encoder and tokenizer for uncached instructions |
| `datasets--nvidia--RoboCasa-Cosmos-Policy` | Reason1 embeddings; also training data if downloaded |
| `.cache/predict2.5/` | Local statistics, base embeddings, and generated instruction embeddings |
| `robocasa-cosmos-policy/` | Simulator, kitchen assets, macros |
| `.cache/ffmpeg6/` | New CUDA 13 Decord shared libraries |
| `.cache/python/` | New environment may link to these interpreters |
| `.cache/uv/` | Shared package cache; optional selective pruning later |
| `slurm/` | Shared cluster launcher, not Predict 2 specific |

The old `models--nvidia--Cosmos-Predict2-2B-Video2World` base-model cache is also
removable once the new tokenizer is cached. The applied patch uses the identical
file in `models--nvidia--Cosmos-Predict2.5-2B` instead. Keep the new cache; delete
only the obsolete model directory rather than individual shared blobs.

## Original source checkout

Keeping the root Predict 2 source costs little relative to environments/weights
and preserves existing user edits and provenance. It is no longer the active
runtime. Do not `git reset`/restore the root dependency files as a cleanup step:
that would discard existing ARM setup edits, not uninstall packages.

To eventually remove this entire outer checkout, first move the 2.5 checkout,
simulator/assets, caches, helpers, and docs into a standalone workspace and update
their paths. The current helpers deliberately resolve these paths from the outer
workspace. Deleting the outer directory now would also delete the new setup.

## Validation status

On October 7, 2026, the new environment and inference artifacts were installed.
Shell/Python syntax, launcher argument overrides against upstream config fields,
Torch 2.9.1/CUDA 13 imports, RoboCasa/MuJoCo imports, and a real FFmpeg 6/Decord
video decode passed. The tokenizer files' identical LFS SHA-256 hashes were
verified before changing their repository URL.

SLURM job **210012** completed successfully, including the policy/simulator
imports, CUDA tensor test, video decoding, and FlashAttention kernel comparison.
The first rollout (**210013**) failed while loading the tokenizer because its
subprocess could not find `uvx` in the batch job's PATH. `bin/inference_env.sh`
now adds the new environment's command-line tools to PATH, and the GPU checker
checks that `uvx` is discoverable. The exact tokenizer CLI command passed after
this correction.

Replacement one-trial rollout **210050** completed with exit code 0 in 6 minutes
47 seconds. It successfully solved `TurnOffMicrowave` in 260 simulator steps and
saved rollout videos and results under `results/robocasa-predict2.5/`. This confirms
the cached-instruction RoboCasa inference path on GB10; one episode is not a
benchmark estimate.

Full-evaluation job **210103** later exposed an uncached instruction. The policy
had registered the public Reason1 model but omitted its tokenizer mapping.
`patches/predict2p5-reason1.patch` fixes that mapping, honors the pinned local
snapshot for model and tokenizer loading, and saves new embeddings in atomic
individual files instead of rewriting the shared 25 GB cache. The public
15.45 GiB Reason1 snapshot is now cached locally. Full-benchmark submission
checks these assets and patches before requesting any compute.

At the user's request, all original benchmark jobs were stopped: **69 cancelled**
and **3 already failed**. Offline GPU validation **210201** then completed with
exit code 0 in **2 minutes 57 seconds**, exercising the actual Reason1 model on
the previously missing candle instruction and verifying persistent cache reuse.
The resulting embedding has shape `(1, 512, 100352)` and BF16 dtype. No benchmark
jobs were restarted. The 50-trial benchmark and joint policy/encoder memory peak
have not yet been measured to completion.

```bash
sacct -j 210012,210013,210050,210201 --format=JobID,State,ExitCode,Elapsed
bash slurm/monitor-job.sh 210201
```

Required inference assets and runtime fixes are installed. The policy rollout
and uncached-instruction Reason1 path have both passed GPU checks. Use
`bash bin/submit_robocasa_full_eval.sh` when ready to restart the benchmark.
The old environment can be removed using the cleanup instructions above.
See SETUP.md for the known inherited metadata conflicts.
