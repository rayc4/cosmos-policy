#!/usr/bin/env bash
# Submit the 24-task Predict 2.5 benchmark: 50 trials for each of three seeds.
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source bin/inference_env.sh
"$PYTHON_BIN" bin/check_predict2p5_assets.py

# Persist every accepted job immediately, including a partially submitted batch.
submission_dir="$project_dir/results/robocasa-predict2.5/full-eval/submissions/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$submission_dir"
manifest="$submission_dir/jobs.tsv"
printf 'SUBMITTING\n' > "$submission_dir/submission.state"
printf 'job_id\ttask\tseed\n' > "$manifest"
trap 'status=$?; if (( status != 0 )); then printf "FAILED\n" > "$submission_dir/submission.state"; fi' EXIT
printf 'Job manifest: %s\n' "$manifest"

# The successful GB10 rollout peaked at 43.77 GiB RSS with four CPU cores.
# Keep headroom for long episodes and unified GPU memory. Fifty trials can
# reach 1,000 steps each and retry rendering up to three times; reserve a full
# day rather than extrapolating only from a short successful episode.
job_time=24:00:00
job_memory=96G
job_cpus=4

tasks=(
  PnPCounterToCab PnPCabToCounter PnPCounterToSink PnPSinkToCounter
  PnPCounterToMicrowave PnPMicrowaveToCounter
  PnPCounterToStove PnPStoveToCounter
  OpenSingleDoor CloseSingleDoor OpenDoubleDoor CloseDoubleDoor
  OpenDrawer CloseDrawer TurnOnStove TurnOffStove
  TurnOnSinkFaucet TurnOffSinkFaucet TurnSinkSpout
  CoffeeSetupMug CoffeeServeMug CoffeePressButton
  TurnOnMicrowave TurnOffMicrowave
)

for seed in 195 196 197; do
  for task in "${tasks[@]}"; do
    printf 'Submitting %s, seed %s, 50 trials\n' "$task" "$seed"
    job_id=$(bash slurm/gpu-job.sh --parsable \
      --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task="$job_cpus" --mem="$job_memory" --time="$job_time" -- \
      bin/run_robocasa_predict2p5.py \
      --task_name "$task" \
      --num_trials_per_task 50 \
      --seed "$seed" \
      --run_id_note "full-eval-seed-$seed" \
      --local_log_dir "$project_dir/results/robocasa-predict2.5/full-eval/$task/seed-$seed")
    # --parsable can return JOB_ID;CLUSTER on a federated installation.
    job_id=${job_id%%;*}
    if [[ ! $job_id =~ ^[0-9]+$ ]]; then
      printf 'Unexpected sbatch job ID: %s\n' "$job_id" >&2
      exit 1
    fi
    printf '%s\t%s\t%s\n' "$job_id" "$task" "$seed" >> "$manifest"
    printf 'Submitted batch job %s\n' "$job_id"
  done
done
printf 'COMPLETE\n' > "$submission_dir/submission.state"
printf 'Monitor this batch: bash bin/monitor_robocasa_full_eval.sh %q\n' "$manifest"
