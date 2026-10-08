#!/usr/bin/env bash
# Install the pinned upstream policy implementation into its own environment.
set -euo pipefail
project_dir=$(cd "$(dirname "$0")/.." && pwd)
source "$project_dir/bin/inference_env.sh"
revision=$(cat "$project_dir/bin/predict2p5-revision.txt")
if [[ ! -d "$COSMOS_PREDICT2P5_DIR/.git" ]]; then
  git clone --no-checkout https://github.com/nvidia-cosmos/cosmos-predict2.5.git "$COSMOS_PREDICT2P5_DIR"
  git -C "$COSMOS_PREDICT2P5_DIR" checkout --detach "$revision"
fi
if [[ $(git -C "$COSMOS_PREDICT2P5_DIR" rev-parse HEAD) != "$revision" ]]; then
  printf 'Predict 2.5 checkout must match %s; inspect local changes before changing revisions.\n' "$revision" >&2
  exit 1
fi
for runtime_patch in "$project_dir/patches/predict2p5-tokenizer.patch" \
                     "$project_dir/patches/predict2p5-reason1.patch"; do
  if git -C "$COSMOS_PREDICT2P5_DIR" apply --check "$runtime_patch" 2>/dev/null; then
    git -C "$COSMOS_PREDICT2P5_DIR" apply "$runtime_patch"
  elif ! git -C "$COSMOS_PREDICT2P5_DIR" apply --reverse --check "$runtime_patch" 2>/dev/null; then
    printf 'Runtime patch %s conflicts with local changes; inspect the checkout.\n' "$runtime_patch" >&2
    exit 1
  fi
done
uv sync --project "$COSMOS_PREDICT2P5_DIR" --extra cu130 --group robocasa \
  --no-default-groups --python 3.10 --locked
if [[ ! -d "$project_dir/robocasa-cosmos-policy/.git" ]]; then
  git clone https://github.com/moojink/robocasa-cosmos-policy.git "$project_dir/robocasa-cosmos-policy"
fi
uv pip freeze --python "$PYTHON_BIN" --exclude-editable > "$project_dir/.cache/predict2.5-core-constraints.txt"
uv pip install --python "$PYTHON_BIN" \
  --constraints "$project_dir/.cache/predict2.5-core-constraints.txt" \
  --overrides "$project_dir/bin/robocasa-arm-overrides.txt" \
  -e "$project_dir/robocasa-cosmos-policy"
# Refresh library locations after sync has created the environment.
source "$project_dir/bin/inference_env.sh"
if [[ ! -f "$project_dir/.cache/ffmpeg6/lib/libavcodec.so.60" ]]; then
  bash "$project_dir/bin/build_arm_ffmpeg.sh"
fi
printf 'Predict 2.5 installed. Use %s for downloads and evaluation.\n' "$PYTHON_BIN"
