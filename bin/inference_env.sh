# Source from a shell to configure Predict 2.5 and shared workspace caches.
cosmos_project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export COSMOS_PREDICT2P5_DIR="$cosmos_project_dir/cosmos-predict2.5"
export PYTHON_BIN="$COSMOS_PREDICT2P5_DIR/.venv/bin/python"
# Upstream checkpoint loading invokes uvx as a subprocess. Batch jobs must find
# the tools installed with this environment, as well as its Python interpreter.
export PATH="$COSMOS_PREDICT2P5_DIR/.venv/bin:$PATH"
export UV_CACHE_DIR="$cosmos_project_dir/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$cosmos_project_dir/.cache/python"
export HF_HOME="$cosmos_project_dir/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME"
# Match the policy's pinned public Reason1 revision. Use the completed local
# snapshot directly so cache misses do not need the HF CLI's directory listing.
export COSMOS_REASON1_DIR="$HF_HUB_CACHE/models--nvidia--Cosmos-Reason1-7B/snapshots/3210bec0495fdc7a8d3dbb8d58da5711eab4b423"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
cosmos_library_dirs="$cosmos_project_dir/.cache/ffmpeg6/lib"
for cosmos_library_dir in "$COSMOS_PREDICT2P5_DIR"/.venv/lib/python*/site-packages/nvidia/*/lib; do
  if [[ -d "$cosmos_library_dir" ]]; then
    cosmos_library_dirs+=":$cosmos_library_dir"
  fi
done
export LD_LIBRARY_PATH="$cosmos_library_dirs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset cosmos_library_dirs cosmos_library_dir
unset cosmos_project_dir
