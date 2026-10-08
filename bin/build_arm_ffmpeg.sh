#!/usr/bin/env bash
# Build FFmpeg 6 for the Predict 2.5 CUDA 13 ARM Decord wheel.
set -euo pipefail
project_dir=$(cd "$(dirname "$0")/.." && pwd)
build_dir=$(mktemp -d /tmp/cosmos-ffmpeg.XXXXXX)
trap 'rm -rf "$build_dir"' EXIT
curl -fsSL --retry 3 https://ffmpeg.org/releases/ffmpeg-6.1.2.tar.xz -o "$build_dir/ffmpeg.tar.xz"
tar -xJf "$build_dir/ffmpeg.tar.xz" -C "$build_dir"
cd "$build_dir/ffmpeg-6.1.2"
./configure \
  --prefix="$project_dir/.cache/ffmpeg6" \
  --enable-shared --disable-static \
  --disable-programs --disable-doc --disable-debug --disable-autodetect
make -j "${COSMOS_BUILD_JOBS:-8}"
make install
