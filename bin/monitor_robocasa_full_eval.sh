#!/usr/bin/env bash
# Monitor the latest benchmark batch, or the jobs.tsv passed as an argument.
set -euo pipefail
exec python3 "$(dirname -- "${BASH_SOURCE[0]}")/monitor_robocasa_full_eval.py" "$@"
