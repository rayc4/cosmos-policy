"""Check local Predict 2.5 evaluation assets without downloading or using a GPU."""

import json
from pathlib import Path
import subprocess
import sys

from huggingface_hub import hf_hub_download, snapshot_download

from download_predict2p5 import REASON1_REVISION


def main():
    project = Path(__file__).resolve().parents[1]
    failures = []
    for filename in ("robocasa_dataset_statistics.json", "reason1_embeddings.pkl"):
        path = project / ".cache/predict2.5" / filename
        if not path.is_file() or not path.stat().st_size:
            failures.append(f"Missing metadata: {path}")
    for filename in ("robot/policy/robocasa/model.pt", "tokenizer.pth"):
        try:
            hf_hub_download("nvidia/Cosmos-Predict2.5-2B", filename, local_files_only=True)
        except Exception as exc:
            failures.append(f"Missing policy asset {filename}: {exc}")
    try:
        snapshot = Path(snapshot_download(
            "nvidia/Cosmos-Reason1-7B", revision=REASON1_REVISION, local_files_only=True,
        ))
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        required = set(index["weight_map"].values()) | {
            "config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
        }
        missing = sorted(name for name in required if not (snapshot / name).is_file())
        if missing:
            failures.append("Reason1 snapshot is incomplete: " + ", ".join(missing))
    except Exception as exc:
        failures.append(f"Missing Reason1 encoder/tokenizer: {exc}")
    for filename in ("predict2p5-tokenizer.patch", "predict2p5-reason1.patch"):
        result = subprocess.run([
            "git", "-C", str(project / "cosmos-predict2.5"), "apply", "--reverse", "--check",
            str(project / "patches" / filename),
        ], capture_output=True, text=True)
        if result.returncode:
            failures.append(f"Required runtime patch is not applied: {filename}; run bash bin/setup_predict2p5.sh")
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if failures:
        print('Cache inference assets with: source bin/inference_env.sh; "$PYTHON_BIN" bin/download_predict2p5.py --with-policy --with-reason1', file=sys.stderr)
        return 1
    print("Local policy, metadata, Reason1 encoder/tokenizer, and runtime patches are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
