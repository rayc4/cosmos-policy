"""Download RoboCasa metadata and cache the policy and Reason1 inference assets."""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


# Match the official public Reason1 mappings in the pinned policy checkout.
REASON1_REVISION = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-policy", action="store_true", help="Also cache policy weights and the Wan tokenizer")
    parser.add_argument("--with-reason1", action="store_true", help="Also cache the Reason1 encoder/tokenizer needed for uncached instructions")
    parser.add_argument("--reason1-only", action="store_true", help="Cache only Reason1, without downloading/copying metadata")
    args = parser.parse_args()
    if args.with_policy:
        # Fail early on missing access instead of downloading 25 GB first.
        HfApi().get_paths_info("nvidia/Cosmos-Predict2.5-2B", ["robot/policy/robocasa/model.pt"])
    output_dir = Path(__file__).resolve().parents[1] / ".cache" / "predict2.5"
    output_dir.mkdir(parents=True, exist_ok=True)
    # The official 2.5 recipe deliberately uses the same action/proprio scaling.
    metadata = (
        ("nvidia/Cosmos-Policy-RoboCasa-Predict2-2B", "robocasa_dataset_statistics.json", "model"),
        ("nvidia/RoboCasa-Cosmos-Policy", "success_only/reason1_embeddings.pkl", "dataset"),
    )
    for repo_id, filename, repo_type in (() if args.reason1_only else metadata):
        print(f"Downloading {repo_id}/{filename}", flush=True)
        source = Path(hf_hub_download(repo_id, filename, repo_type=repo_type))
        destination = output_dir / Path(filename).name
        if not destination.exists() or destination.stat().st_size != source.stat().st_size:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        print(f"Saved {destination}", flush=True)
    if args.with_policy:
        for repo_id, filename in (
            ("nvidia/Cosmos-Predict2.5-2B", "robot/policy/robocasa/model.pt"),
            ("nvidia/Cosmos-Predict2.5-2B", "tokenizer.pth"),
        ):
            path = hf_hub_download(repo_id, filename)
            print(f"Cached {path}", flush=True)
    if args.with_reason1 or args.reason1_only:
        print("Caching the public Reason1 encoder and tokenizer for cache misses", flush=True)
        path = snapshot_download("nvidia/Cosmos-Reason1-7B", revision=REASON1_REVISION)
        print(f"Cached Reason1: {path}", flush=True)


if __name__ == "__main__":
    main()
