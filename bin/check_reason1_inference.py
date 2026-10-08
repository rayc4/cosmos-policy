"""Validate real Reason1 inference and persistent cache reuse in one GPU job."""

import os
from pathlib import Path
import sys


def main():
    sys.stdout.reconfigure(line_buffering=True)
    project = Path(__file__).resolve().parents[1]
    checkout = project / "cosmos-predict2.5"
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    import torch
    from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot import cosmos_utils
    from cosmos_predict2._src.predict2.cosmos_policy.datasets.reason1_embedding_utils import get_reason1_text_embedding

    if not torch.cuda.is_available():
        raise RuntimeError("Run this check inside a SLURM GPU allocation")
    instruction = "pick the candle from the cabinet and place it on the counter"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Reason1 snapshot: {os.environ.get('COSMOS_REASON1_DIR')}")
    # Always exercise actual model loading and generation, even after a prior
    # successful check has saved this instruction's extra embedding.
    embedding = get_reason1_text_embedding(instruction)
    assert tuple(embedding.shape) == (1, 512, 100352), embedding.shape
    assert torch.isfinite(embedding).all().item(), "Non-finite Reason1 embedding"
    print(f"Reason1 forward passed: shape={tuple(embedding.shape)}, dtype={embedding.dtype}")

    cosmos_utils.text_embeddings_kind = "reason1"
    cosmos_utils.t5_text_embeddings_path_global = str(project / ".cache/predict2.5/reason1_embeddings.pkl")
    first = cosmos_utils.get_t5_embedding_from_cache(instruction)
    assert first.dtype == torch.bfloat16
    torch.testing.assert_close(first, embedding.to(torch.bfloat16))
    cosmos_utils.t5_text_embeddings_cache.clear()

    def unexpected_recompute(_):
        raise AssertionError("Saved instruction was recomputed instead of reused")

    cosmos_utils.get_reason1_text_embedding = unexpected_recompute
    second = cosmos_utils.get_t5_embedding_from_cache(instruction)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    print("Persistent Reason1 cache reuse passed; shared base cache was not rewritten.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
