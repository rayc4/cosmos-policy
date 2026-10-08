"""Check the inference environment without downloading model checkpoints."""

import importlib
import os
import platform
import shutil
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path


def main():
    sys.stdout.reconfigure(line_buffering=True)
    checkout = Path(__file__).resolve().parents[1] / "cosmos-predict2.5"
    # Upstream simulator and configuration assets use paths relative to its root.
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    print(f"Architecture: {platform.machine()}; Python: {platform.python_version()}")
    print(f"Interpreter: {sys.executable}; Predict 2.5 checkout: {checkout}")
    failures = []
    uvx_path = shutil.which("uvx")
    if uvx_path:
        print(f"Checkpoint CLI: {uvx_path}")
    else:
        failures.append("uvx not found: source bin/inference_env.sh before running checkpoint inference")
    for package in ("cosmos-predict2", "torch", "torchvision", "flash-attn", "transformer-engine", "natten", "decord"):
        try:
            print(f"{package}: {version(package)}")
        except Exception as exc:
            failures.append(f"{package}: {exc}")

    for module in ("torch", "torchvision", "decord", "draccus", "robocasa", "mujoco",
                   "cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils",
                   "cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.robocasa.run_robocasa_eval"):
        try:
            importlib.import_module(module)
            print(f"Import OK: {module}")
        except Exception as exc:
            failures.append(f"Import failed: {module}: {exc}")

    try:
        import av
        import decord
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".mp4") as video_file:
            with av.open(video_file.name, "w") as container:
                stream = container.add_stream("mpeg4", rate=5)
                stream.width = stream.height = 32
                stream.pix_fmt = "yuv420p"
                for intensity in (0, 80, 160):
                    frame = av.VideoFrame.from_ndarray(
                        np.full((32, 32, 3), intensity, dtype=np.uint8), format="rgb24"
                    )
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            reader = decord.VideoReader(video_file.name, ctx=decord.cpu(0))
            assert len(reader) == 3
            assert reader.get_batch([0, 1, 2]).shape == (3, 32, 32, 3)
            print("Decord video decode smoke test passed")
    except Exception as exc:
        failures.append(f"Video decode check failed: {exc}")

    try:
        import torch

        if not torch.cuda.is_available():
            failures.append(
                "CUDA unavailable: run inside a SLURM GPU allocation (--gres=gpu:1); "
                "if already allocated, check the job's GPU resources and device access"
            )
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)}; Torch CUDA: {torch.version.cuda}")
            assert torch.version.cuda.startswith("13."), "Predict 2.5 GB10 runtime must use CUDA 13"
            for module in ("flash_attn", "transformer_engine.pytorch", "natten"):
                importlib.import_module(module)
                print(f"GPU extension import OK: {module}")
            result = (torch.ones(8, device="cuda") * 2).sum().item()
            assert result == 16
            print("CUDA tensor smoke test passed")
            from flash_attn import flash_attn_func
            from torch.nn.functional import scaled_dot_product_attention

            q, k, v = (torch.randn(1, 16, 2, 64, dtype=torch.bfloat16, device="cuda") for _ in range(3))
            actual = flash_attn_func(q, k, v)
            expected = scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
            print("FlashAttention kernel smoke test passed")
    except Exception as exc:
        failures.append(f"CUDA check failed: {exc}")

    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
