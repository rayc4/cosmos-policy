"""Run the upstream Predict 2.5 RoboCasa evaluator with this workspace's defaults."""

import os
import runpy
import sys
from pathlib import Path


def main():
    workspace = Path(__file__).resolve().parents[1]
    checkout = Path(os.environ.get("COSMOS_PREDICT2P5_DIR", workspace / "cosmos-predict2.5"))
    module = "cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.robocasa.run_robocasa_eval"
    metadata_dir = workspace / ".cache" / "predict2.5"
    defaults = {
        "config": "cosmos_predict2p5_2b_480p_robocasa_50_demos_per_task__inference_no_s3",
        "ckpt_path": "nvidia/Cosmos-Predict2.5-2B/robot/policy/robocasa",
        "config_file": "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py",
        "dataset_stats_path": str(metadata_dir / "robocasa_dataset_statistics.json"),
        "t5_text_embeddings_path": str(metadata_dir / "reason1_embeddings.pkl"),
        "text_embeddings_kind": "reason1",
        "shift": "5",
        "use_wrist_image": "True",
        "num_wrist_images": "1",
        "use_proprio": "True",
        "normalize_proprio": "True",
        "unnormalize_actions": "True",
        "trained_with_image_aug": "True",
        "chunk_size": "32",
        "num_open_loop_steps": "16",
        "task_name": "TurnOffMicrowave",
        "num_trials_per_task": "1",
        "run_id_note": "predict2p5-arm-slurm-smoke",
        "local_log_dir": str(workspace / "results" / "robocasa-predict2.5"),
        "seed": "195",
        "randomize_seed": "False",
        "deterministic": "True",
        "use_variance_scale": "False",
        "use_jpeg_compression": "True",
        "flip_images": "True",
        "num_denoising_steps_action": "5",
        "num_denoising_steps_future_state": "1",
        "num_denoising_steps_value": "1",
        "data_collection": "False",
    }
    supplied = {arg.split("=", 1)[0].removeprefix("--") for arg in sys.argv[1:] if arg.startswith("--")}
    arguments = [item for key, value in defaults.items() if key not in supplied for item in (f"--{key}", value)]
    sys.argv = [module, *arguments, *sys.argv[1:]]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
