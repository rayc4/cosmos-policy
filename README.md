# Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning

<p align="center">
  <a href="https://arxiv.org/abs/2601.16163">Paper</a>&nbsp | <a href="https://research.nvidia.com/labs/dir/cosmos-policy/">Project Website</a>&nbsp | 🤗 <a href="https://huggingface.co/collections/nvidia/cosmos-policy">Models & Training Data</a>&nbsp | <a href="https://youtu.be/V2qdFD9n5BM">Summary Video</a>
</p>

## System Requirements

Inference with base Cosmos Policy only (i.e., no model-based planning):
* 1 GPU with 6.8 GB VRAM for LIBERO sim benchmark tasks
* 1 GPU with 8.9 GB VRAM for RoboCasa sim benchmark tasks
* 1 GPU with 6.0 GB VRAM for ALOHA robot tasks

Inference with Cosmos Policy + model-based planning (best-of-N search) on ALOHA robot tasks:
* Minimum (serial inference): 1 GPU with 10.0 GB VRAM
* Recommended (parallel inference): N GPUs with 10.0 GB VRAM each

Training:
* Generally, it is recommended to have at least 1 node of 8 80GB GPUs. For the experiments in the Cosmos Policy paper, we used 8 80GB GPUs (H100s) for 48 hours for small-scale ALOHA robot data fine-tuning (<200 demos), 32 80GB GPUs (H100s) for 48 hours for RoboCasa training (1200 demos), and 64 80GB GPUs (H100s) for 48 hours for LIBERO training (2000 demos). If you have fewer GPUs, you can use gradient accumulation to increase total batch size, which we found leads to faster convergence than taking more gradient steps with a smaller batch size.

## Quick Start: Predict 2.5 RoboCasa

This workspace has migrated RoboCasa inference to the official Predict 2.5 robot
policy in `cosmos-predict2.5/`. The root `cosmos_policy/` source is retained as
Predict 2 history; the new runtime uses a separate environment.

From the workspace root:

```bash
bash bin/setup_predict2p5.sh
source bin/inference_env.sh
"$PYTHON_BIN" bin/download_predict2p5.py --with-policy --with-reason1
bash slurm/gpu-job.sh --time=01:00:00 --mem=96G -- \
  bin/run_robocasa_predict2p5.py --task_name TurnOffMicrowave --num_trials_per_task 1
```

See [SETUP.md](SETUP.md) for CUDA 13/ARM installation and GPU checks,
[ROBOCASA.md](ROBOCASA.md) for evaluations and training details, and
[PREDICT2_MIGRATION.md](PREDICT2_MIGRATION.md) for removal of the old environment
and checkpoints. The original VRAM figures above describe Predict 2; measure
the new policy on your hardware before using them for resource allocation.

## Installation

See [SETUP.md](SETUP.md) for instructions on setting up the environment.

## Training and Evaluation

See [LIBERO.md](LIBERO.md) for the retained legacy Predict 2 LIBERO recipe.

See [ROBOCASA.md](ROBOCASA.md) for fine-tuning/evaluating on RoboCasa simulation benchmark tasks.

See [ALOHA.md](ALOHA.md) for the retained legacy Predict 2 ALOHA recipe.

## Support

If you run into any issues, please open a new GitHub issue. For critical blocking issues, please email Moo Jin Kim (moojink@cs.stanford.edu) to bring the issue to his attention.

## Citation

If you use our code in your work, please cite [our paper](https://arxiv.org/abs/2601.16163):

```bibtex
@article{kim2026cosmos,
  title={Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning},
  author={Kim, Moo Jin and Gao, Yihuai and Lin, Tsung-Yi and Lin, Yen-Chen and Ge, Yunhao and Lam, Grace and Liang, Percy and Song, Shuran and Liu, Ming-Yu and Finn, Chelsea and Gu, Jinwei},
  journal={arXiv preprint arXiv:2601.16163},
  year={2026}
}
```
