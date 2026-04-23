# DeepHAMX

**DeepHAMX** is a [JAX](https://github.com/google/jax)-accelerated reimplementation of [**DeepHAM**](https://github.com/frankhan91/DeepHAM): a global solution method for heterogeneous-agent models with aggregate shocks (Han, Yang, and E). This repository keeps the same high-level algorithm and configuration style as the original PyTorch code, while focusing on faster execution through JAX, `jax.numpy`, [Haiku](https://github.com/deepmind/dm-haiku), and [Optax](https://github.com/deepmind/optax).

> **Scope.** The training entry point in this fork targets the **Krusell–Smith (KS)** model. Additional JSON configs under `srcx/configs/` may be carried over from upstream experiments; only the KS training script is wired for JAX in the current layout.

---

## Original method and paper

<div align="center">

### DeepHAM: A global solution method for heterogeneous agent models with aggregate shocks

Jiequn Han, Yucheng Yang, Weinan E

[![arXiv](https://img.shields.io/badge/arXiv-2112.14377-b31b1b.svg)](https://arxiv.org/abs/2112.14377)
[![SSRN](https://img.shields.io/badge/SSRN-3990409-133a6f.svg)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3990409)
[![PDF](https://img.shields.io/badge/PDF-8A2BE2)](https://yangycpku.github.io/files/DeepHAM_paper.pdf)

Upstream repository: [github.com/frankhan91/DeepHAM](https://github.com/frankhan91/DeepHAM)

</div>

---

## Repository layout

| Path | Role |
|------|------|
| `srcx/` | JAX implementation: training, value networks, policy, datasets, utilities |
| `srcx/configs/KS/` | Example Krusell–Smith experiment configs (JSON) |
| `srcx/train_KS.py` | Main training script (absl flags) |
| `data/` | Place model inputs (for example `.mat` policy matrices) and run outputs under `data/simul_results/` |
| `environment.yml` | Conda environment specification |
| `scripts/smoke_check.py` | Quick JAX / import / shape sanity check (no full training run) |

The KS configs reference MATLAB-derived assets such as `data/KS_policy_N50_v1.mat`. If those files are not in your clone, obtain the corresponding data from the [original DeepHAM repository](https://github.com/frankhan91/DeepHAM) or your own preprocessing pipeline, and align `mats_path` in the JSON config with your local paths.

---

## Installation

**Tested setup:** Python **3.11**, JAX built for **CUDA 12** (GPU). CPU-only JAX installs are possible but are not the configuration used for the benchmarks below.

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate deephamx
```

### Option B — Manual Conda env

```bash
conda create -n deephamx python=3.11
conda activate deephamx
pip install "jax[cuda12]" dm-haiku optax absl-py tqdm scipy
```

Install the JAX variant that matches your platform (`jax[cuda12]` vs CPU wheels) following the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

> **Note.** CUDA 12 is strongly recommended if you use NVIDIA GPUs. Matching JAX wheels to an older CUDA stack can be brittle; the authors tested primarily on **CUDA 12**.

### Smoke test (optional)

After activating your JAX environment, from the **repository root** run:

```bash
python3 scripts/smoke_check.py
```

This script imports `srcx` modules, runs a short `simul_shocks` / `next_wealth` path, and performs one Haiku MLP forward pass. It mirrors `train_KS.py` by turning on **`jax_enable_x64`** when `srcx/param.py` has `DTYPE = "float64"`, and in that case it also **asserts `float64` dtypes** on the sampled tensors so silent float32 truncation is caught early.

---

## Quick start (Krusell–Smith)

Run from the `srcx` directory so relative config and data paths resolve as in the defaults.

```bash
cd srcx
python3 train_KS.py
```

### Command-line flags

`train_KS.py` uses [absl.flags](https://abseil.io/docs/python/guides/flags):

| Flag | Short | Default | Meaning |
|------|-------|---------|---------|
| `--config_path` | `-c` | `./configs/KS/game_nn_n50.json` | Path to the experiment JSON |
| `--exp_name` | `-n` | `test` | Suffix for the output run directory name |

Example:

```bash
cd srcx
python3 train_KS.py -c ./configs/KS/game_nn_n10.json -n my_run
```

Checkpoints and logs are written under `data/simul_results/KS/` according to policy type, sampling mode, number of agents, and `exp_name` (see `train_KS.py` for the exact naming pattern).

### Floating-point precision

Global dtype for the KS run is set in `srcx/param.py` (`DTYPE = "float64"` or `"float32"`). For float64, the script enables `jax_enable_x64` automatically.

---

## Performance

Benchmarks were run on **NVIDIA GeForce RTX 3090** nodes (**2× AMD EPYC 7H12**, 64 cores each). Speedups come from computational improvements (JAX/XLA, batching, etc.); **algorithm parameters and the overall solving procedure were not intentionally changed** relative to the reference implementation. You can cross-check numerical behavior using validation utilities from the original DeepHAM repository.

| Precision | Implementation | CUDA streams | Wall-clock | Valid **U** | End **K** |
|:---------:|:--------------:|:------------:|:----------:|:-----------:|:---------:|
| FP32 | JAX | 1 | 17 min | 104.035 | 39.237 |
| FP32 | JAX | 2 | 17 min | 104.096 | 38.329 |
| FP64 | JAX | 1 | 41 min | 103.728 | 38.380 |
| FP64 | JAX | 2 | 34 min | 103.707 | 39.361 |
| FP32 | Original (ORG) | 1 | 57 min | 103.702 | 39.014 |
| FP64 | Original (ORG) | 1 | 65 min | 104.126 | 39.326 |

Roughly **3.5×** faster than the original PyTorch run in the FP32, single-stream setting reported above.

---

## Citation

If you use DeepHAM in research, please cite the original paper:

```bibtex
@article{HanYangE2021deepham,
  title={Deep{HAM}: A global solution method for heterogeneous agent models with aggregate shocks},
  author={Han, Jiequn and Yang, Yucheng and E, Weinan},
  journal={arXiv preprint arXiv:2112.14377},
  year={2021}
}
```

If this JAX port is useful in your work, a star on the repository is appreciated.

---

## License

This project is distributed under the **GNU Lesser General Public License v2.1** — see [`LICENSE`](LICENSE). Respect upstream licensing and attribution when reusing or redistributing code and data.

---

## Contact

- **DeepHAM (method and theory):** [jiequnhan@gmail.com](mailto:jiequnhan@gmail.com), [yucheng.yang@uzh.ch](mailto:yucheng.yang@uzh.ch)  
- **DeepHAMX (JAX port):** [wang2021@stu.pku.edu.cn](mailto:wang2021@stu.pku.edu.cn)
