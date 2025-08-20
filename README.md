# DeepHAMX: JAX-Accelerated Drop-in Replacement for DeepHAM (3.5× Faster, Algorithm-Preserving)
## Original Version
<div align="center">

### DeepHAM: A global solution method for heterogeneous agent models with aggregate shocks

Jiequn Han, Yucheng Yang, Weinan E

[![arXiv](https://img.shields.io/badge/arXiv-2112.14377-b31b1b.svg)](https://arxiv.org/abs/2112.14377)
[![SSRN](https://img.shields.io/badge/SSRN-3990409-133a6f.svg)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3990409)
[![PDF](https://img.shields.io/badge/PDF-8A2BE2)](https://yangycpku.github.io/files/DeepHAM_paper.pdf)

Link to original repository: https://github.com/frankhan91/DeepHAM

</div>

## Dependencies
```bash
# You can use our environment.yml file
conda env create -f environment.yml
# Or, you can do it manually
conda create -n deephamx python=3.11.0
conda activate deephamx
pip3 install jax[cuda12] dm-haiku optax pickle tqdm scipy
```

*We have only tested this code on CUDA 12. Based on experience, configuring JAX with CUDA 11 can be very time-consuming, so we recommend using CUDA 12 as your CUDA version.*
## Running
### Quick start for the Krusell-Smith (KS) model under default configs:
To use DeepHAM-X to solve the competitive equilibrium of the KS model, run
```bash
cd srcx # Don't forget!
python3 train_KS.py
```
*Current JAX version only support KS model.*
## Performance
### Tested on NVIDIA GeForce RTX 3090 Clusters with 2 AMD EPYC 7H12 64-Core Processor

<div style="text-align:center; margin: 0 auto; display: table">

| FP  | Arch  | CUDA Streams | Runtime | Valid U   | End K   |
|:---:|:-----:|:------------:|:-------:|:---------:|:-------:|
| 32  | JAX   | 1            | 17m     | 104.035   | 39.237  |
| 32  | JAX   | 2            | 17m     | 104.096   | 38.329  |
| 64  | JAX   | 1            | 41m     | 103.728   | 38.380  |
| 64  | JAX   | 2            | 34m     | 103.707   | 39.361  |
| 32  | ORG | 1            | 57m     | 103.702   | 39.014  |
| 64  | ORG | 1            | 65m     | 104.126   | 39.326  |

</div>
We accelerated the original implementation of DeepHams by 3.5× using JAX, with optimizations purely based on computational improvements, without altering any algorithm parameters or the solving process. You can verify the JAX version at different computational precisions using the validation code from the original repository.

## Citation
If you find this work helpful, please consider starring this repo and citing our paper using the following Bibtex.
```bibtex
@article{HanYangE2021deepham,
  title={Deep{HAM}: A global solution method for heterogeneous agent models with aggregate shocks},
  author={Han, Jiequn and Yang, Yucheng and E, Weinan},
  journal={arXiv preprint arXiv:2112.14377},
  year={2021}
}
```

## Contact
Please contact us at jiequnhan@gmail.com and yucheng.yang@uzh.ch if you have any questions of DeepHAM and at wang2021@stu.pku.edu.cn if you have any questions of DeepHAM-X.
