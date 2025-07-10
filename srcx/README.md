### A Quick Start 🚀
I have only tested this code on CUDA 12. Based on experience, configuring JAX with CUDA 11 can be very time-consuming, so I recommend using CUDA 12 as your CUDA version. Just follow the steps below to set up the environment properly.

```bash
conda create -n jax_env python=3.11.0
conda activate jax_env
pip install jax[cuda12] dm-haiku optax pickle tqdm scipy # If I missed anything, install the latest version! 📥
cd srcx
python train_KS.py # In order to train a KS model.
```
### Performance Tests
**Decice Info**

- 8 Nvidia GeForce 3090 24GB
- 2 AMD EPYC 7H12 64-Core Processor
- CUDA 12.2

<div style="text-align:center; margin: 0 auto; display: table">

| FP  | Arch  | CUDA Streams | Runtime | Valid U   | End K   |
|:---:|:-----:|:------------:|:-------:|:---------:|:-------:|
| 32  | JAX   | 1            | 17m     | 104.035   | 39.237  |
| 32  | JAX   | 2            | 17m     | 104.096   | 38.329  |
| 64  | JAX   | 1            | 41m     | 103.728   | 38.380  |
| 64  | JAX   | 2            | 34m     | 103.707   | 39.361  |
| 32  | NUMPY | 1            | 57m     | 103.702   | 39.014  |
| 64  | NUMPY | 1            | 65m     | 104.126   | 39.326  |

</div>

### Attention
*Current JAX version only support KS model.*