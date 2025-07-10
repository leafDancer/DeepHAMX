### A Quick Start
I have only tested this code on CUDA 12. Based on experience, configuring JAX with CUDA 11 can be very time-consuming, so I recommend using CUDA 12 as your GPU driver directly. Just follow the steps below to set up the environment properly.

```bash
conda create -n jax_env python=3.11.0
conda activate jax_env
pip install jax[cuda12] dm-haiku optax pickle tqdm scipy # If I missed anything, install the latest version! 📥
cd srcx
python train_KS.py
```