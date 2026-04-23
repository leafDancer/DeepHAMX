#!/usr/bin/env python3
"""
Lightweight smoke test for DeepHAMX (imports, JIT paths, Haiku MLP).

Run from the repository root:

    python3 scripts/smoke_check.py

Matches `srcx/train_KS.py` by enabling `jax_enable_x64` when `srcx/param.py`
sets `DTYPE == "float64"`. Exits with status 0 on success, 1 on failure.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = _repo_root()
    srcx = root / "srcx"
    sys.path.insert(0, str(srcx))

    import jax
    import jax.numpy as jnp

    import param

    if param.DTYPE == "float64":
        jax.config.update("jax_enable_x64", True)

    import haiku as hk

    import simulation_KS as ks
    import util

    # Import remaining modules to catch wiring / syntax issues early.
    import dataset  # noqa: F401
    import policy  # noqa: F401
    import value  # noqa: F401

    print("JAX", jax.__version__, "| devices:", jax.devices())
    print("param.DTYPE:", param.DTYPE, "| JNP_DTYPE:", param.JNP_DTYPE)

    mp = param.KSParam(n_agt=4, beta=0.99, mats_path=str(root / "data" / "KS_policy_N50_v1.mat"))
    key = jax.random.PRNGKey(0)
    k2, _ = jax.random.split(key)
    ashock, ishock = ks.simul_shocks(k2, 8, 16, mp, None)
    assert ashock.shape == (8, 16) and ishock.shape == (8, 4, 16), ashock.shape

    w = ks.next_wealth(
        jnp.ones((8, 4), dtype=param.JNP_DTYPE),
        jnp.ones((8, 1), dtype=param.JNP_DTYPE),
        jnp.ones((8, 4), dtype=param.JNP_DTYPE),
        mp,
    )
    assert w.shape == (8, 4), w.shape

    cfg = {"net_width": [8, 8], "activation": "tanh"}

    def fwd(x):
        return util.FeedforwardModel(4, 1, cfg, name="smoke_mlp")(x)

    tr = hk.without_apply_rng(hk.transform_with_state(fwd))
    x = jnp.ones((3, 4), dtype=param.JNP_DTYPE)
    params, st = tr.init(jax.random.PRNGKey(1), x)
    y, _ = tr.apply(params, st, x)
    assert y.shape == (3, 1), y.shape

    if param.DTYPE == "float64":
        assert ashock.dtype == jnp.float64, ashock.dtype
        assert y.dtype == jnp.float64, y.dtype

    print("smoke_check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
