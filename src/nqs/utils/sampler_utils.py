#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from threading import RLock as TRLock

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from pathos.pools import ProcessPool
from tqdm.auto import tqdm

from .state import State


def early_stopping(new_value, old_value, tolerance=1e-5):
    """Criterion for early stopping.

    If the Euclidean distance between the new and old value of a quantity is
    below a specified tolerance, early stopping will be recommended.

    Arguments
    ---------
    new_value : float
        The updated value
    old_value : float
        The previous value
    tolerance : float
        Tolerance level. Default: 1e-05

    Returns
    -------
    bool
        Flag that indicates whether to early stop or not
    """

    dist = np.linalg.norm(new_value - old_value)
    return dist < tolerance


def numpy_multiproc(
    proc_sample, nchains, nsamples, state, params, scale, seeds, logger=None
):
    """Enable multiprocessing for numpy."""
    if logger is not None:
        # for managing output contention
        tqdm.set_lock(TRLock())
        initializer = tqdm.set_lock
        initargs = (tqdm.get_lock(),)
    else:
        initializer = None
        initargs = None

    # Handle iterables
    nsamples = (nsamples,) * nchains
    state = (state,) * nchains
    params = (params,) * nchains
    scale = (scale,) * nchains
    chain_ids = range(nchains)

    with ProcessPool(nchains, initializer=initializer, initargs=initargs) as pool:
        results, energies = zip(
            *pool.map(
                proc_sample,
                nsamples,
                state,
                params,
                scale,
                seeds,
                chain_ids,
            )
        )
        return results, energies


def jax_multiproc(
    proc_sample, nchains, nsamples, state, params, scale, seeds, logger=None
):
    """Enable multiprocessing for jax."""

    nsamples = jnp.array([nsamples] * nchains)

    # Replicate and stack the positions
    positions_stacked = jnp.stack([state.positions] * nchains, axis=0)
    # Replicate scalar fields
    logp_replicated = jnp.array([state.logp] * nchains)
    n_accepted_replicated = jnp.array([state.n_accepted] * nchains)
    delta_replicated = jnp.array([state.delta] * nchains)

    # Reconstruct the State named tuple for each chain
    state = [
        State(
            positions_stacked[i],
            logp_replicated[i],
            n_accepted_replicated[i],
            delta_replicated[i],
        )
        for i in range(nchains)
    ]

    params = params.to_jax()
    param_keys = params.keys()
    params = {key: jnp.stack([params.get(key)] * nchains, axis=0) for key in param_keys}

    scale = jnp.array([scale] * nchains)
    key = random.PRNGKey(42)
    seeds = random.split(key, num=nchains)
    chain_ids = jnp.arange(nchains)
    # Parallelize proc_sample using jax.pmap
    pmap_sample = jax.pmap(proc_sample)

    print("nsamples shape", nsamples.shape)
    # print("state shape", state.shape)
    print("params shape", params.shape)
    print("scale shape", scale.shape)
    print("seeds shape", seeds.shape)
    print("chain_ids shape", chain_ids.shape)

    results, energies = pmap_sample(nsamples, state, params, scale, seeds, chain_ids)

    # Convert the results to numpy arrays if necessary for further processing
    results = np.array(results)
    energies = np.array(energies)

    return results, energies
