#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from joblib import delayed
from joblib import Parallel


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


def multiproc(proc_sample, wf, nchains, nsamples, state, scale, seeds, logger=None):
    """Enable multiprocessing for jax."""
    params = wf.params

    # Handle iterable
    wf = [wf] * nchains
    nsamples = [nsamples] * nchains
    state = [state] * nchains
    params = [params] * nchains
    scale = [scale] * nchains
    chain_ids = list(range(nchains))

    # Define a helper function to package the delayed computation
    def compute(i):
        return proc_sample(
            wf[i], nsamples[i], state[i], scale[i], seeds[i], chain_ids[i]
        )

    results = Parallel(n_jobs=-1)(delayed(compute)(i) for i in range(nchains))

    # Assuming that proc_sample returns a tuple (result, energy), you can unpack them
    results, energies = zip(*results)

    return results, energies


def jax_multiproc(proc_sample, wf, nchains, nsamples, state, scale, seeds, logger=None):
    """#TODO: Enable multiprocessing with jax pmap to allow GPU multiprocessing."""
    raise NotImplementedError("jax_multiproc not implemented yet")
