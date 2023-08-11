#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


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


def tune_scale_rwm_table(scale, acc_rate):
    """Proposal scale lookup table. (Original)

    Aims to obtain an acceptance rate between 20-50%.

    Retrieved from the source code of PyMC [1].

    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:

                    Rate    Variance adaptation
                    ----    -------------------
                    <0.001        x 0.1
                    <0.05         x 0.5
                    <0.2          x 0.9
                    >0.5          x 1.1
                    >0.75         x 2
                    >0.95         x 10

    References
    ----------
    [1] https://github.com/pymc-devs/pymc/blob/main/pymc/step_methods/metropolis.py#L263

    Arguments
    ---------
    scale : float
        Scale of the proposal distribution
    acc_rate : float
        Acceptance rate of the last tuning interval

    Returns
    -------
    scale : float
        Updated scale parameter
    """
    if acc_rate < 0.001:
        # reduce by 90 percent
        return scale * 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        scale *= 0.5
    elif acc_rate < 0.2:
        # reduce by ten percent
        scale *= 0.9
    elif acc_rate > 0.5:
        # increase by ten percent
        scale *= 1.1
    elif acc_rate > 0.75:
        # increase by double
        scale *= 2.0
    elif acc_rate > 0.95:
        # increase by factor of ten
        scale *= 10.0

    return scale


def tune_scale_lmh_table(scale, acc_rate):
    if acc_rate < 0.001:
        # reduce by 90 percent
        return scale * 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        scale *= 0.5
    elif acc_rate < 0.4:
        # reduce by ten percent
        scale *= 0.9
    elif acc_rate > 0.8:
        # increase by ten percent
        scale *= 1.1
    elif acc_rate > 0.75:
        # increase by double
        scale *= 2.0
    elif acc_rate > 0.95:
        # increase by factor of ten
        scale *= 10.0

    return scale


def tune_scale_lmh_table_orig(dt, acc_rate):
    """Proposal dt (squared scale for importance sampler) lookup table.

    Aims to obtain an acceptance rate between 40-80%.

    Arguments
    ---------
    dt : float
        Scale of the proposal distribution
    acc_rate : float
        Acceptance rate of the last tuning interval

    Returns
    -------
    scale : float
        Updated scale parameter
    """
    if acc_rate < 0.001:
        # reduce by 90 percent
        dt *= 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        dt *= 0.5
    elif acc_rate < 0.2:
        # reduce by 25 percent
        dt *= 0.75
    elif acc_rate < 0.4:
        # reduce by ten percent
        dt *= 0.9
    elif acc_rate > 0.8:
        # increase by ten percent
        dt *= 1.21
    elif acc_rate > 0.9:
        # increase by factor of 5
        dt *= 25.0
    elif acc_rate > 0.95:
        # increase by factor of 100
        dt *= 100.0
    elif acc_rate > 0.98:
        # increase by factor of 1000
        dt *= 1000.0

    return dt
