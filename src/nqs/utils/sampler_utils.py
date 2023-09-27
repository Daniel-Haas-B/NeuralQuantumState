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
