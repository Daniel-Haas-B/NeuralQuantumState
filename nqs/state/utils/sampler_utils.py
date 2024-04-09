#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from joblib import delayed
from joblib import Parallel
from tqdm.auto import tqdm

from .pool_tools import generate_seed_sequence
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


def multiproc(proc_sample, wf, nchains, nsamples, state, seeds, save_positions=False):
    params = wf.params

    # Handle iterable
    wf = [wf] * nchains
    nsamples = [nsamples] * nchains
    state = [state] * nchains
    params = [params] * nchains
    chain_ids = list(range(nchains))
    save_positions = [save_positions] * nchains

    # Define a helper function to package the delayed computation
    def compute(i):
        return proc_sample(
            wf[i], nsamples[i], state[i], seeds[i], chain_ids[i], save_positions[i]
        )

    results = Parallel(n_jobs=-1)(delayed(compute)(i) for i in range(nchains))

    # Assuming that proc_sample returns a tuple (result, energy), you can unpack them
    results, energies = zip(*results)

    return results, energies


def jax_multiproc(proc_sample, wf, nchains, nsamples, state, scale, seeds, logger=None):
    """#TODO: Enable multiprocessing with jax pmap to allow GPU multiprocessing."""
    raise NotImplementedError("jax_multiproc not implemented yet")


def tune_sampler(
    wf,
    sampler,
    current_state,
    tune_iter=500,
    tune_batch=500,
    seed=None,
    mode="standard",
    logger=None,
    logger_level="INFO",
):
    """
    Tune proposal scale so that the acceptance rate is around 0.7.
    """
    # seed = seed + 1 if seed is not None else None
    seed_seq = generate_seed_sequence(seed, 1)[0]

    # Reset n_accepted
    state = current_state
    state = State(
        state.positions, state.logp, 0, state.delta
    )  # get positions but Reset n_accepted
    states = state.create_batch_of_states(batch_size=tune_batch)

    if logger_level != "SILENT":
        t_range = tqdm(
            range(tune_iter),
            desc="[Tuning progress]",
            position=0,
            leave=False,
            colour="cyan",
        )
    else:
        t_range = range(tune_iter)

    if mode == "standard":
        for _ in t_range:
            state = State(
                state.positions, state.logp, 0, state.delta
            )  # get positions but Reset n_accepted
            states = state.create_batch_of_states(batch_size=tune_batch)

            states = sampler.step(wf, states, seed_seq, batch_size=tune_batch)

            # Tune proposal scale
            old_scale = sampler.scale
            accept_rate = states[-1].n_accepted / tune_batch
            state = states[-1]
            t_range.set_postfix(
                acc_rate=f"{accept_rate:.2f}",
                scale=f"{sampler.scale:.2f}",
                refresh=True,
            )

            if 0.5 < accept_rate < 0.7:
                return

            if accept_rate < 0.001 or accept_rate > 0.999:
                logger.warning(
                    "Acceptance rate is too low or too high. Rescaling positions..."
                )  # maybe reset the positions also
                wf.reinit_positions()
                state = wf.state

                states = state.create_batch_of_states(batch_size=tune_batch)
                states = sampler.step(wf, states, seed_seq, batch_size=tune_batch)

                wf.rescale_parameters(
                    0.1
                )  # this hopefully makes the predictions be more around 0 because the weights are smaller
                # reset scale
                sampler.reset_scale(1 / np.sqrt(wf.N * wf.dim))
                return  # start over

            sampler.tune_scale(old_scale, accept_rate)

    elif mode == "infinite":
        while True:
            states = state.create_batch_of_states(batch_size=tune_batch)
            print("state positions pre step", states[-1].positions)
            states = sampler.step(wf, states, seed_seq, batch_size=tune_batch)
            print("state positions post step", states[-1].positions)
            print("acc rate", states[-1].n_accepted / tune_batch)
            print("prob", np.exp(states[-1].logp))
            # Tune proposal scale
            old_scale = sampler.scale
            accept_rate = states[-1].n_accepted / tune_batch

            if accept_rate < 0.001 or accept_rate > 0.99:
                logger.warning(
                    "Acceptance rate is too low or too high. Rescaling positions..."
                )  # maybe reset the positions also
                wf.reinit_positions()
                state = wf.state
                # if accept_rate < 0.001:
                #    wf.rescale_parameters(1.1) # this will make the logprob around 0

                # else: # accept rate > 0.99
                #    wf.rescale_parameters(0.9)
                # reset scale to 1
                # sampler.reset_scale(1)
                continue  # start over

            if 0.4 <= accept_rate <= 0.8:
                print(
                    f"Acceptance rate: {accept_rate:.10f}, scale {sampler.scale:.4f},  going to next epoch..."
                )
                return
            else:
                sampler.tune_scale(old_scale, accept_rate)
                print(
                    f"Acceptance rate: {accept_rate:.10f}, scale {sampler.scale:.4f},  continue tuning sampler scale..."
                )
