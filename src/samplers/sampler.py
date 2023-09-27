# import sys
from threading import RLock as TRLock

import numpy as np
import pandas as pd
from nqs.utils import block
from nqs.utils import check_and_set_nchains
from nqs.utils import generate_seed_sequence
from nqs.utils import State
from pathos.pools import ProcessPool
from tqdm.auto import tqdm  # progress bar


class Sampler:
    def __init__(self, rbm, rng, scale, logger=None):
        self._rbm = rbm
        self._rng = rng
        self._scale = scale  # to be set by child class
        self._logger = logger

    def sample(self, state, params, nsamples, nchains=1, seed=None):
        """ """

        # TODO: accept biases and kernel as parameters and
        # assume they are optimized if passed
        v_bias = params["v_bias"]
        h_bias = params["h_bias"]
        kernel = params["kernel"]
        scale = self._scale

        nchains = check_and_set_nchains(nchains, self._logger)
        seeds = generate_seed_sequence(seed, nchains)

        if nchains == 1:
            chain_id = 0
            results, self._energies = self._sample(
                nsamples, state, v_bias, h_bias, kernel, scale, seeds[0], chain_id
            )
            self._results = pd.DataFrame([results])
        else:
            if self._logger is not None:
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
            v_bias = (v_bias,) * nchains
            h_bias = (h_bias,) * nchains
            kernel = (kernel,) * nchains
            scale = (scale,) * nchains
            chain_ids = range(nchains)

            with ProcessPool(
                nchains, initializer=initializer, initargs=initargs
            ) as pool:
                results, self._energies = zip(
                    *pool.map(
                        self._sample,
                        nsamples,
                        state,
                        v_bias,
                        h_bias,
                        kernel,
                        scale,
                        seeds,
                        chain_ids,
                    )
                )
            self._results = pd.DataFrame(results)

        self._sampling_performed_ = True
        if self._logger is not None:
            self._logger.info("Sampling done")

        return self._results

    def _sample(self, nsamples, state, v_bias, h_bias, kernel, scale, seed, chain_id):
        """To be called by process"""
        if self._logger is not None:
            t_range = tqdm(
                range(nsamples),
                desc=f"[Sampling progress] Chain {chain_id+1}",
                position=chain_id,
                leave=True,
                colour="green",
            )
        else:
            t_range = range(nsamples)

        # Config
        state = State(state.positions, state.logp, 0, state.delta)
        energies = np.zeros(nsamples)

        for i in t_range:
            state = self._step(state, v_bias, h_bias, kernel, seed)
            energies[i] = self._rbm.local_energy(
                state.positions, v_bias, h_bias, kernel
            )
        if self._logger is not None:
            t_range.clear()

        energy = np.mean(energies)
        error = block(energies)
        variance = np.mean(energies**2) - energy**2
        acc_rate = state.n_accepted / nsamples

        sample_results = {
            "chain_id": chain_id + 1,
            "energy": energy,
            "std_error": error,
            "variance": variance,
            "accept_rate": acc_rate,
            "scale": scale,
            "nsamples": nsamples,
        }

        return sample_results, energies

    def step(self, state, v_bias, h_bias, kernel, seed):
        """
        To be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
