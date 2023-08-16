import sys
from threading import RLock as TRLock

import numpy as np
import pandas as pd
from pathos.pools import ProcessPool
from tqdm.auto import tqdm  # progress bar
from utils import advance_PRNG_state
from utils import block
from utils import check_and_set_nchains
from utils import generate_seed_sequence
from utils import State
from utils import tune_scale_lmh_table
from utils import tune_scale_rwm_table

sys.path.insert(0, "../nqs/")


class Sampler:
    def __init__(self, mcmc_alg: str, rbm, rng, logger=None):
        self._mcmc_alg = mcmc_alg
        self._rbm = rbm
        self._rng = rng
        self._scale = None
        self._logger = logger

        if mcmc_alg == "rwm":
            self._step = self._rwm_step
            self._tune_table = tune_scale_rwm_table
        elif mcmc_alg == "lmh":
            self._step = self._lmh_step
            self._tune_table = tune_scale_lmh_table
        else:
            msg = "Unsupported MCMC algorithm, only 'rwm' or 'lmh' is allowed"
            raise ValueError(msg)

    def _rwm_step(self, state, v_bias, h_bias, kernel, seed):
        """One step of the random walk Metropolis algorithm

        Parameters
        ----------
        state : nqs.State
            Current state of the system. See state.py

        scale : float
            Scale of proposal distribution. Default: 0.5

        Returns
        -------
        new_state : nqs.State
            The updated state of the system.
        """

        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)

        # Sample proposal positions, i.e., move walkers
        proposals = rng.normal(loc=state.positions, scale=self._scale)

        # Sample log uniform rvs
        log_unif = np.log(rng.random())

        # Compute proposal log density
        logp_proposal = self._rbm.logprob(proposals, v_bias, h_bias, kernel)

        # Metroplis acceptance criterion
        accept = log_unif < logp_proposal - state.logp

        # If accept is True, yield proposal, otherwise keep old state
        new_positions = proposals if accept else state.positions

        # Create new state
        new_logp = self._rbm.logprob(new_positions, v_bias, h_bias, kernel)
        new_n_accepted = state.n_accepted + accept
        new_delta = state.delta + 1
        new_state = State(new_positions, new_logp, new_n_accepted, new_delta)

        return new_state

    def _lmh_step(self, state, v_bias, h_bias, kernel, seed):
        """One step of the Langevin Metropolis-Hastings algorithm

        Parameters
        ----------
        state : State
            Current state of the system. See state.py
        alpha :
            Variational parameter
        D : float
            Diffusion constant. Default: 0.5
        dt : float
            Scale of proposal distribution. Default: 1.0
        """

        # Precompute
        dt = self._scale**2
        Ddt = 0.5 * dt
        quarterDdt = 1 / (4 * Ddt)
        sys_size = state.positions.shape

        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)

        # Compute drift force at current positions
        F = self._rbm.drift_force(state.positions, v_bias, h_bias, kernel)

        # Sample proposal positions, i.e., move walkers
        proposals = (
            state.positions
            + F * Ddt
            + rng.normal(loc=0, scale=self._scale, size=sys_size)
        )

        # Compute proposal log density
        logp_prop = self._rbm.logprob(proposals, v_bias, h_bias, kernel)

        # Green's function conditioned on proposals
        F_prop = self._rbm.drift_force(proposals, v_bias, h_bias, kernel)
        G_prop = -((state.positions - proposals - Ddt * F_prop) ** 2) * quarterDdt

        # Green's function conditioned on current positions
        G_cur = -((proposals - state.positions - Ddt * F) ** 2) * quarterDdt

        # Metroplis-Hastings ratio
        ratio = logp_prop + np.sum(G_prop) - state.logp - np.sum(G_cur)

        # Sample log uniform rvs
        log_unif = np.log(rng.random())

        # Metroplis acceptance criterion
        accept = log_unif < ratio

        # If accept is True, yield proposal, otherwise keep old state
        new_positions = proposals if accept else state.positions

        # Create new state
        new_logp = self._rbm.logprob(new_positions, v_bias, h_bias, kernel)
        new_n_accepted = state.n_accepted + accept
        new_delta = state.delta + 1
        new_state = State(new_positions, new_logp, new_n_accepted, new_delta)

        return new_state

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

    @property
    def scale(self):
        return self._scale

    def step(self, state, v_bias, h_bias, kernel, seed):
        return self._step(state, v_bias, h_bias, kernel, seed)

    @scale.setter
    def scale(self, value):
        self._scale = value
