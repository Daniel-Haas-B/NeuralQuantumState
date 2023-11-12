import jax
import numpy as np
import pandas as pd
from nqs.utils import block
from nqs.utils import check_and_set_nchains
from nqs.utils import generate_seed_sequence
from nqs.utils import sampler_utils
from nqs.utils import State
from tqdm.auto import tqdm  # progress bar

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class Sampler:
    def __init__(self, rng, scale, logger=None):
        self._rng = rng
        self._scale = scale  # to be set by child class
        self._logger = logger

    def sample(self, wf, state, nsamples, nchains=1, seed=None):
        """ """
        scale = self._scale
        nchains = check_and_set_nchains(nchains, self._logger)
        seeds = generate_seed_sequence(seed, nchains)
        if nchains == 1:
            chain_id = 0
            results, self._energies = self._sample(
                nsamples, state, scale, seeds[0], chain_id
            )
            self._results = pd.DataFrame([results])
        else:
            multi_sampler = sampler_utils.multiproc
            results, self._energies = multi_sampler(
                self._sample,
                wf,
                nchains,
                nsamples,
                state,
                scale,
                seeds,
                self._logger,
            )
            self._results = pd.DataFrame(results)

        self._sampling_performed_ = True
        if self._logger is not None:
            self._logger.info("Sampling done")

        return self._results

    def _sample(self, wf, nsamples, state, scale, seed, chain_id):
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
            state = self._step(wf, state, seed)
            energies[i] = self.hamiltonian.local_energy(wf, state.positions)

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

    def marginal_sample(self, wf, state, nsamples, nchains=1, seed=None):
        """
        Fix-position sampling to be used in the calculation of the
        one-body density. Of the wave function.
        Mathematicaly what we are doind is
        ρ(r)=∫∣Ψ(r,R)∣^2dR
         - r represents the position of one particle (in two dimensions, this is (x, y))
        - R represents the position of all particles (in two dimensions, this is (x1, y1, x2, y2, ...))
        - ∣Ψ(r,R)∣^2 is the probability density of finding the particles at position r. In our case this is pdf
        """
        scale = self._scale
        # for now this will not be parallelized sampling
        nchains = check_and_set_nchains(nchains, self._logger)
        seeds = generate_seed_sequence(seed, nchains)
        samples, weights = 0, 0
        if nchains == 1:
            chain_id = 0
            samples, weights = self._marginal_sample(
                nsamples, state, scale, seeds[0], chain_id
            )

        else:
            raise NotImplementedError(
                "Parallel marginal sampling is not implemented yet"
            )

        if self._logger is not None:
            self._logger.info("Marginal sampling done")

        return samples, weights

    def step(self, wf, state, seed):
        """
        To be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def set_hamiltonian(self, hamiltonian):
        self.hamiltonian = hamiltonian

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
