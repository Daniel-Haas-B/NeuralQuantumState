import copy

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

    def sample_obd(
        self,
        wf,
        state,
        nsamples,
        nchains=1,
        seed=None,
        lim_inf=-5,
        lim_sup=5,
        points=80,
    ):
        if nchains != 1:
            raise NotImplementedError("OBD is not implemented for parallel sampling")
        scale = self._scale
        nchains = check_and_set_nchains(nchains, self._logger)
        seeds = generate_seed_sequence(seed, nchains)
        chain_id = 0

        # create 100 states that are clones of the original state but with different coordinates for the first particle
        # then sample the wave function for each of these states
        positions = np.linspace(lim_inf, lim_sup, points)
        one_body_densities = np.zeros(points)
        for i in range(points):
            # create copy of the state
            new_state = copy.deepcopy(state)
            new_state.positions[0] = positions[i]

            one_body_density = self._marginal_sample(
                wf, 2**10, new_state, scale, seeds[0], chain_id
            )

            one_body_densities[i] = one_body_density

        # now we plot
        if self._logger is not None:
            self._logger.info("Marginal sampling done")

        return positions, one_body_densities

    def sample(self, wf, state, nsamples, nchains=1, seed=None):
        """ """
        scale = self._scale
        nchains = check_and_set_nchains(nchains, self._logger)
        seeds = generate_seed_sequence(seed, nchains)
        if nchains == 1:
            chain_id = 0

            results, self._energies = self._sample(
                wf, nsamples, state, scale, seeds[0], chain_id
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

    def _marginal_sample(self, wf, nsamples, state, scale, seed=None, chain_id=0):
        """
        Fix-position sampling to be used in the calculation of the
        one-body density. Of the wave function.
        Mathematicaly what we are doind is
        ρ(r)=∫∣Ψ(r,R)∣^2dR
         - r represents the position of one particle (in two dimensions, this is (x, y))
        - R represents the position of all particles (in two dimensions, this is (x1, y1, x2, y2, ...))
        - ∣Ψ(r,R)∣^2 is the probability density of finding the particles at position r. In our case this is pdf

        ρ(r) is of shape (ndimensions)
        """
        # TODO: CHECK THIS DIMENSIONALITY
        one_body_density = np.zeros_like(
            state.positions[0]
        )  # initialise the one body density

        for i in range(nsamples):
            # fix the position of one particle

            state = self._fixed_step(
                wf, state, seed, fixed_index=0
            )  # sample the other particles
            one_body_density += wf.pdf(
                state.positions
            )  # add the probability density of finding the particles at position r

        one_body_density /= nsamples  # average over the number of samples. This is a normalisation factor

        return one_body_density

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
