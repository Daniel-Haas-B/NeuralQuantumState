import copy
import os

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from nqs.state.utils import block
from nqs.state.utils import check_and_set_nchains
from nqs.state.utils import generate_seed_sequence
from nqs.state.utils import sampler_utils


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
        method="histogram",
        points=80,
    ):
        if nchains != 1:
            raise NotImplementedError("OBD is not implemented for parallel sampling")
        scale = self.scale
        nchains = check_and_set_nchains(nchains, self._logger)
        seeds = generate_seed_sequence(seed, nchains)
        chain_id = 0

        # create 100 states that are clones of the original state but with different coordinates for the first particle
        # then sample the wave function for each of these states
        if method == "integrate":
            positions = np.linspace(lim_inf, lim_sup, points)
            one_body_densities = np.zeros(points)
            for i in range(points):
                # create copy of the state
                new_state = copy.deepcopy(state)
                new_state.positions[0] = positions[i]

                one_body_density = self._marginal_sample(
                    wf, nsamples, new_state, scale, seeds[0], chain_id
                )

                one_body_densities[i] = one_body_density

            # now we plot
            if self._logger is not None:
                self._logger.info("Marginal sampling done")

            return positions, one_body_densities

    def sample(self, wf, state, nsamples, nchains=1, seed=None, save_positions=False):
        """ """
        nchains = check_and_set_nchains(nchains, self._logger)
        seeds = generate_seed_sequence(seed, nchains)
        if nchains == 1:
            chain_id = 0

            results, self._energies = self._sample(
                wf, nsamples, state, seeds[0], chain_id, save_positions
            )
            self._results = pd.DataFrame([results])
        else:
            results, self._energies = sampler_utils.multiproc(
                self._sample,
                wf,
                nchains,
                nsamples,
                state,
                seeds,
                save_positions,
            )
            self._results = pd.DataFrame(results)

        self._sampling_performed_ = True
        if self._logger is not None:
            self._logger.info("Sampling done")
            self._logger.info(f"Save positions: {save_positions}")

        return self._results

    def _sample(self, wf, nsamples, state, seed, chain_id, save_positions=False):
        """To be called by process in the big sampler function."""
        batch_size = 2**10
        filename = f"data/energies_and_pos_{wf.__class__.__name__}_ch{chain_id}.h5"
        if self._logger is not None:
            t_range = tqdm(
                range(0, nsamples // batch_size),
                desc=f"[Sampling progress] Chain {chain_id + 1}",
                position=chain_id,
                leave=True,
                colour="green",
            )
        else:
            t_range = range(0, nsamples // batch_size)

        batch_state = state.create_batch_of_states(batch_size=batch_size)

        if os.path.exists(filename):
            os.remove(filename)
        for i in t_range:  # 2**18
            batch_state = self._step(wf, batch_state, seed, batch_size=batch_size)
            energies = self.hamiltonian.local_energy(wf, batch_state.positions)

            f1 = h5py.File(filename, "a")
            if i == 0:
                f1.create_dataset(
                    "energies",
                    data=energies,
                    compression="gzip",
                    chunks=True,
                    maxshape=(None,),
                )
                if save_positions:
                    f1.create_dataset(
                        "positions",
                        data=batch_state.positions,
                        compression="gzip",
                        chunks=True,
                        maxshape=(None, batch_state.positions.shape[1]),
                    )
            else:
                f1["energies"].resize(
                    (f1["energies"].shape[0] + energies.shape[0]),
                    axis=0,
                )
                f1["energies"][-energies.shape[0] :] = energies
                if save_positions:
                    f1["positions"].resize(
                        (f1["positions"].shape[0] + batch_state.positions.shape[0]),
                        axis=0,
                    )
                    f1["positions"][
                        -batch_state.positions.shape[0] :
                    ] = batch_state.positions

        if self._logger is not None:
            t_range.clear()

        # open energies file
        f = h5py.File(filename, "r")
        energies = f["energies"][:]
        assert (
            np.sum(energies == 0) == 0
        ), "There are empty energies which would give wrong statistics"

        energy = np.mean(energies)
        error = block(energies)
        variance = np.mean(energies**2) - energy**2
        acc_rate = batch_state.n_accepted[-1] / nsamples

        sample_results = {
            "chain_id": chain_id + 1,
            "energy": energy,
            "std_error": error,
            "variance": variance,
            "accept_rate": acc_rate,
            "scale": self.scale,
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
