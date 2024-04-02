import copy
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.state.utils import block
from src.state.utils import check_and_set_nchains
from src.state.utils import generate_seed_sequence
from src.state.utils import sampler_utils


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
        scale = self.scale
        nchains = check_and_set_nchains(nchains, self._logger)
        seeds = generate_seed_sequence(seed, nchains)
        if nchains == 1:
            chain_id = 0

            results, self._energies = self._sample(
                wf, nsamples, state, seeds[0], chain_id, save_positions
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
        batch_size = 2**17

        if self._logger is not None:
            t_range = tqdm(
                range(0, nsamples // batch_size),
                desc=f"[Sampling progress] Chain {chain_id+1}",
                position=chain_id,
                leave=True,
                colour="green",
            )
        else:
            t_range = range(0, nsamples // batch_size)

        batch_state = state.create_batch_of_states(batch_size=batch_size)
        batch_state.positions = np.random.randn(batch_size, 4)
        print("batch_state: ", batch_state)
        energies = np.zeros(nsamples)

        positions = np.zeros((nsamples, batch_state.positions.shape[1]))
        # print("params pre: ", wf.params)
        wf.params.set("alpha", np.array([0.5, 0.5, 0.5, 0.5]))
        # print("params post: ", wf.params)

        # delete file data/positions_{wf.__class__.__name__}.h5
        (
            os.remove(f"data/positions_{wf.__class__.__name__}.h5")
            if os.path.exists(f"data/positions_{wf.__class__.__name__}.h5")
            else None
        )
        for i in t_range:  # 2**18
            # state.positions = np.zeros((batch_size, 4))
            batch_state = self._step(wf, batch_state, seed, batch_size=batch_size)
            # energies[i * batch_size : (i + 1) * batch_size] = (
            #     self.hamiltonian.local_energy(wf, batch_state.positions)
            # )
            positions[i * batch_size : (i + 1) * batch_size, ...] = (
                batch_state.positions
            )
            if save_positions:
                f = h5py.File(f"data/positions_{wf.__class__.__name__}.h5", "a")
                if i == 0:
                    f.create_dataset(
                        "positions",
                        data=batch_state.positions,
                        compression="gzip",
                        chunks=True,
                        maxshape=(None, batch_state.positions.shape[1]),
                    )
                else:
                    f["positions"].resize(
                        (f["positions"].shape[0] + batch_state.positions.shape[0]),
                        axis=0,
                    )
                    f["positions"][
                        -batch_state.positions.shape[0] :
                    ] = batch_state.positions

        plt.figure(figsize=(10, 10))
        print("positions: ", positions.shape)
        plt.scatter(positions[:, 0], positions[:, 1])
        # # set x lim and y lim
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.show()

        if self._logger is not None:
            t_range.clear()

        assert (
            np.sum(energies == 0) == 0
        ), "There are empty energies which would give wrong statistics"
        energy = np.mean(energies)
        error = block(energies)
        variance = np.mean(energies**2) - energy**2
        acc_rate = state.n_accepted[-1] / nsamples

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
