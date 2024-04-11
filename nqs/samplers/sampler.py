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
    """
    A sampler for quantum wavefunctions.

    This class is designed to sample from a given wavefunction (wf) across different quantum states.
    It supports both single-chain and parallel sampling modes. The sampler can be extended by subclasses
    to implement specific sampling strategies (e.g., Markov Chain Monte Carlo methods).

    Attributes:
        _rng (RandomState): A numpy RandomState object for generating random numbers.
        _scale (float): A scale parameter to adjust the sampling resolution. This parameter should be set by child classes.
        _logger (Logger, optional): An optional logging.Logger object for logging messages. Defaults to None.

    Methods:
        sample_obd(wf, state, nsamples, nchains=1, seed=None, lim_inf=-5, lim_sup=5, method="histogram", points=80):
            Samples the one-body density (OBD) of the wavefunction over a range of positions.

        sample(wf, state, nsamples, nchains=1, seed=None, save_positions=False):
            Samples from the wavefunction, optionally saving the positions and energies to disk.

        _sample(wf, nsamples, state, seed, chain_id, save_positions=False):
            Private method to perform the actual sampling. Intended to be called internally.

        _marginal_sample(wf, nsamples, state, scale, seed=None, chain_id=0):
            Performs fixed-position sampling to calculate the one-body density of the wavefunction.

        step(wf, state, seed):
            Abstract method to perform a single sampling step. This must be implemented by subclasses.

        set_hamiltonian(hamiltonian):
            Sets the Hamiltonian for the sampler. The Hamiltonian is used to calculate local energies during sampling.

        scale:
            Property to get or set the scale parameter.
    """

    def __init__(self, rng, scale, logger=None):
        """
        Initializes the Sampler object.

        Parameters:
            rng (RandomState): A numpy RandomState object for generating random numbers.
            scale (float): A scale parameter to adjust the sampling resolution.
            logger (Logger, optional): An optional logging.Logger object for logging messages. Defaults to None.
        """

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
        """
        Samples the one-body density (OBD) of the wavefunction over a specified range of positions.

        Parameters:
            wf (Wavefunction): The wavefunction to sample from.
            state (QuantumState): The initial quantum state.
            nsamples (int): The number of samples to draw.
            nchains (int, optional): The number of parallel chains for sampling. Currently, only single-chain sampling is implemented.
            seed (int, optional): An optional seed for random number generation. If None, a random seed is used.
            lim_inf (float, optional): The lower bound of the position range for sampling.
            lim_sup (float, optional): The upper bound of the position range for sampling.
            method (str, optional): The method to use for sampling. Currently supports 'histogram' and 'integrate'.
            points (int, optional): The number of points to sample between `lim_inf` and `lim_sup`.

        Returns:
            tuple: A tuple containing the positions and the sampled one-body densities at those positions.
        """
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
        """
        Samples from the wavefunction, optionally saving the positions and energies to disk.

        Parameters:
            wf (Wavefunction): The wavefunction to sample from.
            state (QuantumState): The quantum state to start sampling from.
            nsamples (int): The number of samples to draw.
            nchains (int, optional): The number of parallel chains for sampling. Supports both single and multi-chain modes.
            seed (int, optional): An optional seed for random number generation. If None, a random seed is used.
            save_positions (bool, optional): If True, saves the positions of the sampled states to disk.

        Returns:
            DataFrame: A pandas DataFrame containing the results of the sampling.
        """
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
        """
        Private method to perform the actual sampling. Intended to be called internally.

        Parameters:
            wf (Wavefunction): The wavefunction to sample from.
            nsamples (int): The number of samples to draw.
            state (QuantumState): The quantum state to start sampling from.
            seed (int): A seed for random number generation.
            chain_id (int): The ID of the current chain.
            save_positions (bool): If True, saves the positions of the sampled states to disk.

        Returns:
            tuple: A tuple containing the sampling results and energies.
        """
        batch_size = 2**4
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

        state.delta = 0
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
        Performs fixed-position sampling to calculate the one-body density of the wavefunction.

        Parameters:
            wf (Wavefunction): The wavefunction to sample from.
            nsamples (int): The number of samples to draw.
            state (QuantumState): The quantum state, with one particle's position being fixed.
            scale (float): A scale parameter to adjust the sampling resolution.
            seed (int, optional): A seed for random number generation. If None, a random seed is used.
            chain_id (int, optional): The ID of the current chain. Defaults to 0.

        Returns:
            np.ndarray: The one-body density sampled at the fixed position.
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
        Sets the Hamiltonian for the sampler.

        The Hamiltonian is used to calculate local energies during sampling.

        Parameters:
            hamiltonian (Hamiltonian): The Hamiltonian to use for energy calculations.
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
