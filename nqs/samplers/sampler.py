import copy
import os
import warnings

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

    def sample(
        self,
        wf,
        state,
        nsamples,
        nchains=1,
        seed=None,
        save_positions=False,
        foldername=".",
    ):
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
        self.foldername = foldername
        nchains = check_and_set_nchains(nchains, self._logger)
        seeds = generate_seed_sequence(seed, nchains)
        if nchains == 1:
            chain_id = 0

            results, self._energies = self._sample(
                wf, nsamples, state, seeds[0], chain_id, save_positions, foldername
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

    def _sample(
        self, wf, nsamples, state, seed, chain_id, save_positions=False, foldername="."
    ):
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
        batch_size = int(nsamples) // 256

        # Function to ensure data has correct shape
        def ensure_array(data):
            return np.atleast_1d(data)

        foldername = self.foldername
        energy_pos_filename = (
            f"{foldername}_energies_and_pos_{wf.__class__.__name__}_ch{chain_id}.h5"
        )
        obdm_filename = f"{foldername}_obdm_{wf.__class__.__name__}_ch{chain_id}.h5"

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

        if os.path.exists(energy_pos_filename):
            os.remove(energy_pos_filename)
        if os.path.exists(obdm_filename):
            os.remove(obdm_filename)
        for i in t_range:
            batch_state = self._step(wf, batch_state, seed, batch_size=batch_size)
            positions = batch_state.positions

            signs = batch_state.signs
            # print(f"positions.shape: {positions.shape}")
            # print("signs shape", signs.shape)
            ke = self.hamiltonian.local_kinetic_energy(wf, positions)
            pe_trap, pe_int = map(ensure_array, self.hamiltonian.potential(positions))
            energies = {"ke": ke, "pe_trap": pe_trap, "pe_int": pe_int}
            # if obdm:
            #     with h5py.File(obdm_filename, 'a') as f:
            #         if 'xdata' not in f:
            #             f.create_dataset('xdata', (0, N), maxshape=(None, N), compression="gzip", chunks=True)
            #             f.create_dataset('ydata', (0, N), maxshape=(None, N), compression="gzip", chunks=True)
            #             f.create_dataset('zdata', (0, N), maxshape=(None, N), compression="gzip", chunks=True)

            #     self.obdm_sample_1d(positions, logps, signs, wf, obdm_filename)

            with h5py.File(energy_pos_filename, "a") as f1:
                if i == 0:
                    for energy_type, data in energies.items():
                        f1.create_dataset(
                            energy_type,
                            data=data,
                            compression="gzip",
                            chunks=True,
                            maxshape=(None,),
                        )
                    if save_positions:
                        f1.create_dataset(
                            "positions",
                            data=positions,
                            compression="gzip",
                            chunks=True,
                            maxshape=(None, positions.shape[1]),
                        )
                        f1.create_dataset(
                            "signs",
                            data=signs[
                                :, None
                            ],  # Add a new axis to signs to match the shape (65536, 1)
                            compression="gzip",
                            chunks=True,
                            maxshape=(None, 1),
                        )
                else:
                    for energy_type, data in energies.items():
                        f1[energy_type].resize(
                            (f1[energy_type].shape[0] + data.shape[0]), axis=0
                        )
                        f1[energy_type][-data.shape[0] :] = data

                    if save_positions:
                        f1["positions"].resize(
                            (f1["positions"].shape[0] + positions.shape[0]), axis=0
                        )
                        f1["positions"][-positions.shape[0] :] = positions
                        f1["signs"].resize(
                            (f1["signs"].shape[0] + signs.shape[0]), axis=0
                        )
                        f1["signs"][-signs.shape[0] :] = signs[:, None]  #

        if self._logger is not None:
            t_range.clear()

        # open energies file
        f = h5py.File(energy_pos_filename, "r")
        ke = f["ke"][:]
        pe_trap = f["pe_trap"][:]
        pe_int = f["pe_int"][:]
        # pad pe if it is not the same size as ke
        if pe_int.shape[0] != ke.shape[0]:
            pe_int = np.pad(pe_int, (0, ke.shape[0] - pe_int.shape[0]))

        energies = ke + pe_trap + pe_int

        if np.sum(energies == 0) >= 10:
            warnings.warn(
                f"There are {np.sum(energies == 0)} empty energies which would give wrong statistics"
            )

        energy = np.mean(energies)
        error_e = block(energies)
        variance_e = np.mean(energies**2) - energy**2

        error_ke = block(ke)
        error_pe_trap = block(pe_trap)
        error_pe_int = block(pe_int)

        acc_rate = batch_state.n_accepted[-1] / nsamples

        sample_results = {
            "chain_id": chain_id + 1,
            "E_energy": energy,
            "E_variance": variance_e,
            "E_std_error": error_e,
            "K_energy": np.mean(ke),
            "K_std_error": error_ke,
            "PE_trap_energy": np.mean(pe_trap),
            "PE_trap_std_error": error_pe_trap,
            "PE_int_energy": np.mean(pe_int),
            "PE_int_std_error": error_pe_int,
            "accept_rate": acc_rate,
            "scale": self.scale,
            "nsamples": nsamples,
        }

        return sample_results, energies

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

    def obdm_sample_1d(self, positions, logps, signs, wf, filename):
        xmin = -5
        xmax = +5
        # here, positions is actually a batch of positions of size (batch_size, npar x ndim)
        # Create ghost particle configuration
        batch_size = positions.shape[0]
        xps = np.random.uniform(xmin, xmax, positions.shape)
        xdata = np.zeros(
            (batch_size, positions.shape[1])
        )  # this positions.shape[1] is only correct in the case of the 1D harmonic oscillator
        ydata = np.zeros(
            (batch_size, positions.shape[1])
        )  # this positions.shape[1] is only correct in the case of the 1D harmonic oscillator
        zdata = np.zeros(
            (batch_size, positions.shape[1])
        )  # this positions.shape[1] is only correct in the case of the 1D harmonic oscillator

        for i in range(batch_size):
            pos = positions[i]
            sgn = signs[i]
            logp = logps[i]
            xp = xps[i]
            # Evaluate the wave function on both configurations
            sgn_p, logabs_p = wf.logprob(xp)
            # Store the data
            xdata[i] = xp
            ydata[i] = pos
            zdata[i] = (xmax - xmin) * sgn_p * sgn * np.exp(logabs_p - logp)

        # Save the data
        # Append data to the HDF5 file
        with h5py.File(filename, "a") as f:
            f["xdata"].resize((f["xdata"].shape[0] + xdata.shape[0]), axis=0)
            f["xdata"][-xdata.shape[0] :] = xdata

            f["ydata"].resize((f["ydata"].shape[0] + ydata.shape[0]), axis=0)
            f["ydata"][-ydata.shape[0] :] = ydata

            f["zdata"].resize((f["zdata"].shape[0] + zdata.shape[0]), axis=0)
            f["zdata"][-zdata.shape[0] :] = zdata
