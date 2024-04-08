import warnings

import numpy as np
import pandas as pd
from numpy.random import default_rng
from tqdm.auto import tqdm

from nqs.physics.hamiltonians import HarmonicOscillator as HO
from nqs.samplers import Metropolis as Metro
from nqs.samplers import MetropolisHastings as MetroHastings
from nqs.state import pretrain
from nqs.state.utils import errors
from nqs.state.utils import generate_seed_sequence
from nqs.state.utils import optimizer_factory
from nqs.state.utils import setup_logger
from nqs.state.utils import State
from nqs.state.utils import tune_sampler
from nqs.state.utils import wf_factory

warnings.filterwarnings("ignore", message="divide by zero encountered")


class NQS:
    """Represents a Neural Network Quantum State (NQS) for simulating quantum systems.

    This class ties together various components such as the wave function, Hamiltonian,
    sampler, and optimizer to represent and simulate a quantum system.

    Attributes:
        logger (logging.Logger): Logger instance for the class.
        nqs_type (str): Type of NQS representation ('psi' or 'psi2').
        hamiltonian: Instance of the Hamiltonian used for simulation.
        mcmc_alg (str): Type of MCMC algorithm used ('m' for Metropolis, 'lmh' for Metropolis-Hastings).
        _optimizer: Optimizer instance for parameter update.
        sr_matrices: Stochastic Reconfiguration matrices, if applicable.
        wf: Wave function instance for the simulation.
        _seed (int): Seed for random number generation to ensure reproducibility.
    """

    def __init__(
        self,
        nqs_repr="psi2",
        backend="numpy",
        logger_level="INFO",
        rng=None,
        seed=None,
    ):
        """
        Initialize the Neural Quantum State (NQS) simulation environment.

        Parameters:
            nqs_repr (str): Representation of the NQS, either 'psi' for wave function itself or 'psi2' for the wave function amplitude. Default is 'psi2'.
            backend (str): The backend to use for calculations. Default is 'numpy'.
            log (bool): Flag to determine if logging is enabled. Default is True.
            logger_level (str): The logging level. Default is 'INFO'.
            rng (Generator, optional): A NumPy random number generator instance. If None, a default generator is created.
            seed (int, optional): A seed for the random number generator to ensure reproducibility.
        """

        self.logger_level = logger_level
        self._check_logger()

        self.nqs_type = None
        self.hamiltonian = None
        self._backend = backend
        self.mcmc_alg = None
        self._optimizer = None
        self.sr_matrices = None
        self.wf = None
        self._seed = seed

        if logger_level != "SILENT":
            self.logger = setup_logger(self.__class__.__name__, level=logger_level)
        else:
            self.logger = None

        if rng is None:
            self.rng = default_rng

        if nqs_repr == "psi":
            self.factor = 1.0
        elif nqs_repr == "psi2":
            self.factor = 0.5
        else:
            msg = (
                "The NQS can only represent the wave function itself "
                "('psi') or the wave function amplitude ('psi2')"
            )
            raise ValueError(msg)

        # flags
        self._is_initialized_ = False
        self._is_trained_ = False
        self._is_tuned_ = False
        self._sampling_performed

    def set_wf(self, wf_type, nparticles, dim, **kwargs):
        """
        Set and initialize the wave function for the NQS simulation.

        Parameters:
            wf_type (str): The type of wave function to use.
            nparticles (int): The number of particles in the system.
            dim (int): The dimensionality of the system.
            **kwargs: Additional keyword arguments for the wave function initialization.

        Raises:
            ValueError: If an invalid wave function type is provided.
        """
        self.N = nparticles
        self.dim = dim
        self._symmetry = kwargs.get("symmetry", "none")
        common_args = {
            "nparticles": self.N,
            "dim": self.dim,
            "rng": self.rng(self._seed) if self.rng else np.random.default_rng(),
            "logger": self.logger,
            "logger_level": "INFO",
            "backend": self._backend,
        }
        specific_args = kwargs
        self.wf = wf_factory(wf_type, **common_args, **specific_args)
        self.nqs_type = wf_type
        self._is_initialized_ = True

    def set_hamiltonian(self, type_, int_type, **kwargs):
        """
        Set the Hamiltonian for the NQS simulation.

        Parameters:
            type_ (str): The type of Hamiltonian to use (e.g., 'ho' for Harmonic Oscillator).
            int_type (str): The type of interaction between particles.
            **kwargs: Additional keyword arguments for Hamiltonian initialization.

        Raises:
            NotImplementedError: If an unsupported Hamiltonian type is provided.
        """
        if type_.lower() == "ho":
            self.hamiltonian = HO(self.N, self.dim, int_type, self._backend, kwargs)
            self.wf.sqrt_omega = np.sqrt(kwargs.get("omega", 1.0))
        else:
            raise NotImplementedError(
                "Only the Harmonic Oscillator and Cologero-Sutherland supported for now."
            )

        self._sampler.set_hamiltonian(self.hamiltonian)

    def set_sampler(self, mcmc_alg, scale=0.5):
        """
        Set the Markov Chain Monte Carlo (MCMC) sampling algorithm for the NQS simulation.

        Parameters:
            mcmc_alg (str): The type of MCMC algorithm to use ('m' for Metropolis, 'lmh' for Metropolis-Hastings).
            scale (float): The scale parameter for the MCMC sampling. Default is 0.5.

        Raises:
            ValueError: If an unsupported MCMC algorithm is specified."""

        if not isinstance(mcmc_alg, str):
            raise TypeError("'mcmc_alg' must be passed as str")

        self.mcmc_alg = mcmc_alg

        if self.mcmc_alg == "m":
            self._sampler = Metro(self.rng, scale, self.logger)
        elif self.mcmc_alg == "lmh":
            self._sampler = MetroHastings(self.rng, scale, self.logger)
        else:
            msg = "Unsupported sampler, only Metropolis 'm' or Metropolis-Hastings 'lmh' is allowed"
            raise ValueError(msg)

    def set_optimizer(self, optimizer, eta, **kwargs):
        """
        Set the optimization algorithm for parameter updates in the NQS simulation.

        Parameters:
            optimizer (str): The optimizer to use (e.g., 'adam').
            eta (float): The learning rate.
            **kwargs: Additional keyword arguments for the optimizer.

        Raises:
            ValueError: If an unsupported optimizer is specified."""
        self._eta = eta
        common_args = {
            "params": self.wf.params,
            "eta": eta,
        }
        self._optimizer = optimizer_factory(optimizer, **common_args, **kwargs)

    def _is_initialized(self):
        """
        Check if the NQS simulation environment has been initialized properly.

        Raises:
            NotInitialized: If the environment has not been initialized.
        """
        if not self._is_initialized_:
            msg = "A call to 'init' must be made before training"
            raise errors.NotInitialized(msg)

    def _is_trained(self):
        """
        Check if the NQS model has been trained.

        Raises:
            NotTrained: If the model has not undergone training.
        """
        if not self._is_trained_:
            msg = "A call to 'train' must be made before sampling"
            raise errors.NotTrained(msg)

    def _sampling_performed(self):
        """
        Check if sampling has been performed.

        Raises:
            SamplingNotPerformed: If sampling has not yet been performed.
        """
        if not self._is_trained_:
            msg = "A call to 'sample' must be made in order to access results"
            raise errors.SamplingNotPerformed(msg)

    def _check_logger(self):
        """
        Check the validity of logging parameters.

        Parameters:
            logger_level (str): The logging level.

        Raises:
            TypeError: If 'logger_level' is not a string.
        """

        if not isinstance(self.logger_level, str):
            raise TypeError("'logger_level' must be passed as str")

    def train(self, max_iter, batch_size, **kwargs):
        """
        Train the wave function parameters using the provided optimizer and sampler.

        Parameters:
            max_iter (int): The maximum number of training iterations.
            batch_size (int): The batch size for sampling.
            **kwargs: Additional keyword arguments for training control, such as 'history', 'early_stop', etc.

        Returns:
            dict: A dictionary containing training history if 'history' is enabled, otherwise None.

        Raises:
            Various exceptions for invalid training states or configurations.
        """
        self._is_initialized()
        self._training_cycles = max_iter
        self._training_batch = batch_size
        self._history = (
            {"energy": [], "std": [], "grad_params": []}
            if kwargs.get("history", False)
            else None
        )

        self._early_stop = kwargs.get("early_stop", False)
        self._tune = kwargs.get("tune", False)
        self._grad_clip = kwargs.get("grad_clip", False)
        self._agent = kwargs.get("agent", False)

        if self.logger_level != "SILENT":
            t_range = tqdm(
                range(max_iter),
                desc="[Training progress]",
                position=0,
                leave=True,
                colour="green",
            )
        else:
            t_range = range(max_iter)

        params = self.wf.params
        param_keys = params.keys()
        if self._history:
            self._history.update({key: [] for key in param_keys})
        seed_seq = generate_seed_sequence(self._seed, 1)[0]

        energies = []
        grad_params_E = {key: None for key in param_keys}

        expval_energies_dict = {key: None for key in param_keys}
        expval_grad_dict = {key: None for key in param_keys}
        # steps_before_optimize = batch_size

        self.state = self.wf.state
        grad_params_dict = {key: [] for key in param_keys}
        epoch = 0
        states = self.state.create_batch_of_states(batch_size=batch_size)
        for _ in t_range:
            epoch += 1
            # states = self.state.create_batch_of_states(batch_size=batch_size)

            # print("states", states[-1].positions.type)
            # reset the acceptance rate
            states.n_accepted = np.zeros(batch_size)
            states = self._sampler.step(
                self.wf, states, seed_seq, batch_size=batch_size
            )
            current_acc = states.n_accepted[-1] / batch_size

            tune_batch = 1000  # int(batch_size*0.1) # needs to be big else does not stabilize the acceptance rate
            tune_iter = 100
            if not (current_acc > 0.3 and current_acc < 0.7) and self._tune:
                # this will be very inneficient now
                tune_sampler(
                    wf=self.wf,
                    current_state=self.state,
                    sampler=self._sampler,
                    seed=self._seed,
                    tune_batch=tune_batch,
                    tune_iter=tune_iter,
                    mode="standard",
                    logger=self.logger,
                    logger_level=self.logger_level,
                )

                # then kinda do it again
                states = self.state.create_batch_of_states(batch_size=batch_size)
                states = self._sampler.step(
                    self.wf, states, seed_seq, batch_size=batch_size
                )
                current_acc = states.n_accepted[-1] / batch_size

            energies = self.hamiltonian.local_energy(self.wf, states.positions)
            local_grad_params_dict = self.wf.grad_params(states.positions)

            energies = np.array(energies)
            expval_energy = np.mean(energies)
            sigma_l1 = np.mean(np.abs(energies - expval_energy))

            # Define the acceptable window as ⟨EL⟩ ±5σℓ1
            lower_bound = expval_energy - 5 * sigma_l1
            upper_bound = expval_energy + 5 * sigma_l1

            # clip
            energies = np.clip(energies, lower_bound, upper_bound)
            # comput expval_energy again
            expval_energy = np.mean(energies)

            std_energy = np.std(energies)
            if self.logger_level != "SILENT":
                t_range.set_postfix(
                    avg_E_l=f"{expval_energy:.2f}",
                    acc=f"{current_acc:.2f}",
                    refresh=True,
                )

            for key in param_keys:
                grad_np = np.array(local_grad_params_dict.get(key))
                grad_params_dict[key] = grad_np
                if self._history:
                    self._history[key].append(np.linalg.norm(grad_np))
                new_shape = (batch_size,) + (1,) * (
                    grad_np.ndim - 1
                )  # Subtracting 1 because the first dimension is already provided by batch_size
                energies = energies.reshape(new_shape)

                expval_energies_dict[key] = np.mean(energies * grad_np, axis=0)
                expval_grad_dict[key] = np.mean(grad_np, axis=0)

                grad_params_E[key] = 2 * (
                    expval_energies_dict[key] - expval_energy * expval_grad_dict[key]
                )

                if self._grad_clip:
                    # print("grad_np before", grad_np.shape )
                    grad_norm = np.linalg.norm(grad_params_E[key])
                    # print("grad_norm", grad_norm)
                    if grad_norm > self._grad_clip:
                        grad_params_E[key] = (
                            self._grad_clip * grad_params_E[key] / grad_norm
                        )

            if self._optimizer.__class__.__name__ == "Sr":
                self.sr_matrices = self.wf.compute_sr_matrix(
                    expval_grad_dict, grad_params_dict
                )

            grad_norms = [np.linalg.norm(grad_params_E[key]) for key in param_keys]
            grad_norms = np.mean(grad_norms)
            if self._history:
                self._history["energy"].append(expval_energy)
                self._history["std"].append(std_energy)
                self._history["grad_params"].append(grad_norms)

            # Descent
            self._optimizer.step(
                self.wf.params, grad_params_E, self.sr_matrices
            )  # changes wf params inplace

            if grad_norms < 10**-15:
                if self.logger is not None:
                    self.logger.warning("Gradient norm is zero, stopping training")
                break

            # update wf state after each epoch?
            self.state = State(
                states[-1].positions, states[-1].logp, 0, states[-1].delta
            )
            grad_params_E = {key: None for key in param_keys}
            grad_params_dict = {key: [] for key in param_keys}

        self._is_trained_ = True
        if self.logger_level != "SILENT":
            self.logger.info("Training done")
        if self._history:
            return self._history

    def sample(
        self,
        nsamples,
        nchains=1,
        seed=None,
        one_body_density=False,
        save_positions=False,
    ):
        """
        Perform sampling of the system's quantum state using the previously set wave function and sampler.

        Parameters:
            nsamples (int): The number of samples to generate.
            nchains (int): The number of independent Markov chains to use for sampling. Default is 1.
            seed (int, optional): A seed for the random number generator to ensure reproducibility of the sampling process.
            one_body_density (bool): If True, computes the one-body density of the quantum state. Default is False.
            save_positions (bool): If True, saves the particle positions during sampling. Default is False.

        Returns:
            pd.DataFrame: A DataFrame containing the sampling results or one-body density results, depending on the parameters.

        Raises:
            Various exceptions if the system is not properly initialized or trained.
        """

        self._is_initialized()
        self._is_trained()
        self.hamiltonian.turn_reg_off()

        system_info = {
            "nparticles": self.N,
            "dim": self.dim,
            "eta": self._eta,
            "mcmc_alg": self.mcmc_alg,
            "nqs_type": self.nqs_type,
            "training_cycles": self._training_cycles,
            "training_batch": self._training_batch,
            "Opti": self._optimizer.__class__.__name__,
            "symmetry": self._symmetry,
        }

        system_info = pd.DataFrame(system_info, index=[0])

        if not one_body_density:
            sample_results = self._sampler.sample(
                self.wf, self.state, nsamples, nchains, seed, save_positions
            )
            print("sample_results", sample_results)
            sample_results["accept_rate"] = float(sample_results["accept_rate"].iloc[0])

            # convert to numpy array

            system_info_repeated = system_info.loc[
                system_info.index.repeat(len(sample_results))
            ].reset_index(drop=True)

            self._results = pd.concat([system_info_repeated, sample_results], axis=1)

            return self._results

        else:
            sample_results = self._sampler.sample_obd(
                self.wf,
                self.state,
                nsamples,
                1,
                seed,
                lim_inf=-5,
                lim_sup=5,
                method="histogram",
                # points=200, # if method is integral
            )

        return sample_results

    def pretrain(self, model, max_iter, batch_size, args=None, **kwargs):
        """
        Pretrain the wave function using a specified model. This is typically used to initialize the wave function parameters to sensible values before the main training phase.

        Parameters:
            model (str): The pretraining model to use (e.g., 'gaussian' for Gaussian-based pretraining).
            max_iter (int): The maximum number of iterations for the pretraining.
            batch_size (int): The batch size to use during pretraining.
            args (dict, optional): Additional arguments specific to the pretraining model.

        Raises:
            NotImplementedError: If a non-supported pretraining model is specified.
            ValueError: If invalid model-specific parameters are provided.

        Returns:
            None: The method updates the wave function parameters in place.
        """
        if model.lower() == "gaussian":
            pre_system = pretrain.Gaussian(
                logger_level=kwargs.get("logger_level", "INFO"),
                seed=self._seed * 2,
            )
            pre_system.set_wf(
                self.wf.__class__.__name__.lower(),
                self.N,
                self.dim,
                **args,
            )
            # if jastrow, save the WJ params to be used later
            if str(args["correlation"]).lower() == "j":
                WJ_params = self.wf.params.get("WJ")
            elif str(args["correlation"]).lower() == "pj":
                CPJ_params = self.wf.params.get("CPJ")
            elif args["correlation"] is not None:
                if args["correlation"].lower() != "none":
                    raise ValueError(
                        f"Invalid correlation {args['correlation']} of type {type(args['correlation'])}"
                    )
        else:
            raise NotImplementedError("Only Gaussian pretraining is supported for now")

        pre_system.set_optimizer(
            optimizer="adam",
            eta=0.1,
            gamma=0,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )

        params = pre_system.pretrain(
            max_iter=max_iter,
            batch_size=batch_size,
            seed=self._seed * 2,
            history=False,
            pretrain_sampler=False,  # there is no true for now
            pretrain_jastrow=False,  # there is no true for now
        )
        self.wf.params = params
        if str(args["correlation"]).lower() == "j":
            self.wf.params.set("WJ", WJ_params)
        elif str(args["correlation"]).lower() == "pj":
            self.wf.params.set("CPJ", CPJ_params)
