import warnings

import numpy as np
import pandas as pd
from numpy.random import default_rng
from tqdm.auto import tqdm

from src.physics.hamiltonians import HarmonicOscillator as HO
from src.samplers import Metropolis as Metro
from src.samplers import MetropolisHastings as MetroHastings
from src.state import pretrain
from src.state.utils import errors
from src.state.utils import generate_seed_sequence
from src.state.utils import optimizer_factory
from src.state.utils import setup_logger
from src.state.utils import State
from src.state.utils import tune_sampler
from src.state.utils import wf_factory

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
        log=True,
        logger_level="INFO",
        rng=None,
        seed=None,
    ):

        self._check_logger(log, logger_level)
        self._log = log

        self.nqs_type = None
        self.hamiltonian = None
        self._backend = backend
        self.mcmc_alg = None
        self._optimizer = None
        self.sr_matrices = None
        self.wf = None
        self._seed = seed

        if self._log:
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
        Set the wave function to be used for sampling.
        Successfully setting the wave function will also initialize it.
        """
        self._N = nparticles
        self._dim = dim
        self._symmetry = kwargs.get("symmetry", "none")
        common_args = {
            "nparticles": self._N,
            "dim": self._dim,
            "rng": self.rng(self._seed) if self.rng else np.random.default_rng(),
            "log": self._log,
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
        Set the hamiltonian to be used for sampling.

        Hamiltonian also needs to be propagated to the sampler.

        If int_type is None, we assume non interacting particles.
        """
        if type_.lower() == "ho":
            self.hamiltonian = HO(self._N, self._dim, int_type, self._backend, kwargs)
            self.wf.sqrt_omega = np.sqrt(kwargs.get("omega", 1.0))
        else:
            raise NotImplementedError(
                "Only the Harmonic Oscillator and Cologero-Sutherland supported for now."
            )

        self._sampler.set_hamiltonian(self.hamiltonian)

    def set_sampler(self, mcmc_alg, scale=0.5):
        """
        Set the MCMC algorithm to be used for sampling.
        """

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
        Set the optimizer algorithm to be used for param update.
        """
        self._eta = eta
        common_args = {
            "params": self.wf.params,
            "eta": eta,
        }
        self._optimizer = optimizer_factory(optimizer, **common_args, **kwargs)

    def _is_initialized(self):
        if not self._is_initialized_:
            msg = "A call to 'init' must be made before training"
            raise errors.NotInitialized(msg)

    def _is_trained(self):
        if not self._is_trained_:
            msg = "A call to 'train' must be made before sampling"
            raise errors.NotTrained(msg)

    def _sampling_performed(self):
        if not self._is_trained_:
            msg = "A call to 'sample' must be made in order to access results"
            raise errors.SamplingNotPerformed(msg)

    def _check_logger(self, log, logger_level):
        if not isinstance(log, bool):
            raise TypeError("'log' must be True or False")

        if not isinstance(logger_level, str):
            raise TypeError("'logger_level' must be passed as str")

    def train(self, max_iter, batch_size, **kwargs):
        """
        Train the wave function parameters.
        """
        self._is_initialized()
        self._training_cycles = max_iter
        self._training_batch = batch_size
        self._history = (
            {"energy": [], "std": [], "grads": []}
            if kwargs.get("history", False)
            else None
        )

        self._early_stop = kwargs.get("early_stop", False)
        self._tune = kwargs.get("tune", False)
        self._grad_clip = kwargs.get("grad_clip", False)
        self._agent = kwargs.get("agent", False)

        if self._log:
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
        final_grads = {key: None for key in param_keys}

        expval_energies_dict = {key: None for key in param_keys}
        expval_grad_dict = {key: None for key in param_keys}
        # steps_before_optimize = batch_size

        self.state = self.wf.state
        grads_dict = {key: [] for key in param_keys}
        epoch = 0

        for _ in t_range:
            epoch += 1
            states = self.state.create_batch_of_states(batch_size=batch_size)

            # print("states", states[-1].positions.type)
            states = self._sampler.step(
                self.wf, states, seed_seq, batch_size=batch_size
            )
            current_acc = states[-1].n_accepted / batch_size

            tune_batch = 1000  # int(batch_size*0.1) # needs to be big else does not stabilize the acceptance rate
            tune_iter = 100
            if not (current_acc > 0.3 and current_acc < 0.7) and self._tune:
                # this will be very inneficient now
                tune_sampler(
                    wf=self.wf,
                    current_state=self.state,
                    sampler=self._sampler,
                    seed=self._seed,
                    log=self._log,
                    tune_batch=tune_batch,
                    tune_iter=tune_iter,
                    mode="standard",
                    logger=self.logger,
                )

                # then kinda do it again
                states = self.state.create_batch_of_states(batch_size=batch_size)
                states = self._sampler.step(
                    self.wf, states, seed_seq, batch_size=batch_size
                )
                current_acc = states[-1].n_accepted / batch_size

            energies = self.hamiltonian.local_energy(self.wf, states.positions)
            local_grads_dict = self.wf.grads(states.positions)

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
            t_range.set_postfix(
                avg_E_l=f"{expval_energy:.2f}", acc=f"{current_acc:.2f}", refresh=True
            )

            for key in param_keys:
                grad_np = np.array(local_grads_dict.get(key))
                grads_dict[key] = grad_np
                if self._history:
                    self._history[key].append(np.linalg.norm(grad_np))
                new_shape = (batch_size,) + (1,) * (
                    grad_np.ndim - 1
                )  # Subtracting 1 because the first dimension is already provided by batch_size
                energies = energies.reshape(new_shape)

                expval_energies_dict[key] = np.mean(energies * grad_np, axis=0)
                expval_grad_dict[key] = np.mean(grad_np, axis=0)

                final_grads[key] = 2 * (
                    expval_energies_dict[key] - expval_energy * expval_grad_dict[key]
                )

                if self._grad_clip:
                    # print("grad_np before", grad_np.shape )
                    grad_norm = np.linalg.norm(final_grads[key])
                    # print("grad_norm", grad_norm)
                    if grad_norm > self._grad_clip:
                        final_grads[key] = (
                            self._grad_clip * final_grads[key] / grad_norm
                        )

            if self._optimizer.__class__.__name__ == "Sr":
                self.sr_matrices = self.wf.compute_sr_matrix(
                    expval_grad_dict, grads_dict
                )

            grad_norms = [np.linalg.norm(final_grads[key]) for key in param_keys]
            grad_norms = np.mean(grad_norms)
            if self._history:
                self._history["energy"].append(expval_energy)
                self._history["std"].append(std_energy)
                self._history["grads"].append(grad_norms)

            # Descent
            self._optimizer.step(
                self.wf.params, final_grads, self.sr_matrices
            )  # changes wf params inplace

            if grad_norms < 10**-15:
                if self.logger is not None:
                    self.logger.warning("Gradient norm is zero, stopping training")
                break

            # update wf state after each epoch?
            self.state = State(
                states[-1].positions, states[-1].logp, 0, states[-1].delta
            )
            final_grads = {key: None for key in param_keys}
            grads_dict = {key: [] for key in param_keys}

        self._is_trained_ = True
        if self.logger is not None:
            self.logger.info("Training done")
        if self._history:
            return self._history

    def sample(self, nsamples, nchains=1, seed=None, one_body_density=False):
        """helper for the sample method from the Sampler class"""
        self._is_initialized()
        self._is_trained()
        self.hamiltonian.turn_reg_off()

        system_info = {
            "nparticles": self._N,
            "dim": self._dim,
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
                self.wf, self.state, nsamples, nchains, seed
            )
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
                points=200,
            )

        return sample_results

    def pretrain(self, model, max_iter, batch_size, args=None):
        """
        # TODO: make this less repetitive
        """
        if model.lower() == "gaussian":
            pre_system = pretrain.Gaussian(
                log=True,
                logger_level="INFO",
                seed=self._seed * 2,
            )
            pre_system.set_wf(
                self.wf.__class__.__name__.lower(),
                self._N,
                self._dim,
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
