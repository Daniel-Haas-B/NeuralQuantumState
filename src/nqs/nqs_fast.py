# import copy
import sys
import warnings

sys.path.insert(0, "../src/")
# print(sys.path)
# from nqs.utils import early_stopping
from nqs.utils import errors
from nqs.utils import generate_seed_sequence
from nqs.utils import setup_logger
from nqs.utils import State
import jax


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


import numpy as np
import pandas as pd

from nqs.models import RBM, FFNN, VMC

from numpy.random import default_rng
from tqdm.auto import tqdm

from physics.hamiltonians import HarmonicOscillator as HO

# sys.path.insert(0, "../samplers/")

from samplers.metropolis_hastings import MetroHastings
from samplers.metropolis import Metropolis as Metro

from optimizers import adam, gd

# import sys
# from abc import abstractmethod
# from functools import partial
# from multiprocessing import Lock
# from multiprocessing import RLock

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

warnings.filterwarnings("ignore", message="divide by zero encountered")


class NQS:
    def __init__(
        self,
        nqs_repr="psi2",
        backend="numpy",
        log=True,
        logger_level="INFO",
        rng=None,
        use_sr=False,
        seed=None,
    ):
        """Neural Network Quantum State
        It is conceptually important to understand that this is the system.
        The system is composed of a wave function, a hamiltonian, a sampler and an optimizer.
        This is the high level class that ties all the other classes together.
        """

        self._check_logger(log, logger_level)
        self._log = log

        self.use_sr = use_sr
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
        For now we only support the RBM.
        Successfully setting the wave function will also initialize it.
        """
        self._N = nparticles
        self._dim = dim
        wf_type = wf_type.lower() if isinstance(wf_type, str) else wf_type
        match wf_type:
            case "rbm":
                self.wf = RBM(
                    nparticles,
                    dim,
                    kwargs["nhidden"],
                    kwargs["sigma2"],
                    log=self._log,
                    logger=self.logger,
                    rng=self.rng(self._seed),
                    backend=self._backend,
                )
            case "ffnn":
                self.wf = FFNN(
                    nparticles,
                    dim,
                    kwargs["layer_sizes"],
                    kwargs["activations"],
                    kwargs["sigma2"],
                    log=self._log,
                    logger=self.logger,
                    rng=self.rng(self._seed),
                    backend=self._backend,
                )
            case "vmc":
                self.wf = VMC(
                    nparticles,
                    dim,
                    log=self._log,
                    logger=self.logger,
                    rng=self.rng(self._seed),
                    backend=self._backend,
                )
            case _:  # noqa
                raise NotImplementedError(
                    "Only the RBM is supported for now. FFNN is WIP"
                )

        self._is_initialized_ = True

    def set_hamiltonian(self, type_, int_type, **kwargs):
        """
        Set the hamiltonian to be used for sampling.
        For now we only support the Harmonic Oscillator.

        Hamiltonian also needs to be propagated to the sampler.

        If int_type is None, we assume non interacting particles.
        """
        if type_.lower() == "ho":
            self.hamiltonian = HO(self._N, self._dim, int_type, self._backend, kwargs)
        else:
            raise NotImplementedError(
                "Only the Harmonic Oscillator and Cologero-Sutherland supported for now."
            )

        self._sampler.set_hamiltonian(self.hamiltonian)

    def set_sampler(self, mcmc_alg, scale=0.5):
        """
        Set the MCMC algorithm to be used for sampling.
        """
        self.scale = scale
        if not isinstance(mcmc_alg, str):
            raise TypeError("'mcmc_alg' must be passed as str")

        self.mcmc_alg = mcmc_alg

        if self.mcmc_alg == "m":
            self._sampler = Metro(self.rng, self.scale, self.logger)
        elif self.mcmc_alg == "lmh":
            self._sampler = MetroHastings(self.rng, self.scale, self.logger)
        else:
            msg = "Unsupported sampler, only Metropolis 'm' or Metropolis-Hastings 'lmh' is allowed"
            raise ValueError(msg)

    def set_optimizer(self, optimizer, eta, **kwargs):
        """
        Set the optimizer algorithm to be used for param update.
        """
        self._eta = eta
        if not isinstance(optimizer, str):
            raise TypeError("'optimizer' must be passed as str")

        if optimizer == "gd":
            self._optimizer = gd.Gd(self.wf.params, eta)  # dumb for now
        elif optimizer == "adam":
            beta1 = kwargs["beta1"] if "beta1" in kwargs else 0.9
            beta2 = kwargs["beta2"] if "beta2" in kwargs else 0.999
            epsilon = kwargs["epsilon"] if "epsilon" in kwargs else 1e-8
            self._optimizer = adam.Adam(
                self.wf.params, eta, beta1=beta1, beta2=beta2, epsilon=epsilon
            )  # _params gets passed to construct the mom and v arrays
        else:
            msg = "Unsupported optimizer, only adam 'adam' or gd 'gd' is allowed"
            raise ValueError(msg)

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
            {"energy": [], "grads": []} if kwargs.get("history", False) else None
        )
        self._early_stop = kwargs.get("early_stop", False)
        self._tune = kwargs.get("tune", False)

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
        seed_seq = generate_seed_sequence(self._seed, 1)[0]

        energies = []
        final_grads = {key: None for key in param_keys}

        expval_energies_dict = {key: None for key in param_keys}
        expval_grad_dict = {key: None for key in param_keys}
        steps_before_optimize = batch_size

        state = self.wf.state
        state = State(state.positions, state.logp, 0, state.delta)

        # equilibrate, burn in lets see if makes a difference
        # for _ in range(1000):
        #    state = self._sampler.step(self.wf, state, seed_seq)
        grads_dict = {key: [] for key in param_keys}
        burn_in = batch_size
        for _ in range(burn_in):
            state = self._sampler.step(self.wf, state, seed_seq)
        for _ in t_range:
            batch_state = self._sampler.batch_step(self.wf, state, seed_seq)

            print("batch_state", batch_state)
            exit()
            loc_energy = self.hamiltonian.local_energy(self.wf, state.positions)
            energies.append(loc_energy)
            local_grads_dict = self.wf.grads(state.positions)

            for key in param_keys:
                grads_dict[key].append(local_grads_dict.get(key))

            steps_before_optimize -= 1
            if steps_before_optimize == 0:
                energies = np.array(energies)
                # print("Energy: ", energies)
                expval_energy = np.mean(energies)

                for key in param_keys:
                    grad_np = np.array(grads_dict[key])

                    new_shape = (batch_size,) + (1,) * (
                        grad_np.ndim - 1
                    )  # Subtracting 1 because the first dimension is already provided by batch_size
                    reshaped_energy = energies.reshape(new_shape)

                    expval_energies_dict[key] = np.mean(
                        reshaped_energy * grad_np, axis=0
                    )

                    expval_grad_dict[key] = np.mean(grad_np, axis=0)
                    final_grads[key] = 2 * (
                        expval_energies_dict[key]
                        - expval_energy * expval_grad_dict[key]
                    )

                if self.use_sr:
                    self.sr_matrices = self.wf.compute_sr_matrix(
                        expval_grad_dict, grads_dict
                    )

                # Descent
                self._optimizer.step(
                    self.wf.params, final_grads, self.sr_matrices
                )  # changes wf params inplace
                if self._history:
                    self._history["energy"].append(expval_energy)
                    grad_norms = [
                        np.linalg.norm(final_grads[key]) for key in param_keys
                    ]
                    self._history["grads"].append(np.mean(grad_norms))

                # self.tune() # tune the sampler scales
                energies = []
                final_grads = {key: None for key in param_keys}
                grads_dict = {key: [] for key in param_keys}
                steps_before_optimize = batch_size

        self.state = state
        self._is_trained_ = True

        if self.logger is not None:
            self.logger.info("Training done")

        if self._history:
            return self._history

    def sample(self, nsamples, nchains=1, seed=None, one_body_density=False):
        """helper for the sample method from the Sampler class"""

        self._is_initialized()
        self._is_trained()

        system_info = {
            "nparticles": self._N,
            "dim": self._dim,
            "eta": self._eta,
            # "nvisible": self._nvisible,
            # "nhidden": self._nhidden,
            "mcmc_alg": self.mcmc_alg,
            "nqs_type": self.nqs_type,
            "training_cycles": self._training_cycles,
            "training_batch": self._training_batch,
            "sr": self.use_sr,
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
                self.wf, self.state, nsamples, 1, seed, lim_inf=-5, lim_sup=5, points=80
            )

        return sample_results

    def tune(
        self,
        tune_iter=2_000,
        tune_interval=200,
        rtol=1e-05,
        atol=1e-08,
        seed=None,
        mcmc_alg=None,
        log=False,
    ):
        """
        BROKEN NOW due to self.scale
        Tune proposal scale so that the acceptance rate is around 0.5.
        """

        self._is_initialized()
        state = self.wf.state
        scale = self.scale
        self._log = log

        # Config
        # did_early_stop = False
        seed_seq = generate_seed_sequence(seed, 1)[0]

        # Reset n_accepted
        # state = State(state.positions, state.logp, 0, state.delta)

        if self._log:
            t_range = tqdm(
                range(tune_iter),
                desc="[Tuning progress]",
                position=0,
                leave=True,
                colour="green",
            )
        else:
            t_range = range(tune_iter)

        steps_before_tune = tune_interval

        for i in t_range:
            state = self._sampler.step(self.wf, state, seed_seq)
            steps_before_tune -= 1

            if steps_before_tune == 0:
                # Tune proposal scale
                old_scale = scale
                accept_rate = state.n_accepted / tune_interval
                scale = self._sampler.tune_scale(old_scale, accept_rate)
                # print("new scale: ", scale)
                # Reset
                steps_before_tune = tune_interval
                state = State(state.positions, state.logp, 0, state.delta)

        # Update shared values
        # self.state = state
        self.scale = scale
        self._is_tuned_ = True