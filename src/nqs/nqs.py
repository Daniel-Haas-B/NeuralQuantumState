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

# from samplers.sampler import Sampler

# from nqs.models import IRBM, JAXIRBM, JAXNIRBM#, NIRBM


from nqs.models.rbm import RBM

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
        if wf_type.lower() == "rbm":
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
        else:
            raise NotImplementedError("Only the RBM is supported for now.")

        if self._backend == "jax":
            self.wf.laplacian_closure = jax.jit(self.wf.laplacian_closure)
            self.wf.grad_closure = jax.jit(self.wf.grad_closure)
            self.wf.grads_closure = jax.jit(self.wf.grads_closure)
            self.wf.wf = jax.jit(self.wf.wf)
            self.wf._softplus = jax.jit(self.wf._softplus)
            self.wf._log_wf = jax.jit(self.wf._log_wf)
            self.wf.pdf = jax.jit(self.wf.pdf)
            self.wf.logprob_closure = jax.jit(self.wf.logprob_closure)
            self.wf.grads = jax.jit(self.wf.grads)
            self.wf.compute_sr_matrix = jax.jit(self.wf.compute_sr_matrix)

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
            if self._backend == "jax":
                self.hamiltonian.potential = jax.jit(self.hamiltonian.potential)
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
            self._sampler = Metro(self.wf, self.rng, scale, logger=self.logger)
        elif self.mcmc_alg == "lmh":
            self._sampler = MetroHastings(self.wf, self.rng, scale, logger=self.logger)
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

    def train(self, max_iter, batch_size, early_stop, **kwargs):
        """
        Train the wave function parameters.
        """
        self._is_initialized()

        self._training_cycles = max_iter
        self._training_batch = batch_size

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

        state = self.wf.state
        state = State(state.positions, state.logp, 0, state.delta)
        params = self.wf.params
        param_keys = params.keys()
        seed_seq = generate_seed_sequence(self._seed, 1)[0]

        energies = []
        final_grads = {key: None for key in param_keys}
        grads_dict = {key: [] for key in param_keys}
        expval_energies_dict = {key: None for key in param_keys}
        expval_grad_dict = {key: None for key in param_keys}
        steps_before_optimize = batch_size

        for _ in t_range:
            state = self._sampler.step(state, params, seed_seq)
            loc_energy = self.hamiltonian.local_energy(
                self.wf, state.positions
            )  # testing adding params
            energies.append(loc_energy)
            grads = self.wf.grads(state.positions)

            for key, grad in zip(param_keys, grads):
                grads_dict[key].append(grad)

            steps_before_optimize -= 1
            if steps_before_optimize == 0:
                energies = np.array(energies)  # dont know why
                expval_energy = np.mean(energies)

                for key in param_keys:
                    reshaped_energy = energies.reshape(
                        batch_size, *(1,) * grads_dict[key][0].ndim
                    )
                    expval_energies_dict[key] = np.mean(
                        reshaped_energy * grads_dict[key], axis=0
                    )

                    expval_grad_dict[key] = np.mean(grads_dict[key], axis=0)

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

                energies = []
                final_grads = {key: None for key in param_keys}
                grads_dict = {key: [] for key in param_keys}
                steps_before_optimize = batch_size

        self.state = state
        self._is_trained_ = True
        if self.logger is not None:
            self.logger.info("Training done")

    def sample(self, nsamples, nchains=1, seed=None):
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
        sample_results = self._sampler.sample(
            self.state, self.wf.params, nsamples, nchains, seed
        )
        system_info_repeated = system_info.loc[
            system_info.index.repeat(len(sample_results))
        ].reset_index(drop=True)

        self._results = pd.concat([system_info_repeated, sample_results], axis=1)

        return self._results
