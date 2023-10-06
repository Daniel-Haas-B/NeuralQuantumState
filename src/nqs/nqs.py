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


import numpy as np
import pandas as pd

# from samplers.sampler import Sampler

# from nqs.models import IRBM, JAXIRBM, JAXNIRBM#, NIRBM
from nqs.models.base_analytical_rbm import BaseRBM
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
        self.sr_matrix = None
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
            print("Setting WF as RBM")
            self.wf = RBM(
                nparticles,
                dim,
                kwargs["nhidden"],
                kwargs["sigma2"],
                log=self._log,
                logger=self.logger,
                rng=self.rng(self._seed),
            )

        else:
            raise NotImplementedError("Only the RBM is supported for now.")

        self._is_initialized_ = True

    def set_hamiltonian(self, type_, int_type, **kwargs):
        """
        Set the hamiltonian to be used for sampling.
        For now we only support the Harmonic Oscillator.

        Hamiltonian also needs to be propagated to the sampler.

        If int_type is None, we assume non interacting particles.
        """
        if type_.lower() == "ho":
            self.hamiltonian = HO(self._N, self._dim, int_type, self._backend, **kwargs)
        else:
            raise NotImplementedError(
                "Only the Harmonic Oscillator is supported for now."
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
        final_grads = []
        grads_dict = {key: [] for key in param_keys}
        expval_energies_dict = {key: None for key in param_keys}
        expval_grad_dict = {key: None for key in param_keys}
        steps_before_optimize = batch_size

        for _ in t_range:
            state = self._sampler.step(state, params, seed_seq)
            loc_energy = self.hamiltonian.local_energy(self.wf, state.positions)
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
                    final_grads.append(
                        2
                        * (
                            expval_energies_dict[key]
                            - expval_energy * expval_grad_dict[key]
                        )
                    )

                if self.use_sr:
                    self.sr_matrix = self.wf.compute_sr_matrix(
                        expval_grad_dict["kernel"], grads_dict["kernel"]
                    )

                # Descent
                self._optimizer.step(
                    self.wf.params, final_grads, self.sr_matrix
                )  # changes wf params inplace

                energies = []
                final_grads = []
                grads_dict = {key: [] for key in param_keys}
                steps_before_optimize = batch_size

        self.state = state
        self._is_trained_ = True

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


class RBM(BaseRBM):
    def __init__(
        self,
        nparticles,
        dim,
        nhidden=1,
        factor=1.0,
        sigma2=1.0,
        rng=None,
        log=False,
        logger=None,
        logger_level="INFO",
    ):
        """RBM Neural Network Quantum State"""
        super().__init__(factor, sigma2)

        self._N = nparticles
        self._dim = dim
        self._nvisible = self._N * self._dim
        self._nhidden = nhidden
        self.logger = logger

        r = rng.standard_normal(size=self._nvisible)

        # Initialize visible bias
        v_bias = rng.standard_normal(size=self._nvisible) * 0.01
        h_bias = rng.standard_normal(size=self._nhidden) * 0.01
        kernel = rng.standard_normal(size=(self._nvisible, self._nhidden))
        kernel *= np.sqrt(1 / self._nvisible)

        self.params = Parameter()
        self.params.set(["v_bias", "h_bias", "kernel"], [v_bias, h_bias, kernel])
        self.log = log
        logp = self.logprob(r)
        self.state = State(r, logp, 0, 0)

        # if backend == "numpy":
        #    if interaction:
        #        self.rbm = IRBM(self._N, self._dim, factor=self.factor)
        #    else:
        #        self.rbm = NIRBM(factor=self.factor)
        # elif backend == "jax":
        #    if interaction:
        #        self.rbm = JAXIRBM(self._N, self._dim, factor=self.factor)
        #    else:
        #        self.rbm = JAXNIRBM(factor=self.factor)
        # else:
        #    msg = "Unsupported backend, only 'numpy' or 'jax' is allowed"
        #    raise ValueError(msg)

        if self.log:
            neuron_str = "neurons" if self._nhidden > 1 else "neuron"
            msg = (
                f"Neural Network Quantum State initialized as RBM with "
                f"{self._nhidden} hidden {neuron_str}"
            )
            self.logger.info(msg)

    # def tune(
    #     self,
    #     tune_iter=20_000,
    #     tune_interval=500,
    #     early_stop=False,  # set to True later
    #     rtol=1e-05,
    #     atol=1e-08,
    #     seed=None,
    #     mcmc_alg=None,
    # ):
    #     """
    #     !! BROKEN NOW due to self.scale
    #     Tune proposal scale so that the acceptance rate is around 0.5.
    #     """

    #     state = self.state
    #     v_bias, h_bias, kernel = self.wf.params.get(["v_bias", "h_bias", "kernel"])

    #     scale = self.scale

    #     if mcmc_alg is not None:
    #         self._sampler = Sampler(self.mcmc_alg, self.rbm, self.rng, self._log)

    #     # Used to throw warnings if tuned alg mismatch chosen alg
    #     # in other procedures
    #     self._tuned_mcmc_alg = self.mcmc_alg

    #     # Config
    #     # did_early_stop = False
    #     seed_seq = generate_seed_sequence(seed, 1)[0]

    #     # Reset n_accepted
    #     state = State(state.positions, state.logp, 0, state.delta)

    #     if self._log:
    #         t_range = tqdm(
    #             range(tune_iter),
    #             desc="[Tuning progress]",
    #             position=0,
    #             leave=True,
    #             colour="green",
    #         )
    #     else:
    #         t_range = range(tune_iter)

    #     steps_before_tune = tune_interval

    #     for i in t_range:
    #         state = self._sampler.step(state, v_bias, h_bias, kernel, seed_seq)
    #         steps_before_tune -= 1

    #         if steps_before_tune == 0:
    #             # Tune proposal scale
    #             old_scale = scale
    #             accept_rate = state.n_accepted / tune_interval
    #             scale = self._sampler.tune_scale(old_scale, accept_rate)

    #             # Reset
    #             steps_before_tune = tune_interval
    #             state = State(state.positions, state.logp, 0, state.delta)

    #     # Update shared values
    #     self.state = state
    #     self.wf.params.set(["v_bias", "h_bias", "kernel"], [v_bias, h_bias, kernel])
    #     self.scale = scale
    #     self._is_tuned_ = True


class Parameter:
    def __init__(self) -> None:
        self.data = {}

    def set(self, names_or_parameter, values=None):
        if isinstance(names_or_parameter, Parameter):
            self.data = names_or_parameter.data
        elif values is not None:
            for key, value in zip(names_or_parameter, values):
                self.data[key] = value
        else:
            raise ValueError("Invalid arguments")

    def get(self, names):
        # note this can be a list of names
        return [self.data[name] for name in names]

    def keys(self):
        return self.data.keys()

    # def parallelize(self, nchains):
    #     """
    #     will make something like
    #         v_bias = (v_bias,) * nchains
    #         h_bias = (h_bias,) * nchains
    #         kernel = (kernel,) * nchains
    #     If self.data = {'v_bias': v_bias, 'h_bias': h_bias, 'kernel': kernel}
    #     """
    #     for key in self.data.keys():
    #         self.data[key] = (self.data[key],) * nchains
    #     return self.data

    # Implementing the __iter__ method
    # def __iter__(self):
    #     return iter(self.data.items())
