import copy
import sys
import warnings

sys.path.insert(0, "../src/")
# print(sys.path)
from nqs.utils import early_stopping
from nqs.utils import errors
from nqs.utils import generate_seed_sequence
from nqs.utils import setup_logger
from nqs.utils import State


import numpy as np
import pandas as pd
from samplers.sampler import Sampler
from nqs.models import IRBM, JAXIRBM, JAXNIRBM, NIRBM
from numpy.random import default_rng
from tqdm.auto import tqdm

# sys.path.insert(0, "../samplers/")

from samplers.metropolis_hastings import MetroHastings
from samplers.metropolis import Metropolis as Metro


# import sys
# from abc import abstractmethod
# from functools import partial
# from multiprocessing import Lock
# from multiprocessing import RLock

warnings.filterwarnings("ignore", message="divide by zero encountered")


class RBMNQS:
    def __init__(
        self,
        nparticles,
        dim,
        nhidden=1,
        interaction=False,
        nqs_repr="psi2",
        backend="numpy",
        log=True,
        logger_level="INFO",
        rng=None,
    ):
        """RBM Neural Network Quantum State"""

        self._check_logger(log, logger_level)
        self._log = log
        self.mcmc_alg = None

        if self._log:
            self.logger = setup_logger(self.__class__.__name__, level=logger_level)
        else:
            self.logger = None

        self._P = nparticles
        self._dim = dim
        self._nhidden = nhidden
        self._nvisible = self._P * self._dim

        if rng is None:
            self.rng = default_rng

        if nqs_repr == "psi":
            factor = 1.0
        elif nqs_repr == "psi2":
            factor = 0.5
        else:
            msg = (
                "The NQS can only represent the wave function itself "
                "('psi') or the wave function amplitude ('psi2')"
            )
            raise ValueError(msg)

        if backend == "numpy":
            if interaction:
                self.rbm = IRBM(self._P, self._dim, factor=factor)
            else:
                self.rbm = NIRBM(factor=factor)
        elif backend == "jax":
            if interaction:
                self.rbm = JAXIRBM(self._P, self._dim, factor=factor)
            else:
                self.rbm = JAXNIRBM(factor=factor)
        else:
            msg = "Unsupported backend, only 'numpy' or 'jax' is allowed"
            raise ValueError(msg)

        # set mcmc step

        # self._sampler = Sampler(self.rbm, self.rng, logger=self.logger)

        if self._log:
            neuron_str = "neurons" if self._nhidden > 1 else "neuron"
            msg = (
                f"Neural Network Quantum State initialized as RBM with "
                f"{self._nhidden} hidden {neuron_str}"
            )
            self.logger.info(msg)

        # flags
        self._is_initialized_ = False
        self._is_trained_ = False
        self._is_tuned_ = False
        self._sampling_performed_ = False

    def set_sampler(self, mcmc_alg, scale=0.5):
        """
        Set the MCMC algorithm to be used for sampling.
        """

        if not isinstance(mcmc_alg, str):
            raise TypeError("'mcmc_alg' must be passed as str")

        if mcmc_alg == "m":
            self.mcmc_alg = "m"
            self._sampler = Metro(self.rbm, self.rng, scale, logger=self.logger)
        elif mcmc_alg == "lmh":
            self.mcmc_alg = "lmh"
            self._sampler = MetroHastings(self.rbm, self.rng, scale, logger=self.logger)
        else:
            msg = "Unsupported backend, only 'numpy' or 'jax' is allowed"
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

    def init(self, sigma2=1.0, seed=None):
        """ """
        self.rbm.sigma2 = sigma2
        # self.scale = scale
        # print("init scale", self.scale)
        rng = self.rng(seed)

        r = rng.standard_normal(size=self._nvisible)

        # Initialize visible bias
        self._v_bias = rng.standard_normal(size=self._nvisible) * 0.01
        # self._v_bias = np.zeros(self._nvisible)

        # Initialize hidden bias
        self._h_bias = rng.standard_normal(size=self._nhidden) * 0.01
        # self._h_bias = np.zeros(self._nhidden)

        # Initialize kernel (weight matrix)
        self._kernel = rng.standard_normal(size=(self._nvisible, self._nhidden))
        # self._kernel *= np.sqrt(1 / self._nvisible)
        self._kernel *= np.sqrt(1 / self._nvisible)
        # self._kernel *= np.sqrt(2 / (self._nvisible + self._nhidden))

        logp = self.rbm.logprob(r, self._v_bias, self._h_bias, self._kernel)
        self.state = State(r, logp, 0, 0)
        self._is_initialized_ = True

    def train(
        self,
        max_iter=100_000,
        batch_size=1000,
        optimizer="adam",
        eta=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        early_stop=False,  # set to True later
        rtol=1e-05,
        atol=1e-08,
        seed=None,
        mcmc_alg=None,
    ):
        """
        Train the NQS model using the specified gradient method.
        """

        self._is_initialized()
        self._training_cycles = max_iter
        self._training_batch = batch_size
        self._eta = eta

        if mcmc_alg is not None:
            self._sampler = Sampler(
                self.mcmc_alg, self.rbm, self.rng, logger=self.logger
            )

        state = self.state
        v_bias = self._v_bias
        h_bias = self._h_bias
        kernel = self._kernel

        # Reset n_accepted
        state = State(state.positions, state.logp, 0, state.delta)

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

        # initialize optimizer

        # Set parameters for Adam
        if optimizer == "adam":
            t = 0
            # visible bias
            m_v_bias = np.zeros_like(v_bias)
            v_v_bias = np.zeros_like(v_bias)
            # hidden bias
            m_h_bias = np.zeros_like(h_bias)
            v_h_bias = np.zeros_like(h_bias)
            # kernel
            m_kernel = np.zeros_like(kernel)
            v_kernel = np.zeros_like(kernel)

        # Config
        did_early_stop = False
        seed_seq = generate_seed_sequence(seed, 1)[0]
        steps_before_optimize = batch_size
        energies = []
        grads_v_bias = []
        grads_h_bias = []
        grads_kernel = []

        # Training
        for _ in t_range:
            state = self._sampler.step(state, v_bias, h_bias, kernel, seed_seq)
            loc_energy = self.rbm.local_energy(state.positions, v_bias, h_bias, kernel)
            gr_v_bias = self.rbm.grad_v_bias(state.positions, v_bias, h_bias, kernel)
            gr_h_bias = self.rbm.grad_h_bias(state.positions, v_bias, h_bias, kernel)
            gr_kernel = self.rbm.grad_kernel(state.positions, v_bias, h_bias, kernel)
            energies.append(loc_energy)
            grads_v_bias.append(gr_v_bias)
            grads_h_bias.append(gr_h_bias)
            grads_kernel.append(gr_kernel)

            steps_before_optimize -= 1

            if steps_before_optimize == 0:
                # Expectation values
                energies = np.array(energies)
                grads_v_bias = np.array(grads_v_bias)
                grads_h_bias = np.array(grads_h_bias)
                grads_kernel = np.array(grads_kernel)

                expval_energy = np.mean(energies)
                expval_grad_v_bias = np.mean(grads_v_bias, axis=0)
                expval_grad_h_bias = np.mean(grads_h_bias, axis=0)
                expval_grad_kernel = np.mean(grads_kernel, axis=0)
                expval_energy_v_bias = np.mean(
                    energies.reshape(batch_size, 1) * grads_v_bias, axis=0
                )
                expval_energy_h_bias = np.mean(
                    energies.reshape(batch_size, 1) * grads_h_bias, axis=0
                )
                expval_energy_kernel = np.mean(
                    energies.reshape(batch_size, 1, 1) * grads_kernel, axis=0
                )

                # variance = np.mean(energies**2) - energy**2

                # Gradients
                final_gr_v_bias = 2 * (
                    expval_energy_v_bias - expval_energy * expval_grad_v_bias
                )
                final_gr_h_bias = 2 * (
                    expval_energy_h_bias - expval_energy * expval_grad_h_bias
                )
                final_gr_kernel = 2 * (
                    expval_energy_kernel - expval_energy * expval_grad_kernel
                )

                if early_stopping:
                    # make copies of current values before update
                    v_bias_old = copy.deepcopy(v_bias)  # noqa
                    h_bias_old = copy.deepcopy(h_bias)  # noqa
                    kernel_old = copy.deepcopy(kernel)  # noqa

                # Gradient descent
                if optimizer == "gd":
                    v_bias -= eta * final_gr_v_bias
                    h_bias -= eta * final_gr_h_bias
                    kernel -= eta * final_gr_kernel

                elif optimizer == "adam":
                    t += 1
                    # update visible bias
                    m_v_bias = beta1 * m_v_bias + (1 - beta1) * final_gr_v_bias
                    v_v_bias = beta2 * v_v_bias + (1 - beta2) * final_gr_v_bias**2
                    m_hat_v_bias = m_v_bias / (1 - beta1**t)
                    v_hat_v_bias = v_v_bias / (1 - beta2**t)
                    v_bias -= eta * m_hat_v_bias / (np.sqrt(v_hat_v_bias) - epsilon)
                    # update hidden bias
                    m_h_bias = beta1 * m_h_bias + (1 - beta1) * final_gr_h_bias
                    v_h_bias = beta2 * v_h_bias + (1 - beta2) * final_gr_h_bias**2
                    m_hat_h_bias = m_h_bias / (1 - beta1**t)
                    v_hat_h_bias = v_h_bias / (1 - beta2**t)
                    h_bias -= eta * m_hat_h_bias / (np.sqrt(v_hat_h_bias) - epsilon)
                    # update kernel
                    m_kernel = beta1 * m_kernel + (1 - beta1) * final_gr_kernel
                    v_kernel = beta2 * v_kernel + (1 - beta2) * final_gr_kernel**2
                    m_hat_kernel = m_kernel / (1 - beta1**t)
                    v_hat_kernel = v_kernel / (1 - beta2**t)
                    kernel -= eta * m_hat_kernel / (np.sqrt(v_hat_kernel) - epsilon)

                energies = []
                grads_v_bias = []
                grads_h_bias = []
                grads_kernel = []
                steps_before_optimize = batch_size
                """
                if early_stopping:
                    v_bias_converged = np.allclose(v_bias,
                                                   v_bias_old,
                                                   rtol=rtol,
                                                   atol=atol)
                    h_bias_converged = np.allclose(h_bias,
                                                   h_bias_old,
                                                   rtol=rtol,
                                                   atol=atol)
                    kernel_converged = np.allclose(kernel,
                                                   kernel_old,
                                                   rtol=rtol,
                                                   atol=atol)

                    if v_bias_converged and h_bias_converged and kernel_converged:
                        did_early_stop = True
                        break
                """

        # early stop flag activated
        if did_early_stop:
            msg = "Early stopping, training converged"
            self.logger.info(msg)
        # msg: Early stopping, training converged

        # end
        # Update shared values
        self.state = state
        self._v_bias = v_bias
        self._h_bias = h_bias
        self._kernel = kernel
        # self.scale = scale
        self._is_trained_ = True

    def tune(
        self,
        tune_iter=20_000,
        tune_interval=500,
        early_stop=False,  # set to True later
        rtol=1e-05,
        atol=1e-08,
        seed=None,
        mcmc_alg=None,
    ):
        """
        BROKEN NOW due to self.scale
        Tune proposal scale so that the acceptance rate is around 0.5.
        """

        self._is_initialized()
        state = self.state
        v_bias = self._v_bias
        h_bias = self._h_bias
        kernel = self._kernel
        scale = self.scale

        if mcmc_alg is not None:
            self._sampler = Sampler(self.mcmc_alg, self.rbm, self.rng, self._log)

        # Used to throw warnings if tuned alg mismatch chosen alg
        # in other procedures
        self._tuned_mcmc_alg = self.mcmc_alg

        # Config
        # did_early_stop = False
        seed_seq = generate_seed_sequence(seed, 1)[0]

        # Reset n_accepted
        state = State(state.positions, state.logp, 0, state.delta)

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
            state = self._sampler.step(state, v_bias, h_bias, kernel, seed_seq)
            steps_before_tune -= 1

            if steps_before_tune == 0:
                # Tune proposal scale
                old_scale = scale
                accept_rate = state.n_accepted / tune_interval
                scale = self._sampler.tune_scale(old_scale, accept_rate)

                # Reset
                steps_before_tune = tune_interval
                state = State(state.positions, state.logp, 0, state.delta)

        # Update shared values
        self.state = state
        self._v_bias = v_bias
        self._h_bias = h_bias
        self._kernel = kernel
        self.scale = scale
        self._is_tuned_ = True

    def sample(self, nsamples, nchains=1, seed=None):
        """helper for the sample method from the Sampler class"""

        self._is_initialized()
        self._is_trained()

        params = {
            "v_bias": self._v_bias,
            "h_bias": self._h_bias,
            "kernel": self._kernel,
        }

        system_info = {
            "nparticles": self._P,
            "dim": self._dim,
            "eta": self._eta,
            "nvisible": self._nvisible,
            "nhidden": self._nhidden,
            "mcmc_alg": self.mcmc_alg,
            "training_cycles": self._training_cycles,
            "training_batch": self._training_batch,
        }

        system_info = pd.DataFrame(system_info, index=[0])
        sample_results = self._sampler.sample(
            self.state, params, nsamples, nchains, seed
        )

        # combine system info and sample results

        self._results = pd.concat([system_info, sample_results], axis=1)

        return self._results

    # @property
    # def scale(self):
    #    return self._sampler.scale

    # @scale.setter
    # def scale(self, value):
    #    self._sampler.scale = value

    @property
    def results(self):
        try:
            return self._results
        except AttributeError:
            msg = "Unavailable, a call to sample must be made first"
            raise errors.SamplingNotPerformed(msg)

    @property
    def energies(self):
        try:
            return self._energies
        except AttributeError:
            msg = "Unavailable, a call to sample must be made first"
            raise errors.SamplingNotPerformed(msg)

    def to_csv(self, filename):
        """Write (full) results dataframe to csv.

        Parameters
        ----------
        filename : str
            Output filename
        """
        self.results.to_csv(filename, index=False)
