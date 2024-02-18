# import copy
import sys
import warnings

sys.path.insert(0, "../src/")

from nqs.utils import errors
from nqs.utils import generate_seed_sequence
from nqs.utils import setup_logger
from nqs.utils import State
from nqs.utils import wf_factory
import jax
from nqs.utils import advance_PRNG_state

from sklearn.metrics import mean_squared_error as mse


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


import numpy as np


# from nqs.models import RBM, FFNN, VMC, Dummy

from numpy.random import default_rng
from tqdm.auto import tqdm

# from physics.hamiltonians import HarmonicOscillator as HO

# sys.path.insert(0, "../samplers/")

from samplers.metropolis_hastings import MetroHastings
from samplers.metropolis import Metropolis as Metro

import optimizers as opt

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

warnings.filterwarnings("ignore", message="divide by zero encountered")
import jax.numpy as jnp


class Gaussian:
    def __init__(
        self,
        log=True,
        logger_level="INFO",
        rng=None,
        seed=None,
    ):
        """Neural Network Quantum State
        It is conceptually important to understand that this is the system.
        The system is composed of a wave function, a hamiltonian, a sampler and an optimizer.
        This is the high level class that ties all the other classes together.
        """

        self._check_logger(log, logger_level)
        self._log = log

        self.nqs_type = None
        self.hamiltonian = None
        self.backend = np  # jnp
        self.la = np.linalg  # jnp.linalg
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

        # flags
        self._is_initialized_ = False
        self._is_trained_ = False
        self._is_tuned_ = False

    def set_wf(self, wf_type, nparticles, dim, **kwargs):
        """
        Set the wave function to be used for sampling.
        Successfully setting the wave function will also initialize it.
        """
        self._N = nparticles
        self._dim = dim
        common_args = {
            "nparticles": self._N,
            "dim": self._dim,
            "rng": self.rng(self._seed) if self.rng else np.random.default_rng(),
            "log": self._log,
            "logger": self.logger,
            "logger_level": "INFO",
            "backend": "jax",  # make dynamic
        }
        specific_args = kwargs
        self.wf = wf_factory(wf_type, **common_args, **specific_args)
        self._is_initialized_ = True

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

        if not isinstance(optimizer, str):
            raise TypeError("'optimizer' must be passed as str")

        match optimizer:
            case "gd":
                gamma = kwargs["gamma"] if "gamma" in kwargs else 0
                self._optimizer = opt.Gd(self.wf.params, eta, gamma)  # dumb for now
            case "adam":
                beta1 = kwargs["beta1"] if "beta1" in kwargs else 0.9
                beta2 = kwargs["beta2"] if "beta2" in kwargs else 0.999
                epsilon = kwargs["epsilon"] if "epsilon" in kwargs else 1e-8
                self._optimizer = opt.Adam(
                    self.wf.params, eta, beta1=beta1, beta2=beta2, epsilon=epsilon
                )  # _params gets passed to construct the mom and v arrays
            case "rmsprop":
                beta = kwargs["beta1"] if "beta1" in kwargs else 0.9
                epsilon = kwargs["epsilon"] if "epsilon" in kwargs else 1e-8

                self._optimizer = opt.RmsProp(
                    self.wf.params, eta, beta=beta, epsilon=epsilon
                )

            case "adagrad":
                self._optimizer = opt.Adagrad(self.wf.params, eta)
            case "sr":
                self._optimizer = opt.Sr(self.wf.params, eta)
            case _:  # noqa
                msg = "Unsupported optimizer. Choose between: \n"
                msg += "    gd, adam, rmsprop, adagrad, sr"
                raise ValueError(msg)

    def _is_initialized(self):
        if not self._is_initialized_:
            msg = "A call to 'init' must be made before training"
            raise errors.NotInitialized(msg)

    def _is_trained(self):
        if not self._is_trained_:
            msg = "A call to 'train' must be made before sampling"
            raise errors.NotTrained(msg)

    def _check_logger(self, log, logger_level):
        if not isinstance(log, bool):
            raise TypeError("'log' must be True or False")

        if not isinstance(logger_level, str):
            raise TypeError("'logger_level' must be passed as str")

    def pretrain(self, max_iter, batch_size, **kwargs):
        """
        Train the wave function parameters.
        """
        self._is_initialized()
        self._training_cycles = max_iter
        self._training_batch = batch_size
        self._history = (
            {"loss": [], "grads": []} if kwargs.get("history", False) else None
        )

        self._early_stop = kwargs.get("early_stop", False)
        self._grad_clip = kwargs.get("grad_clip", False)
        self.pretrain_sampler = kwargs.get("pretrain_sampler", False)
        if self._log:
            t_range = tqdm(
                range(max_iter),
                desc="[Pre-training progress]",
                position=0,
                leave=True,
                colour="blue",
            )
        else:
            t_range = range(max_iter)

        params = self.wf.params
        param_keys = params.keys()
        self._history.update({key: [] for key in param_keys}) if self._history else None
        seed_seq = generate_seed_sequence(self._seed, 1)[0]

        epoch = 0
        loss_func = lambda x, param: self.jaxmse(  # noqa
            self.wf.logprob_closure(x, param), self.multivar_gaussian_pdf(x)
        )
        for _ in t_range:
            state = self.wf.state
            state = State(state.positions, state.logp, 0, state.delta)
            states = state.create_batch_of_states(batch_size=batch_size)
            if self.pretrain_sampler:
                raise NotImplementedError("Pretrain sampler not implemented yet")
                # sample from the wf object which is now (supposed to be) a gaussian
                states = self._sampler.step(
                    self.wf, states, seed_seq, batch_size=batch_size
                )
                loss = mse(
                    self.wf(states.positions),
                    self.multivar_gaussian_pdf(
                        states.positions, mean=self.backend.zeros(self._dim * self._N)
                    ),
                )

            else:
                # generate uniform random numbers and regress them to the gaussian
                # Advance RNG
                next_gen = advance_PRNG_state(self._seed, epoch)
                rng = self.rng(next_gen)
                mean = self.backend.zeros(self._dim * self._N)
                twopi = 2 * self.backend.pi
                states.positions = rng.normal(
                    loc=0, scale=1, size=(batch_size, self._dim * self._N)
                )
                twopi ** len(mean)
                # rng.uniform(-5, 5, (batch_size, self._dim * self._N))

                # print(mse(self.wf.logprob_closure(states.positions, self.wf.params), self.multivar_gaussian_pdf(states.positions, mean)))
                loss = loss_func(states.positions, self.wf.params)

            grad_loss_fn = jax.grad(loss_func, argnums=1)
            grad_loss_dict = grad_loss_fn(states.positions, params)

            epoch += 1
            t_range.set_postfix(loss=f"{loss:.2f}", refresh=True)

            if self._history:
                grad_norms = [
                    self.backend.mean(grad_loss_dict.get(key)) for key in param_keys
                ]
                self._history["loss"].append(loss)
                self._history["grads"].append(grad_norms)

            # Descent
            self._optimizer.step(
                self.wf.params, grad_loss_dict, self.sr_matrices
            )  # changes wf params inplace

            if loss < 10**-15:
                if self.logger is not None:
                    self.logger.warning("loss is zero, stopping training")
                break

            # update wf state after each epoch?
            # self.wf.state = states[-1] ## check this!

        self.state = State(states[-1].positions, states[-1].logp, 0, states[-1].delta)
        self._is_trained_ = True

        if self.logger is not None:
            self.logger.info("Pre-training done")

        if self._history:
            return self.wf.params, self._history

        return self.wf.params

    def multivar_gaussian_pdf(self, x):
        """
        Given an input, it outputs the probability density function of a multivariate Gaussian.

        input x: np.array
        input mean: np.array

        output: float
        """
        mean = self.backend.zeros(self._dim * self._N)
        covariance = self.backend.eye(len(mean))
        x_minus_mean = x - mean
        inv_cov = self.la.inv(covariance)

        det_cov = self.la.det(covariance)
        twopi = 2 * self.backend.pi

        incov_at_xmm = self.backend.einsum("ij,nj->ni", inv_cov, x_minus_mean)
        multivar_array = -0.5 * self.backend.einsum(
            "jn,nj->n", x_minus_mean.T, incov_at_xmm
        )

        return multivar_array - jnp.log(jnp.sqrt(twopi ** len(mean) * det_cov))

    def jaxmse(self, x, y):
        return self.backend.mean((x - y) ** 2)
