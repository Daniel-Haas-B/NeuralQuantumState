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
        nparticles,
        dim,
        nhidden=1,
        interaction=False,
        nqs_repr="psi2",
        backend="numpy",
        log=True,
        logger_level="INFO",
        rng=None,
        use_sr=False,
        nqs_type=None,
    ):
        """Neural Network Quantum State"""

        self._check_logger(log, logger_level)
        self._log = log
        self.mcmc_alg = None
        self._optimizer = None
        self.sr_matrix = None
        self.use_sr = use_sr
        self.nqs_type = nqs_type

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
            self._optimizer = gd.Gd(self._params, eta)  # dumb for now
        elif optimizer == "adam":
            beta1 = kwargs["beta1"] if "beta1" in kwargs else 0.9
            beta2 = kwargs["beta2"] if "beta2" in kwargs else 0.999
            epsilon = kwargs["epsilon"] if "epsilon" in kwargs else 1e-8
            self._optimizer = adam.Adam(
                self._params, eta, beta1=beta1, beta2=beta2, epsilon=epsilon
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

    def sample(self, nsamples, nchains=1, seed=None):
        """helper for the sample method from the Sampler class"""

        self._is_initialized()
        self._is_trained()

        system_info = {
            "nparticles": self._P,
            "dim": self._dim,
            "eta": self._eta,
            "nvisible": self._nvisible,
            "nhidden": self._nhidden,
            "mcmc_alg": self.mcmc_alg,
            "nqs_type": self.nqs_type,
            "training_cycles": self._training_cycles,
            "training_batch": self._training_batch,
            "sr": self.use_sr,
        }

        system_info = pd.DataFrame(system_info, index=[0])
        sample_results = self._sampler.sample(
            self.state, self._params, nsamples, nchains, seed
        )
        system_info_repeated = system_info.loc[
            system_info.index.repeat(len(sample_results))
        ].reset_index(drop=True)

        self._results = pd.concat([system_info_repeated, sample_results], axis=1)

        return self._results


class RBM(NQS):
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
        use_sr=False,
    ):
        """RBM Neural Network Quantum State"""
        super().__init__(
            nparticles,
            dim,
            nhidden,
            interaction,
            nqs_repr,
            backend,
            log,
            logger_level,
            rng,
            use_sr,
            nqs_type="rbm",
        )

        self._nvisible = self._P * self._dim

        if backend == "numpy":
            if interaction:
                self.rbm = IRBM(self._P, self._dim, factor=self.factor)
            else:
                self.rbm = NIRBM(factor=self.factor)
        elif backend == "jax":
            if interaction:
                self.rbm = JAXIRBM(self._P, self._dim, factor=self.factor)
            else:
                self.rbm = JAXNIRBM(factor=self.factor)
        else:
            msg = "Unsupported backend, only 'numpy' or 'jax' is allowed"
            raise ValueError(msg)

        if self._log:
            neuron_str = "neurons" if self._nhidden > 1 else "neuron"
            msg = (
                f"Neural Network Quantum State initialized as RBM with "
                f"{self._nhidden} hidden {neuron_str}"
            )
            self.logger.info(msg)

    def init(self, sigma2=1.0, seed=None):
        """ """
        self.rbm.sigma2 = sigma2

        rng = self.rng(seed)

        r = rng.standard_normal(size=self._nvisible)

        # Initialize visible bias
        v_bias = rng.standard_normal(size=self._nvisible) * 0.01
        h_bias = rng.standard_normal(size=self._nhidden) * 0.01
        kernel = rng.standard_normal(size=(self._nvisible, self._nhidden))
        kernel *= np.sqrt(1 / self._nvisible)

        self._params = Parameter()
        self._params.set(["v_bias", "h_bias", "kernel"], [v_bias, h_bias, kernel])

        # make logprob deal with the generator later instead of indexing
        logp = self.rbm.logprob(
            r,
            self._params.get(["v_bias"]),
            self._params.get(["h_bias"]),
            self._params.get(["kernel"]),
        )
        self.state = State(r, logp, 0, 0)
        self._is_initialized_ = True

    def train(
        self,
        max_iter=100_000,
        batch_size=1000,
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

        if mcmc_alg is not None:
            self._sampler = Sampler(
                self.mcmc_alg, self.rbm, self.rng, logger=self.logger
            )

        state = self.state
        v_bias, h_bias, kernel = self._params.get(["v_bias", "h_bias", "kernel"])

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
            # sampler step
            state = self._sampler.step(state, v_bias, h_bias, kernel, seed_seq)

            # getting and saving local energy
            # print("state.positions", state.positions.shape)

            loc_energy = self.rbm.local_energy(state.positions, v_bias, h_bias, kernel)
            energies.append(loc_energy)

            # getting and saving gradients
            gr_v_bias, gr_h_bias, gr_kernel = self.rbm.grads(
                state.positions, v_bias, h_bias, kernel
            )

            grads_v_bias.append(gr_v_bias)
            grads_h_bias.append(gr_h_bias)
            grads_kernel.append(gr_kernel)

            steps_before_optimize -= 1
            # optimize
            if steps_before_optimize == 0:
                # Expectation values
                energies = np.array(energies)
                grads_v_bias = np.array(grads_v_bias)
                grads_h_bias = np.array(grads_h_bias)
                grads_kernel = np.array(grads_kernel)

                expval_energy = np.mean(energies)
                expval_grad_v_bias = np.mean(grads_v_bias, axis=0)
                expval_grad_h_bias = np.mean(grads_h_bias, axis=0)
                expval_grad_kernel = np.mean(
                    grads_kernel, axis=0
                )  # we shall use this in SR. I think this is avg dlogpsi/dW

                if self.use_sr:
                    self.sr_matrix = self.rbm.compute_sr_matrix(
                        expval_grad_kernel, grads_kernel
                    )

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
                expval_energies = [
                    expval_energy_v_bias,
                    expval_energy_h_bias,
                    expval_energy_kernel,
                ]
                expval_grads = [
                    expval_grad_v_bias,
                    expval_grad_h_bias,
                    expval_grad_kernel,
                ]

                final_grads = [
                    2 * (expval_energy_param - expval_energy * expval_grad_param)
                    for expval_energy_param, expval_grad_param in zip(
                        expval_energies, expval_grads
                    )
                ]

                if early_stopping:
                    # make copies of current values before update
                    v_bias_old = copy.deepcopy(v_bias)  # noqa
                    h_bias_old = copy.deepcopy(h_bias)  # noqa
                    kernel_old = copy.deepcopy(kernel)  # noqa

                # Descent
                params = self._optimizer.step(self._params, final_grads, self.sr_matrix)
                v_bias, h_bias, kernel = params.get(["v_bias", "h_bias", "kernel"])

                energies = []
                grads_v_bias = []
                grads_h_bias = []
                grads_kernel = []
                steps_before_optimize = batch_size

        # early stop flag activated
        if did_early_stop:
            msg = "Early stopping, training converged"
            self.logger.info(msg)
        # msg: Early stopping, training converged

        # end
        # Update shared values. we separate them before because we want to keep the old values
        self.state = state
        self._params.set(["v_bias", "h_bias", "kernel"], [v_bias, h_bias, kernel])
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
        v_bias, h_bias, kernel = self._params.get(["v_bias", "h_bias", "kernel"])

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
        self._params.set(["v_bias", "h_bias", "kernel"], [v_bias, h_bias, kernel])
        self.scale = scale
        self._is_tuned_ = True


class Parameter:
    def __init__(self) -> None:
        self.data = {}

    def set(self, names, values):
        for key, value in zip(names, values):
            self.data[key] = value

    def get(self, names):
        # note this can be a list of names
        return [self.data[name] for name in names]

    def keys(self):
        return self.data.keys()
