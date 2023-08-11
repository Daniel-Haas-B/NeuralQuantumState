import copy
import warnings
from threading import RLock as TRLock

import numpy as np
import pandas as pd
from models import IRBM
from models import JAXIRBM
from models import JAXNIRBM
from models import NIRBM
from numpy.random import default_rng
from pathos.pools import ProcessPool
from tqdm.auto import tqdm
from utils import advance_PRNG_state
from utils import block
from utils import check_and_set_nchains
from utils import early_stopping
from utils import generate_seed_sequence
from utils import setup_logger
from utils import State
from utils import tune_scale_lmh_table
from utils import tune_scale_rwm_table

# import sys
# from abc import abstractmethod
# from functools import partial
# from multiprocessing import Lock
# from multiprocessing import RLock

warnings.filterwarnings("ignore", message="divide by zero encountered")


class NotInitialized(Exception):
    r"""Failed attempt at accessing posterior samples.
    A call to the sample method must be carried out first.
    """
    pass


class NotTrained(Exception):
    r"""Failed attempt at accessing posterior samples.
    A call to the sample method must be carried out first.
    """
    pass


class SamplingNotPerformed(Exception):
    r"""Failed attempt at accessing posterior samples.
    A call to the sample method must be carried out first.
    """
    pass


class NQS:
    def __init__(
        self,
        nparticles,
        dim,
        nhidden=1,
        interaction=False,
        mcmc_alg="rwm",
        nqs_repr="psi2",
        backend="numpy",
        log=True,
        logger_level="INFO",
        rng=None,
    ):
        """Neural Network Quantum State"""

        self._check_logger(log, logger_level)
        self._log = log

        if self._log:
            self.logger = setup_logger(self.__class__.__name__, level=logger_level)
        else:
            self.logger = None

        self._P = nparticles
        self._dim = dim
        self._mcmc_alg = mcmc_alg
        self._nhidden = nhidden
        self._nvisible = self._P * self._dim

        if rng is None:
            rng = default_rng
        self._rng = rng

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
                self._rbm = IRBM(self._P, self._dim, factor=factor)
            else:
                self._rbm = NIRBM(factor=factor)
        elif backend == "jax":
            if interaction:
                self._rbm = JAXIRBM(self._P, self._dim, factor=factor)
            else:
                self._rbm = JAXNIRBM(factor=factor)
        else:
            msg = "Unsupported backend, only 'numpy' or 'jax' is allowed"
            raise ValueError(msg)

        # set mcmc step
        self._set_mcmc_alg(mcmc_alg)

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

    def _set_mcmc_alg(self, mcmc_alg):
        if mcmc_alg == "rwm":
            self._mcmc_alg = mcmc_alg
            self._step = self._rwm_step
            self._tune_table = tune_scale_rwm_table
        elif mcmc_alg == "lmh":
            self._mcmc_alg = mcmc_alg
            self._step = self._lmh_step
            self._tune_table = tune_scale_lmh_table
        else:
            msg = "Unsupported MCMC algorithm, only 'rwm' or 'lmh' is allowed"
            raise ValueError(msg)

    def _is_initialized(self):
        if not self._is_initialized_:
            msg = "A call to 'init' must be made before training"
            raise NotInitialized(msg)

    def _is_trained(self):
        if not self._is_trained_:
            msg = "A call to 'train' must be made before sampling"
            raise NotTrained(msg)

    def _sampling_performed(self):
        if not self._is_trained_:
            msg = "A call to 'sample' must be made in order to access results"
            raise SamplingNotPerformed(msg)

    def _check_logger(self, log, logger_level):
        if not isinstance(log, bool):
            raise TypeError("'log' must be True or False")

        if not isinstance(logger_level, str):
            raise TypeError("'logger_level' must be passed as str")

    def init(self, sigma2=1.0, scale=0.5, seed=None):
        """ """
        self._rbm.sigma2 = sigma2
        self._scale = scale

        rng = self._rng(seed)

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

        logp = self._rbm.logprob(r, self._v_bias, self._h_bias, self._kernel)
        self._state = State(r, logp, 0, 0)

        self._is_initialized_ = True

    def train(
        self,
        max_iter=100_000,
        batch_size=1000,
        gradient_method="adam",
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
        """ """

        self._is_initialized()
        self._training_cycles = max_iter
        self._training_batch = batch_size
        self._eta = eta

        if mcmc_alg is not None:
            self._set_mcmc_alg(mcmc_alg)

        state = self._state
        v_bias = self._v_bias
        h_bias = self._h_bias
        kernel = self._kernel
        scale = self._scale
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

        # Set parameters for Adam
        if gradient_method == "adam":
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
        for i in t_range:
            state = self._step(state, v_bias, h_bias, kernel, seed_seq)
            loc_energy = self._rbm.local_energy(state.positions, v_bias, h_bias, kernel)
            gr_v_bias = self._rbm.grad_v_bias(state.positions, v_bias, h_bias, kernel)
            gr_h_bias = self._rbm.grad_h_bias(state.positions, v_bias, h_bias, kernel)
            gr_kernel = self._rbm.grad_kernel(state.positions, v_bias, h_bias, kernel)
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
                if gradient_method == "gd":
                    v_bias -= eta * final_gr_v_bias
                    h_bias -= eta * final_gr_h_bias
                    kernel -= eta * final_gr_kernel

                elif gradient_method == "adam":
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
        self._state = state
        self._v_bias = v_bias
        self._h_bias = h_bias
        self._kernel = kernel
        self._scale = scale
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
        Tune proposal scale
        """

        self._is_initialized()
        state = self._state
        v_bias = self._v_bias
        h_bias = self._h_bias
        kernel = self._kernel
        scale = self._scale

        if mcmc_alg is not None:
            self._set_mcmc_alg(mcmc_alg)

        # Used to throw warnings if tuned alg mismatch chosen alg
        # in other procedures
        self._tuned_mcmc_alg = self._mcmc_alg

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
            state = self._step(state, v_bias, h_bias, kernel, seed_seq)
            steps_before_tune -= 1

            if steps_before_tune == 0:
                # Tune proposal scale
                old_scale = scale
                accept_rate = state.n_accepted / tune_interval
                scale = self._tune_table(old_scale, accept_rate)

                # Reset
                steps_before_tune = tune_interval
                state = State(state.positions, state.logp, 0, state.delta)

        # Update shared values
        self._state = state
        self._v_bias = v_bias
        self._h_bias = h_bias
        self._kernel = kernel
        self._scale = scale
        self._is_tuned_ = True

    def sample(self, nsamples, nchains=1, seed=None, mcmc_alg=None):
        """ """

        # TODO: accept biases and kernel as parameters and
        # assume they are optimized if passed
        self._is_initialized()
        self._is_trained()
        state = self._state
        v_bias = self._v_bias
        h_bias = self._h_bias
        kernel = self._kernel
        scale = self._scale
        if mcmc_alg is not None:
            self._set_mcmc_alg(mcmc_alg)

        nchains = check_and_set_nchains(nchains, self.logger)
        seeds = generate_seed_sequence(seed, nchains)

        if nchains == 1:
            chain_id = 0
            results, self._energies = self._sample(
                nsamples, state, v_bias, h_bias, kernel, scale, seeds[0], chain_id
            )
            self._results = pd.DataFrame([results])
        else:
            if self._log:
                # for managing output contention
                tqdm.set_lock(TRLock())
                initializer = tqdm.set_lock
                initargs = (tqdm.get_lock(),)
            else:
                initializer = None
                initargs = None

            # Handle iterables
            nsamples = (nsamples,) * nchains
            state = (state,) * nchains
            v_bias = (v_bias,) * nchains
            h_bias = (h_bias,) * nchains
            kernel = (kernel,) * nchains
            scale = (scale,) * nchains
            chain_ids = range(nchains)

            with ProcessPool(
                nchains, initializer=initializer, initargs=initargs
            ) as pool:
                results, self._energies = zip(
                    *pool.map(
                        self._sample,
                        nsamples,
                        state,
                        v_bias,
                        h_bias,
                        kernel,
                        scale,
                        seeds,
                        chain_ids,
                    )
                )
            self._results = pd.DataFrame(results)

        self._sampling_performed_ = True
        if self._log:
            self.logger.info("Sampling done")

        return self._results

    def _sample(self, nsamples, state, v_bias, h_bias, kernel, scale, seed, chain_id):
        """To be called by process"""
        if self._log:
            t_range = tqdm(
                range(nsamples),
                desc=f"[Sampling progress] Chain {chain_id+1}",
                position=chain_id,
                leave=True,
                colour="green",
            )
        else:
            t_range = range(nsamples)

        # Config
        state = State(state.positions, state.logp, 0, state.delta)
        energies = np.zeros(nsamples)

        for i in t_range:
            state = self._step(state, v_bias, h_bias, kernel, seed)
            energies[i] = self._rbm.local_energy(
                state.positions, v_bias, h_bias, kernel
            )
        if self._log:
            t_range.clear()

        energy = np.mean(energies)
        error = block(energies)
        variance = np.mean(energies**2) - energy**2
        acc_rate = state.n_accepted / nsamples

        results = {
            "chain_id": chain_id + 1,
            "nparticles": self._P,
            "dim": self._dim,
            "energy": energy,
            "std_error": error,
            "variance": variance,
            "accept_rate": acc_rate,
            "eta": self._eta,
            "scale": scale,
            "nvisible": self._nvisible,
            "nhidden": self._nhidden,
            "mcmc_alg": self._mcmc_alg,
            "nsamples": nsamples,
            "training_cycles": self._training_cycles,
            "training_batch": self._training_batch,
        }

        return results, energies

    def _rwm_step(self, state, v_bias, h_bias, kernel, seed):
        """One step of the random walk Metropolis algorithm

        Parameters
        ----------
        state : nqs.State
            Current state of the system. See state.py

        scale : float
            Scale of proposal distribution. Default: 0.5

        Returns
        -------
        new_state : nqs.State
            The updated state of the system.
        """

        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)

        # Sample proposal positions, i.e., move walkers
        proposals = rng.normal(loc=state.positions, scale=self._scale)

        # Sample log uniform rvs
        log_unif = np.log(rng.random())

        # Compute proposal log density
        logp_proposal = self._rbm.logprob(proposals, v_bias, h_bias, kernel)

        # Metroplis acceptance criterion
        accept = log_unif < logp_proposal - state.logp

        # If accept is True, yield proposal, otherwise keep old state
        new_positions = proposals if accept else state.positions

        # Create new state
        new_logp = self._rbm.logprob(new_positions, v_bias, h_bias, kernel)
        new_n_accepted = state.n_accepted + accept
        new_delta = state.delta + 1
        new_state = State(new_positions, new_logp, new_n_accepted, new_delta)

        return new_state

    def _lmh_step(self, state, v_bias, h_bias, kernel, seed):
        """One step of the Langevin Metropolis-Hastings algorithm

        Parameters
        ----------
        state : State
            Current state of the system. See state.py
        alpha :
            Variational parameter
        D : float
            Diffusion constant. Default: 0.5
        dt : float
            Scale of proposal distribution. Default: 1.0
        """

        # Precompute
        dt = self._scale**2
        Ddt = 0.5 * dt
        quarterDdt = 1 / (4 * Ddt)
        sys_size = state.positions.shape

        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)

        # Compute drift force at current positions
        F = self._rbm.drift_force(state.positions, v_bias, h_bias, kernel)

        # Sample proposal positions, i.e., move walkers
        proposals = (
            state.positions
            + F * Ddt
            + rng.normal(loc=0, scale=self._scale, size=sys_size)
        )

        # Compute proposal log density
        logp_prop = self._rbm.logprob(proposals, v_bias, h_bias, kernel)

        # Green's function conditioned on proposals
        F_prop = self._rbm.drift_force(proposals, v_bias, h_bias, kernel)
        G_prop = -((state.positions - proposals - Ddt * F_prop) ** 2) * quarterDdt

        # Green's function conditioned on current positions
        G_cur = -((proposals - state.positions - Ddt * F) ** 2) * quarterDdt

        # Metroplis-Hastings ratio
        ratio = logp_prop + np.sum(G_prop) - state.logp - np.sum(G_cur)

        # Sample log uniform rvs
        log_unif = np.log(rng.random())

        # Metroplis acceptance criterion
        accept = log_unif < ratio

        # If accept is True, yield proposal, otherwise keep old state
        new_positions = proposals if accept else state.positions

        # Create new state
        new_logp = self._rbm.logprob(new_positions, v_bias, h_bias, kernel)
        new_n_accepted = state.n_accepted + accept
        new_delta = state.delta + 1
        new_state = State(new_positions, new_logp, new_n_accepted, new_delta)

        return new_state

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    @property
    def results(self):
        try:
            return self._results
        except AttributeError:
            msg = "Unavailable, a call to sample must be made first"
            raise SamplingNotPerformed(msg)

    @property
    def energies(self):
        try:
            return self._energies
        except AttributeError:
            msg = "Unavailable, a call to sample must be made first"
            raise SamplingNotPerformed(msg)

    def to_csv(self, filename):
        """Write (full) results dataframe to csv.

        Parameters
        ----------
        filename : str
            Output filename
        """
        self.results.to_csv(filename, index=False)
