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
from nqs.utils import wf_factory
from nqs.utils import tune_sampler
import jax
import time
import jax.numpy as jnp

from nqs import pretrain


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


import numpy as np
import pandas as pd

# from nqs.models import RBM, FFNN, VMC, Dummy

from numpy.random import default_rng
from tqdm.auto import tqdm

from physics.hamiltonians import HarmonicOscillator as HO

# sys.path.insert(0, "../samplers/")

from samplers.metropolis_hastings import MetroHastings
from samplers.metropolis import Metropolis as Metro

import optimizers as opt

# import sys
# from abc import abstractmethod
# from functools import partial
# from multiprocessing import Lock
# from multiprocessing import RLock

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

warnings.filterwarnings("ignore", message="divide by zero encountered")


class GNQS:
    def __init__(
        self,
        backend="jax",
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

        # flags
        self._is_initialized_ = False
        self._is_trained_ = False
        self._is_tuned_ = False
        self._sampling_performed

    def generate_ext(self, r):  # r will be noise
        """
        to be called when training the discriminator
        """
        return self.wf(r, self.wf.params)

    def generate_int(self, r, params):
        """
        to be called when training the generator
        """
        return self.wf(r, params)

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
        self.param_keys = self.wf.params.keys()

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

        match optimizer.lower():
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
        self._history.update({key: [] for key in param_keys})
        seed_seq = generate_seed_sequence(self._seed, 1)[0]

        energies = []
        final_grads = {key: None for key in param_keys}

        expval_energies_dict = {key: None for key in param_keys}
        expval_grad_dict = {key: None for key in param_keys}
        # steps_before_optimize = batch_size

        # state = self.wf.state
        # state = State(state.positions, state.logp, 0, state.delta)
        # states = state.create_batch_of_states(batch_size=batch_size)

        # equilibrate, burn in lets see if makes a difference
        # for _ in range(1000):
        #    state = self._sampler.step(self.wf, state, seed_seq)

        grads_dict = {key: [] for key in param_keys}
        epoch = 0
        for _ in t_range:
            # this object contains the states of all the sequence of steps
            # cleans up the state after each batch
            state = self.wf.state
            state = State(state.positions, state.logp, 0, state.delta)

            states = state.create_batch_of_states(batch_size=batch_size)
            states = self._sampler.step(
                self.wf, states, seed_seq, batch_size=batch_size
            )

            energies = self.hamiltonian.local_energy(self.wf, states.positions)
            local_grads_dict = self.wf.grads(states.positions)

            epoch += 1
            energies = np.array(energies)
            expval_energy = np.mean(energies)
            t_range.set_postfix(avg_E_l=f"{expval_energy:.2f}", refresh=True)

            for key in param_keys:
                grad_np = np.array(local_grads_dict.get(key))
                grads_dict[key] = grad_np
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
            if self._history:
                grad_norms = [np.linalg.norm(final_grads[key]) for key in param_keys]
                grad_norms = np.mean(grad_norms)
                self._history["energy"].append(expval_energy)
                self._history["grads"].append(grad_norms)

            # Descent
            self._optimizer.step(
                self.wf.params, final_grads, self.sr_matrices
            )  # changes wf params inplace

            #     self._agent.log(
            #         {"abs(energy - 3)": np.abs(expval_energy - 3)},
            #         epoch,  # change this if not 2 particles 2 dimensions
            #     ) if self._agent else None
            #     self._agent.log({"grads": grad_norms}, epoch) if self._agent else None
            # self._agent.log(
            #     {"scale": self.scale}, epoch
            # ) if self._agent else None

            if grad_norms < 10**-15:
                if self.logger is not None:
                    self.logger.warning("Gradient norm is zero, stopping training")
                break

            if self._tune and epoch % int(max_iter * 0.1) == 0:
                tune_batch = batch_size  # int(batch_size*0.1) # needs to be big else does not stabilize the acceptance rate
                tune_iter = int(max_iter * 0.2)  # max_iter # int(max_iter*0.5)

                # this will be very inneficient now
                tune_sampler(
                    wf=self.wf,
                    sampler=self._sampler,
                    seed=self._seed,
                    log=self._log,
                    tune_batch=tune_batch,
                    tune_iter=tune_iter,
                    mode="standard",
                    logger=self.logger,
                )
                self._is_tuned_ = True

            # update wf state after each epoch?
            # self.wf.state = states[-1] ## check this!

            final_grads = {key: None for key in param_keys}
            grads_dict = {key: [] for key in param_keys}

        self.state = State(states[-1].positions, states[-1].logp, 0, states[-1].delta)

        self._is_trained_ = True

        if self.logger is not None:
            self.logger.info("Training done")

        if self._history:
            return self._history

    def sample(self, nsamples, nchains=1, seed=None, one_body_density=False):
        """helper for the sample method from the Sampler class"""
        t0 = time.time()
        self._is_initialized()
        self._is_trained()
        self.hamiltonian.turn_reg_off()

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
            "Opti": self._optimizer.__class__.__name__,
            "symmetry": self._symmetry,
        }

        system_info = pd.DataFrame(system_info, index=[0])

        sample_results = self._sampler.sample(
            self.wf, self.state, nsamples, nchains, seed
        )
        system_info_repeated = system_info.loc[
            system_info.index.repeat(len(sample_results))
        ].reset_index(drop=True)

        self._results = pd.concat([system_info_repeated, sample_results], axis=1)
        t1 = time.time()
        print("Sampling time: ", t1 - t0)
        return self._results

        return sample_results

    def pretrain(self, model, max_iter, batch_size, **kwargs):
        """
        # TODO: make this less repetitive
        """
        if model.lower() == "gaussian":
            pre_system = pretrain.Gaussian(
                log=True,
                logger_level="INFO",
                seed=self._seed * 2,
                symmetry=self._symmetry,
            )

            pre_system.set_wf(  # TODO: MAKE LESS REPETITIVE
                self.wf.__class__.__name__,
                self._N,
                self._dim,
                layer_sizes=self.wf._layer_sizes,
                activations=self.wf._activations,
                symmetry="none",
            )
        else:
            raise NotImplementedError("Only Gaussian pretraining is supported for now")

        # FOR NOW DOES NOT MAKE SENSE
        # pre_system.set_sampler(mcmc_alg=mcmc_alg, scale=1)

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
            pretrain_sampler=False,
        )
        self.wf.params = params

    def one_vmc_sample(self, params, batch_size, seed_seq):
        state = self.wf.state
        state = State(
            state.positions, state.logp, 0, state.delta
        )  # not sure here if we should reset the naccepted. Using state.delta might make this functionally pure

        states = state.create_batch_of_states(batch_size=batch_size)
        # print("states", states)
        states = self._sampler.step(
            self.wf, states, seed_seq, batch_size=batch_size, params=params
        )
        self.state = states  # to be used in the one_vmc_grads
        return states

    def one_vmc_grads(self, states):
        energies = self.hamiltonian.local_energy(self.wf, states.positions)
        local_grads_dict = self.wf.grads(states.positions)

        energies = np.array(energies)
        expval_energy = np.mean(energies)
        param_keys = self.param_keys
        batch_size = len(states.positions)

        assert batch_size == 500

        grads_dict = {key: [] for key in param_keys}
        grads_vmc = {key: None for key in param_keys}
        expval_energies_dict = {key: None for key in param_keys}
        expval_grad_dict = {key: None for key in param_keys}
        for key in param_keys:
            grad_np = np.array(local_grads_dict.get(key))
            grads_dict[key] = grad_np
            self._history[key].append(np.linalg.norm(grad_np))
            new_shape = (batch_size,) + (1,) * (
                grad_np.ndim - 1
            )  # Subtracting 1 because the first dimension is already provided by batch_size
            energies = energies.reshape(new_shape)

            expval_energies_dict[key] = np.mean(energies * grad_np, axis=0)
            expval_grad_dict[key] = np.mean(grad_np, axis=0)

            grads_vmc[key] = 2 * (
                expval_energies_dict[key] - expval_energy * expval_grad_dict[key]
            )

        if self._optimizer.__class__.__name__ == "Sr":
            self.sr_matrices = self.wf.compute_sr_matrix(expval_grad_dict, grads_dict)

            return grads_vmc, self.sr_matrices

        return grads_vmc

    def gen_loss_fn(self, discriminator, params, noise):  # , batch_size, seed_seq):
        # this should NOT evolve the rng when called or else it will not be functionally pure
        # Let us first make it work and then correct the purity

        batch_gen = self.generate(noise, params)
        D_G_z = discriminator.predict_probas_ext(batch_gen)
        gan_loss = jnp.mean(jnp.log(D_G_z))

        return gan_loss

    # WIP

    def gen_grads(self, noise, params):
        # will return the physics loss just like in the vmc case but without vmc

        states = self.generate_int(
            noise, params
        )  # we know this will be the same as in the GAN loss

        energies = self.hamiltonian.local_energy(self.wf, states)
        print("energies", np.mean(energies))
        local_grads_dict = self.wf.grads(states)

        energies = np.array(energies)
        expval_energy = np.mean(energies)
        param_keys = self.param_keys
        batch_size = len(states)

        grads_dict = {key: [] for key in param_keys}
        grads_gen = {key: None for key in param_keys}
        expval_energies_dict = {key: None for key in param_keys}
        expval_grad_dict = {key: None for key in param_keys}
        for key in param_keys:
            grad_np = np.array(local_grads_dict.get(key))
            grads_dict[key] = grad_np
            # self._history[key].append(np.linalg.norm(grad_np))
            new_shape = (batch_size,) + (1,) * (
                grad_np.ndim - 1
            )  # Subtracting 1 because the first dimension is already provided by batch_size
            energies = energies.reshape(new_shape)

            expval_energies_dict[key] = np.mean(energies * grad_np, axis=0)
            expval_grad_dict[key] = np.mean(grad_np, axis=0)

            grads_gen[key] = 2 * (
                expval_energies_dict[key] - expval_energy * expval_grad_dict[key]
            )

        if self._optimizer.__class__.__name__ == "Sr":
            self.sr_matrices = self.wf.compute_sr_matrix(expval_grad_dict, grads_dict)

        return grads_gen

    # WIP

    def g_step_train(self, discriminator, noise):  # batch_size, seed_seq):
        """
        Shall be called epoch times

        update the generator by asciending its stochastic gradient
            ∇_{\theta_g} 1/m sum_{i=1}^m log(D(G(z_i)))
        IMportant: this gen_sampl is not the same as the one in the discriminator training
        """
        params = self.wf.params
        grad_loss_fn = jax.grad(self.gen_loss_fn, argnums=1)

        grad_loss_dict_gan = grad_loss_fn(discriminator, params, noise)

        # get the batch_gen_state created in gen_loss_fn
        ################
        # this is only if generator is vmc
        # grad_loss_dict_vmc, sr_matrices = self.one_vmc_grads(self.state) # ensure this state is the same as batch_gen in the gen_loss_fn
        #
        # self._optimizer.step(self.net.params, grad_loss_dict_gan, None)
        # self._optimizer.step(self.net.params, grad_loss_dict_vmc, sr_matrices)
        ################

        # let us try first without sr_matrices

        # grad_loss_phys = self.gen_grads(noise, params)

        self._optimizer.step(
            self.wf.params, -grad_loss_dict_gan, None
        )  # minus because we are ascending the gradient
        # self._optimizer.step(self.wf.params, grad_loss_phys, self.sr_matrices)

        return self.wf.params, self.gen_loss_fn(discriminator, params, noise)


class DNQS:
    def __init__(
        self,
        backend="numpy",
        batch_size=500,
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
        self.batch_size = batch_size
        self.nqs_type = None
        self._backend = backend
        self._optimizer = None
        self.net = None
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

    def set_net(self, wf_type, nparticles, dim, **kwargs):
        """
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
        self.net = wf_factory(wf_type, **common_args, **specific_args)
        self.nqs_type = wf_type
        self._is_initialized_ = True

    def set_optimizer(self, optimizer, eta, **kwargs):
        """
        Set the optimizer algorithm to be used for param update.
        # TODO: make this less repetitive
        """
        self._eta = eta

        if not isinstance(optimizer, str):
            raise TypeError("'optimizer' must be passed as str")

        match optimizer.lower():
            case "gd":
                gamma = kwargs["gamma"] if "gamma" in kwargs else 0
                self._optimizer = opt.Gd(self.wf.params, eta, gamma)  # dumb for now
            case "adam":
                beta1 = kwargs["beta1"] if "beta1" in kwargs else 0.9
                beta2 = kwargs["beta2"] if "beta2" in kwargs else 0.999
                epsilon = kwargs["epsilon"] if "epsilon" in kwargs else 1e-8
                self._optimizer = opt.Adam(
                    self.net.params, eta, beta1=beta1, beta2=beta2, epsilon=epsilon
                )  # _params gets passed to construct the mom and v arrays
            case "rmsprop":
                beta = kwargs["beta1"] if "beta1" in kwargs else 0.9
                epsilon = kwargs["epsilon"] if "epsilon" in kwargs else 1e-8

                self._optimizer = opt.RmsProp(
                    self.net.params, eta, beta=beta, epsilon=epsilon
                )

            case "adagrad":
                self._optimizer = opt.Adagrad(self.net.params, eta)
            case "sr":
                self._optimizer = opt.Sr(self.net.params, eta)
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

    def predict_probas_ext(self, data):
        """
        TO BE CALLED ONLY OUTSIDE DNQS
        """
        # soft classification
        return self.net(data, self.net.params)

    def predict_probas(self, data, params):
        """
        should output batchsize of probas
        TO BE CALLED ONLY INSIDE DNQS
        """
        # soft classification
        return self.net(data, params)

    def log_loss(self, y_true, batch, params):
        """
        y_true: array of true labels
        y_pred: array of predicted labels
        """
        y_pred = self.predict_probas(batch, params)
        return jnp.mean(
            y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred)
        )  # average over batch

    def train(self, train_data, train_labels, max_iter, batch_size, **kwargs):
        """
        Train the wave function parameters. but for just normal classification problem
        """
        self._is_initialized()
        losses = []
        data_size = len(train_data)
        for epoch in range(max_iter):
            for i in range(0, data_size, batch_size):
                batch = train_data[i : i + batch_size]

                reality = train_labels[i : i + batch_size]
                params = self.net.params
                loss_func = self.log_loss

                loss = loss_func(reality, batch, params)
                losses.append(loss)
                # print("discriminator loss", loss)
                grad_loss_fn = jax.grad(loss_func, argnums=2)
                grad_loss_dict = grad_loss_fn(reality, batch, params)

                self._optimizer.step(self.net.params, grad_loss_dict, None)

        self._history = losses

        self._is_trained_ = True

        if self.logger is not None:
            self.logger.info("Discriminator Training done")

        if self._history:
            return self._history

    def disc_loss_fn(self, batch_data, batch_gen, params):
        # D(x_i) is D(data_samples)
        log_D_x_i = jnp.log(self.predict_probas(batch_data, params))

        # D(G(z_i)) is D(gen_samples)
        log_1_minus_D_G_z_i = jnp.log(1 - self.predict_probas(batch_gen, params))

        return jnp.mean(log_D_x_i + log_1_minus_D_G_z_i)

    def d_step_train(self, data_sampl_batch, gen_sampl_batch):
        """
        Shall be called epoch times
        This shall be a one step train for the GAN, meaning that the optimization will be the ascend on
        ∇_{\theta_d} 1/m sum_{i=1}^m [log D(x_i) + log(1 - D(G(z_i)))]

        args:
        gen_samples: batchsize samples from the generator at that point
        data_samples: batchsize samples from the data distribution

        train data comes with both pdata and pgen instances
        """

        params = self.net.params
        # print("discriminator loss", loss_func(batch_data, batch_gen, params))
        grad_loss_fn = jax.grad(self.disc_loss_fn, argnums=2)
        grad_loss_dict = grad_loss_fn(data_sampl_batch, gen_sampl_batch, params)

        loss = self.disc_loss_fn(data_sampl_batch, gen_sampl_batch, params)

        self._optimizer.step(
            self.net.params, -grad_loss_dict, None
        )  # MINUS because we want to ascend

        return self.net.params, loss


class ANQS:
    def __init__(self, gen, disc):
        self.gen = gen
        self.disc = disc

    def set_truth(self, truth_generator):
        """
        truth generator should be for example a trained vmc, or a gaussian
        """
        self.truth = truth_generator

    def train(self, max_iter, batch_size, discriminator_updates, **kwargs):
        """
        train the generator and discriminator in an adversarial way
        for number of training cycles do
            for number of discriminator updates do
                sample minibatch of m noise samples {z_1, ..., z_m} from noise prior p_g(z)
                sample minibatch of m examples {x_1, ..., x_m} from data generating distribution p_data(x)
                update the discriminator by ascending its stochastic gradient
                    ∇_{\theta_d} 1/m sum_{i=1}^m [log D(x_i) + log(1 - D(G(z_i)))]
            end for
            sample minibatch of m noise samples {z_1, ..., z_m} from noise prior p_g(z)

            update the generator by asciending its stochastic gradient
                ∇_{\theta_g} 1/m sum_{i=1}^m log(D(G(z_i)))

            (or minimize the log(1 - D(G(z_i)))
        """
        particles = self.gen._N
        dim = self.gen._dim
        seed = 42

        t_range = tqdm(
            range(max_iter),
            desc="[Training progress]",
            position=0,
            leave=True,
            colour="green",
        )
        loss_g_lst = []
        loss_d_lst = []
        for _ in t_range:
            # gen_sampl = self.gen.one_vmc_sample(self.gen.wf.params, data_size, seed_seq) # here it is ok to advance rng
            loss_d = 0
            for _ in range(discriminator_updates):
                data_sampl = self.truth.sample_pos(batch_size, seed)

                noise = np.random.normal(size=(batch_size, particles * dim))
                gen_sampl = self.gen.generate_ext(
                    noise
                )  # just a forward pass without any training. Need to ensure that parameters will be correct tho

                self.disc.net.params, loss = self.disc.d_step_train(
                    data_sampl, gen_sampl
                )
                loss_d += loss

            loss_d_lst.append(np.mean(loss_d))

            # gen_sampl is done inside the g_step_train
            noise = np.random.normal(size=(batch_size, particles * dim))
            self.gen.wf.params, loss_g_i = self.gen.g_step_train(
                self.disc, noise
            )  # here it is NOT ok to advance rng
            loss_g_lst.append(loss_g_i)

        return loss_g_lst, loss_d_lst

    def test_gen(self, samples, particles, dim, **kwargs):
        """
        generate fake samples and real ones to plot and see if they are similar
        """
        noise = np.random.normal(size=(samples, particles * dim))
        gen_sampl = self.gen.generate_ext(noise)

        data_sampl = np.random.normal(
            size=(samples, particles * dim)
        )  # this will follow p_data if we are using the gaussian truth

        return gen_sampl, data_sampl

    def test_disc(self, samples, particles, dim, **kwargs):
        # show some accuracy for the discriminator
        noise = np.random.normal(size=(samples, particles * dim))
        gen_sampl = self.gen.generate_ext(noise)
        gen_sampl_probas = self.disc.predict_probas_ext(gen_sampl)

        data_sampl = self.truth.sample_pos(samples, 42)
        data_sampl_probas = self.disc.predict_probas_ext(data_sampl)

        return gen_sampl_probas, data_sampl_probas
