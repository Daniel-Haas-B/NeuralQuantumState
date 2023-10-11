# from abc import abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# from jax import grad
# from jax import jit
# from jax import lax
# from jax import vmap
# from scipy.special import expit

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class FFNN:
    """
    WIP
    Base class for creating a quantum system where the wave function is
    represented by a FFNN.
    There will be no analytical expressions for the derivatives, so we
    should use numerical differentiation instead.
    The implementation assumes a logarithmic wave function.
    """

    def __init__(self, sigma2=1.0, factor=0.5):
        self._sigma2 = sigma2
        self._factor = factor
        self._ffnn_psi_repr = 2 * self._factor
        # self._precompute()

    # def _precompute(self):
    #    self._sigma4 = self._sigma2 * self._sigma2
    #    self._sigma2_factor = 1.0 / self._sigma2
    #    self._sigma2_factor2 = 0.5 / self._sigma2

    @partial(jax.jit, static_argnums=(0,))
    def _softplus(self, x):
        """Softplus activation function.

        Computes the element-wise function
                softplus(x) = log(1 + e^x)
        """
        return jnp.logaddexp(x, 0)

    def _log_ffnn(self, r):
        """Logarithmic FFNN
        r: input
        params: class containing the weights and biases of the network in the form
        params.weights = [w1, w2, ..., wn] where wi is the weight matrix of layer i
        params.biases = [b1, b2, ..., bn] where bi is the bias vector of layer i
        """

        return

    def wf(self, r, params):
        """Evaluate the wave function"""
        return self._factor * self._log_ffnn(r, params).sum()

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        """Potential energy function
        This assumes non interacting particles!
        #TODO: add interaction
        """
        return 0.5 * jnp.sum(r * r)

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, r, params):
        """Probability amplitude"""
        return jnp.exp(self.logprob(r, params))

    def logprob(self, r, params):
        """Log probability amplitude"""
        psi2 = self._ffnn_psi_repr * self._log_ffnn(r, params).sum()
        return psi2

    def _grad_wf(self, r, params):
        """Gradient of the wave function wrt position"""
        return

    def laplacian_wf(self, r, params):
        """Laplacian of the wave function"""

        return

    def _local_kinetic_energy(self, r, params):
        """Evaluate the local kinetic energy"""
        _laplace = self.laplacian_wf(r, params).sum()
        _grad = self._grad_wf(r, params)
        _grad2 = np.sum(_grad * _grad)
        return -0.5 * (_laplace + _grad2)

    @partial(jax.jit, static_argnums=(0,))
    def local_energy(self, r, params):
        """Local energy of the system"""

        def ke_closure(r):
            return self._local_kinetic_energy(r, params)

        ke = jnp.sum(ke_closure(r))
        pe = self.potential(r)

        return ke + pe

    # def drift_force(self, r, params):
    #     """Drift force at each particle's location"""
    #     F = 2 * self._grad_wf(r, params)
    #     return F

    def grads(self, r, params):
        """Gradients of the wave function w.r.t. the parameters"""

        return


# class FFNN(NQS):
#     def __init__(
#         self,
#         nparticles,
#         dim,
#         nhidden=1,
#         interaction=False,
#         nqs_repr="psi2",
#         backend="numpy",
#         log=True,
#         logger_level="INFO",
#         rng=None,
#         use_sr=False,
#     ):
#         """RBM Neural Network Quantum State"""
#         super().__init__(
#             nparticles,
#             dim,
#             nhidden,
#             interaction,
#             nqs_repr,
#             backend,
#             log,
#             logger_level,
#             rng,
#             use_sr,
#             nqs_type="rbm",
#         )

#         self._nvisible = self._N * self._dim

#         if backend == "numpy":
#             if interaction:
#                 self.rbm = IRBM(self._N, self._dim, factor=self.factor)
#             else:
#                 self.rbm = NIRBM(factor=self.factor)
#         elif backend == "jax":
#             if interaction:
#                 self.rbm = JAXIRBM(self._N, self._dim, factor=self.factor)
#             else:
#                 self.rbm = JAXNIRBM(factor=self.factor)
#         else:
#             msg = "Unsupported backend, only 'numpy' or 'jax' is allowed"
#             raise ValueError(msg)

#         if self._log:
#             neuron_str = "neurons" if self._nhidden > 1 else "neuron"
#             msg = (
#                 f"Neural Network Quantum State initialized as RBM with "
#                 f"{self._nhidden} hidden {neuron_str}"
#             )
#             self.logger.info(msg)

#     def init(self, sigma2=1.0, seed=None):
#         """ """
#         self.rbm.sigma2 = sigma2

#         rng = self.rng(seed)

#         r = rng.standard_normal(size=self._nvisible)

#         # Initialize visible bias
#         v_bias = rng.standard_normal(size=self._nvisible) * 0.01
#         h_bias = rng.standard_normal(size=self._nhidden) * 0.01
#         kernel = rng.standard_normal(size=(self._nvisible, self._nhidden))
#         kernel *= np.sqrt(1 / self._nvisible)

#         self._params = Parameter()
#         self._params.set(["v_bias", "h_bias", "kernel"], [v_bias, h_bias, kernel])

#         # make logprob deal with the generator later instead of indexing
#         logp = self.rbm.logprob(
#             r,
#             self._params.get(["v_bias"]),
#             self._params.get(["h_bias"]),
#             self._params.get(["kernel"]),
#         )
#         self.state = State(r, logp, 0, 0)
#         self._is_initialized_ = True

#     def train(
#         self,
#         max_iter=100_000,
#         batch_size=1000,
#         early_stop=False,  # set to True later
#         rtol=1e-05,
#         atol=1e-08,
#         seed=None,
#         mcmc_alg=None,
#     ):
#         """
#         Train the NQS model using the specified gradient method.
#         """

#         self._is_initialized()
#         self._training_cycles = max_iter
#         self._training_batch = batch_size

#         if mcmc_alg is not None:
#             self._sampler = Sampler(
#                 self.mcmc_alg, self.rbm, self.rng, logger=self.logger
#             )

#         state = self.state
#         v_bias, h_bias, kernel = self._params.get(["v_bias", "h_bias", "kernel"])

#         # Reset n_accepted
#         state = State(state.positions, state.logp, 0, state.delta)

#         if self._log:
#             t_range = tqdm(
#                 range(max_iter),
#                 desc="[Training progress]",
#                 position=0,
#                 leave=True,
#                 colour="green",
#             )
#         else:
#             t_range = range(max_iter)

#         # Config
#         did_early_stop = False
#         seed_seq = generate_seed_sequence(seed, 1)[0]
#         steps_before_optimize = batch_size
#         energies = []
#         grads_v_bias = []
#         grads_h_bias = []
#         grads_kernel = []

#         # Training
#         for _ in t_range:
#             # sampler step
#             state = self._sampler.step(
#                 self.rbm, state, v_bias, h_bias, kernel, seed_seq
#             )

#             # getting and saving local energy
#             # print("state.positions", state.positions.shape)

#             loc_energy = self.hamiltonian.local_energy(
#                 self.rbm, state.positions, self._params
#             )
#             energies.append(loc_energy)

#             # getting and saving gradients
#             gr_v_bias, gr_h_bias, gr_kernel = self.rbm.grads(
#                 state.positions, v_bias, h_bias, kernel
#             )

#             grads_v_bias.append(gr_v_bias)
#             grads_h_bias.append(gr_h_bias)
#             grads_kernel.append(gr_kernel)

#             steps_before_optimize -= 1
#             # optimize
#             if steps_before_optimize == 0:
#                 # Expectation values
#                 energies = np.array(energies)
#                 grads_v_bias = np.array(grads_v_bias)
#                 grads_h_bias = np.array(grads_h_bias)
#                 grads_kernel = np.array(grads_kernel)

#                 expval_energy = np.mean(energies)
#                 expval_grad_v_bias = np.mean(grads_v_bias, axis=0)
#                 expval_grad_h_bias = np.mean(grads_h_bias, axis=0)
#                 expval_grad_kernel = np.mean(
#                     grads_kernel, axis=0
#                 )  # we shall use this in SR. I think this is avg dlogpsi/dW

#                 if self.use_sr:
#                     self.sr_matrix = self.rbm.compute_sr_matrix(
#                         expval_grad_kernel, grads_kernel
#                     )

#                 expval_energy_v_bias = np.mean(
#                     energies.reshape(batch_size, 1) * grads_v_bias, axis=0
#                 )
#                 expval_energy_h_bias = np.mean(
#                     energies.reshape(batch_size, 1) * grads_h_bias, axis=0
#                 )
#                 expval_energy_kernel = np.mean(
#                     energies.reshape(batch_size, 1, 1) * grads_kernel, axis=0
#                 )

#                 # variance = np.mean(energies**2) - energy**2

#                 # Gradients
#                 expval_energies = [
#                     expval_energy_v_bias,
#                     expval_energy_h_bias,
#                     expval_energy_kernel,
#                 ]
#                 expval_grads = [
#                     expval_grad_v_bias,
#                     expval_grad_h_bias,
#                     expval_grad_kernel,
#                 ]

#                 final_grads = [
#                     2 * (expval_energy_param - expval_energy * expval_grad_param)
#                     for expval_energy_param, expval_grad_param in zip(
#                         expval_energies, expval_grads
#                     )
#                 ]

#                 if early_stopping:
#                     # make copies of current values before update
#                     v_bias_old = copy.deepcopy(v_bias)  # noqa
#                     h_bias_old = copy.deepcopy(h_bias)  # noqa
#                     kernel_old = copy.deepcopy(kernel)  # noqa

#                 # Descent
#                 params = self._optimizer.step(self._params, final_grads, self.sr_matrix)
#                 v_bias, h_bias, kernel = params.get(["v_bias", "h_bias", "kernel"])

#                 energies = []
#                 grads_v_bias = []
#                 grads_h_bias = []
#                 grads_kernel = []
#                 steps_before_optimize = batch_size

#         # early stop flag activated
#         if did_early_stop:
#             msg = "Early stopping, training converged"
#             self.logger.info(msg)
#         # msg: Early stopping, training converged

#         # end
#         # Update shared values. we separate them before because we want to keep the old values
#         self.state = state
#         self._params.set(["v_bias", "h_bias", "kernel"], [v_bias, h_bias, kernel])
#         # self.scale = scale
#         self._is_trained_ = True

#     def tune(
#         self,
#         tune_iter=20_000,
#         tune_interval=500,
#         early_stop=False,  # set to True later
#         rtol=1e-05,
#         atol=1e-08,
#         seed=None,
#         mcmc_alg=None,
#     ):
#         """
#         BROKEN NOW due to self.scale
#         Tune proposal scale so that the acceptance rate is around 0.5.
#         """

#         self._is_initialized()
#         state = self.state
#         v_bias, h_bias, kernel = self._params.get(["v_bias", "h_bias", "kernel"])

#         scale = self.scale

#         if mcmc_alg is not None:
#             self._sampler = Sampler(self.mcmc_alg, self.rbm, self.rng, self._log)

#         # Used to throw warnings if tuned alg mismatch chosen alg
#         # in other procedures
#         self._tuned_mcmc_alg = self.mcmc_alg

#         # Config
#         # did_early_stop = False
#         seed_seq = generate_seed_sequence(seed, 1)[0]

#         # Reset n_accepted
#         state = State(state.positions, state.logp, 0, state.delta)

#         if self._log:
#             t_range = tqdm(
#                 range(tune_iter),
#                 desc="[Tuning progress]",
#                 position=0,
#                 leave=True,
#                 colour="green",
#             )
#         else:
#             t_range = range(tune_iter)

#         steps_before_tune = tune_interval

#         for i in t_range:
#             state = self._sampler.step(state, v_bias, h_bias, kernel, seed_seq)
#             steps_before_tune -= 1

#             if steps_before_tune == 0:
#                 # Tune proposal scale
#                 old_scale = scale
#                 accept_rate = state.n_accepted / tune_interval
#                 scale = self._sampler.tune_scale(old_scale, accept_rate)

#                 # Reset
#                 steps_before_tune = tune_interval
#                 state = State(state.positions, state.logp, 0, state.delta)

#         # Update shared values
#         self.state = state
#         self._params.set(["v_bias", "h_bias", "kernel"], [v_bias, h_bias, kernel])
#         self.scale = scale
#         self._is_tuned_ = True
