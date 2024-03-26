from functools import partial  # noqa

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from src.models.base_wf import WaveFunction
from src.state.utils import Parameter
from src.state.utils import State

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class FFNN(WaveFunction):
    def __init__(
        self,
        nparticles,
        dim,
        layer_sizes,
        activations,
        factor=1.0,  # not sure about this value
        rng=None,
        log=False,
        logger=None,
        logger_level="INFO",
        backend="jax",
        symmetry=None,
        correlation=None,
    ):
        super().__init__(
            nparticles,
            dim,
            rng=rng,
            log=log,
            logger=logger,
            logger_level=logger_level,
            backend=backend,
        )

        """
        Initializes the FFNN Neural Network Quantum State.
        We here assume Ψ(x) = exp(FFNN(x)) so logΨ(x) = FFNN(x).
        Note that we are only going to use JAX backend for this, because we do not want to derive the gradients by hand.

        Args:
        - nparticles (int): Number of particles.
        - dim (int): Dimensionality.
        ...
        """
        self._initialize_vars(nparticles, dim, layer_sizes, activations, factor)
        self.configure_symmetry(symmetry)  # need to be before correlation
        self.configure_correlation(correlation)  # NEED TO BE BEFORE CONFIGURE_BACKEND
        self.configure_backend(backend)

        self._initialize_layers(rng)

        logp = self.logprob(self.r0)
        self.state = State(self.r0, logp, 0, 0)

        if self.log:
            msg = f"Neural Network Quantum State initialized with symmetry {self.symmetry} as FFNN with {self.__str__()}."  # noqa
            self.logger.info(msg)

    def reinit_positions(self):
        self._reinit_positions()
        self.state = State(self.r0, self.logprob(self.r0), 0, self.state.delta)

    def _initialize_layers(self, rng):
        self.params = Parameter()  # Initialize empty Parameter object

        for i in range(0, len(self._layer_sizes) - 1):
            # The number of units in the layers
            input_size = self._layer_sizes[i]
            output_size = self._layer_sizes[i + 1]

            # He initialization, considering the size of layers
            limit = np.sqrt(2 / (input_size))

            # Initialize weights and biases
            self.params.set(
                f"W{i}",
                np.array(rng.uniform(-limit, limit, (input_size, output_size))),
            )

            self.params.set(
                f"b{i}",
                np.zeros((output_size,)),
                # jnp.array(rng.standard_normal(size=(output_size,)))
            )

        if self.jastrow:
            input_j_size = self._N * (self._N - 1) // 2
            limit = np.sqrt(2 / (input_j_size))
            self.params.set(
                "WJ", np.array(rng.uniform(-limit, limit, (self._N, self._N)))
            )
        if self.pade_jastrow:
            assert not self.jastrow, "Pade Jastrow requires Jastrow to be false"
            self.params.set("CPJ", np.array(rng.uniform(-limit, limit, 1)))

    def log_wf_pretrain(self, x, params):
        """
        This is the forward pass of the FFNN
        x: (batch_size, part * dim) array

        """
        output = self.log_wf0(x, params)
        return output

    # @WaveFunction.symmetry
    def log_wf0(self, x, params):
        """ """

        for i in range(0, len(self._layer_sizes) - 1):
            x = x @ params.get(f"W{i}") + params.get(f"b{i}")
            x = self.activation(self._activations[i])(x)

        return x.squeeze(-1)

    def grad_wf_closure_jax(self, r, params):
        """
        This is the autograd version of the gradient of the logarithm of the wave function w.r.t. the coordinates
        """
        grad_wf = jax.grad(self.log_wf, argnums=0)
        return vmap(grad_wf, in_axes=(0, None))(r, params)

    def grad_wf(self, r):
        """
        (∇_r) Ψ(r) = ∑_i (∇_r) Ψ(r_i)
        """
        return self.grad_wf_closure(r, self.params)

    def laplacian_closure_jax(self, r, params):
        """
        (∇_r)^2 Ψ(r) = ∑_i (∇_r)^2 Ψ(r_i)
        This can be optimized by using the fact that Ψ = exp(FFNN)
        For now we will use the autograd version
        """

        hessian_wf = vmap(jax.hessian(self.log_wf), in_axes=(0, None))
        trace_hessian = vmap(jnp.trace)

        return trace_hessian(hessian_wf(r, params))

    def laplacian(self, r):
        """
        examine who is which particle and who is which dimension

        """
        return self.laplacian_closure(r, self.params)

    def grads_closure_jax(self, r, params):
        grad_fn = vmap(jax.grad(self.log_wf, argnums=1), in_axes=(0, None))
        grad_eval = grad_fn(r, params)  # still a parameter type
        return grad_eval

    def grads(self, r):
        """
        Gradients of the wave function with respect to the neural network parameters.
        """

        # jax.grad will return a function that computes the gradient of wf.
        # This function will expect a parameter input, which in our case are the neural network parameters.

        grads_dict = self.grads_closure(r, self.params)

        return grads_dict

    def pdf(self, r):
        """Probability density function"""

        return self.backend.exp(self.logprob(r))

    def grads_logprob(self, r):
        """
        Gradients of the log probability amplitude
        Note that grads is the gradient of the log of the wave function
        WIP
        """

        return 2 * self.grads(r)

    def logprob_closure(self, r, params):
        """Log probability amplitude
        that is log(|psi|^2).
        In our case, psi is real, so this is 2 log(|psi|) = 2 * forward(r, params) ?
        """

        return 2.0 * self.log_wf(r, params)

    def logprob_closure_pretrain(self, r, params):
        """Log probability amplitude
        that is log(|psi|^2).
        In our case, psi is real, so this is 2 log(|psi|) = 2 * forward(r, params) ?
        """

        return 2.0 * self.log_wf_pretrain(r, params)

    def logprob(self, r):
        """Log probability amplitude"""

        return self.logprob_closure(r, self.params)

    def compute_sr_matrix(self, expval_grads, grads, shift=1e-6):
        """
        expval_grads and grads should be dictionaries with keys "v_bias", "h_bias", "kernel" in the case of RBM
        in the case of FFNN we have "weights" and "biases" and "kernel" is not present
        WIP: for now this does not involve the averages because r will be a single sample
        Compute the matrix for the stochastic reconfiguration algorithm
            for now we do it only for the kernel
            The expression here is for kernel element W_ij:
                S_ij,kl = < (d/dW_ij log(psi)) (d/dW_kl log(psi)) > - < d/dW_ij log(psi) > < d/dW_kl log(psi) >

            For bias (V or H) we have:
                S_i,j = < (d/dV_i log(psi)) (d/dV_j log(psi)) > - < d/dV_i log(psi) > < d/dV_j log(psi) >


            1. Compute the gradient ∂_W log(ψ) using the _grad_kernel function.
            2. Compute the outer product of the gradient with itself: ∂_W log(ψ) ⊗ ∂_W log(ψ)
            3. Compute the expectation value of the outer product over all the samples
            4. Compute the expectation value of the gradient ∂_W log(ψ) over all the samples
            5. Compute the outer product of the expectation value of the gradient with itself: <∂_W log(ψ)> ⊗ <∂_W log(ψ)>

            OBS: < d/dW_ij log(psi) > is already done inside train of the RBM class but we need still the < (d/dW_ij log(psi)) (d/dW_kl log(psi)) >
        """
        sr_matrices = {}

        for key, grad_value in grads.items():
            grad_value = self.backend.array(grad_value)

            if "W" in key:  # means it is a matrix
                grads_outer = self.backend.einsum(
                    "nij,nkl->nijkl", grad_value, grad_value
                )
            # elif self.backend.ndim(grad_value[0]) == 1:
            else:  # means it is a (bias) vector
                grads_outer = self.backend.einsum("ni,nj->nij", grad_value, grad_value)

            expval_outer_grad = self.backend.mean(grads_outer, axis=0)
            outer_expval_grad = self.backend.outer(expval_grads[key], expval_grads[key])

            sr_mat = (
                expval_outer_grad.reshape(outer_expval_grad.shape) - outer_expval_grad
            )
            sr_matrices[key] = sr_mat + shift * self.backend.eye(sr_mat.shape[0])

        return sr_matrices

    def _initialize_vars(self, nparticles, dim, layer_sizes, activations, factor):
        self._factor = factor
        self._layer_sizes = layer_sizes
        self._activations = activations
        self._ffnn_psi_repr = 2 * self._factor
        self._N = nparticles
        self._dim = dim
        self._nvisible = self._N * self._dim
        if len(layer_sizes) != len(activations) + 1:
            raise ValueError(
                f"num layers ({len(layer_sizes)}) != num activations +1 ({len(activations)})"
            )

    def __str__(self):
        """
        Construct a string representation of the neural network architecture.
        """
        # Start the network representation with the 'input layer'
        net_repr = "Neural Network Architecture\n"
        net_repr += "-" * 30 + "\n"  # Divider for clarity

        for i, num_neurons in enumerate(self._layer_sizes):
            layer_type = (
                "Input"
                if i == 0
                else ("Output" if i == len(self._layer_sizes) - 1 else "Hidden")
            )
            layer_info = f"{layer_type} Layer (Layer {i}): {num_neurons} neurons\n"

            # Compile layer information and visual representation
            net_repr += layer_info

        net_repr += "-" * 30  # Ending divider

        return net_repr
