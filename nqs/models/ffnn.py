from functools import partial  # noqa

import jax
from jax import vmap
from src.models.base_wf import WaveFunction
from src.state.utils import Parameter
from src.state.utils import State

# import time


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
            limit = self.backend.sqrt(2 / (input_size))

            # Initialize weights and biases
            self.params.set(
                f"W{i}",
                self.backend.array(
                    rng.uniform(-limit, limit, (input_size, output_size))
                ),
            )

            self.params.set(
                f"b{i}",
                self.backend.zeros((output_size,)),
            )

        if self.jastrow:
            input_j_size = self.N * (self.N - 1) // 2
            limit = self.backend.sqrt(2 / (input_j_size))
            self.params.set(
                "WJ", self.backend.array(rng.uniform(-limit, limit, (self.N, self.N)))
            )
        if self.pade_jastrow:
            assert not self.jastrow, "Pade Jastrow requires Jastrow to be false"
            self.params.set("CPJ", self.backend.array(rng.uniform(-limit, limit, 1)))

    def log_wf_pretrain(self, x, params):
        """
        This is the forward pass of the FFNN
        x: (batch_size, part * dim) array

        """
        output = self.log_wf0(x, params)
        return output

    # @WaveFunction.symmetry
    # @partial(jax.jit, static_argnums=(0,))
    def log_wf0(self, x, params):
        """ """

        weights = [params.get(f"W{i}") for i in range(len(self._layer_sizes) - 1)]
        biases = [params.get(f"b{i}") for i in range(len(self._layer_sizes) - 1)]

        for i, (w, b) in enumerate(zip(weights, biases)):
            x = self.backend.matmul(x, w) + b
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
        """ """

        hessian_wf = vmap(jax.hessian(self.log_wf), in_axes=(0, None))
        trace_hessian = vmap(self.backend.trace)

        return trace_hessian(hessian_wf(r, params))

    def laplacian(self, r):
        """
        examine who is which particle and who is which dimension

        """
        return self.laplacian_closure(r, self.params)

    def grad_params_closure_jax(self, r, params):
        grad_fn = vmap(jax.grad(self.log_wf, argnums=1), in_axes=(0, None))
        grad_eval = grad_fn(r, params)  # still a parameter type
        return grad_eval

    def grad_params(self, r):
        """
        Gradients of the wave function with respect to the neural network parameters.
        """

        # jax.grad will return a function that computes the gradient of wf.
        # This function will expect a parameter input, which in our case are the neural network parameters.

        return self.grad_params_closure(r, self.params)

    def pdf(self, r):
        """Probability density function"""

        return self.backend.exp(self.logprob(r))

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

    def _initialize_vars(self, nparticles, dim, layer_sizes, activations, factor):
        self._factor = factor
        self._layer_sizes = layer_sizes
        self._activations = activations
        self._ffnn_psi_repr = 2 * self._factor
        self.N = nparticles
        self.dim = dim
        self.Nvisible = self.N * self.dim
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
