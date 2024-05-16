from functools import partial  # noqa

import jax
from jax import vmap

from nqs.models.base_wf import WaveFunction
from nqs.state.utils import Parameter
from nqs.state.utils import State

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
        particle=None,
        correlation=None,
    ):
        """
        Implements a Feed-Forward Neural Network (FFNN) as a quantum wave function approximation.

        This class is designed to model the quantum wave function using a neural network architecture,
        where the wave function is represented as Ψ(x) = exp(FFNN(x)), hence logΨ(x) = FFNN(x). The class
        exclusively uses the JAX backend for efficient, autograd-compatible computations.

        Attributes:
            Inherits all attributes from the `WaveFunction` class and introduces additional attributes
            related to the neural network parameters and architecture.

        Methods:
            reinit_positions: Reinitializes the positions of the particles and updates the state.
            _initialize_layers: Initializes the neural network layers and parameters.
            log_wf_pretrain: Computes the logarithm of the wave function before training.
            log_wf0: The base implementation of the neural network forward pass.
            grad_wf_closure_jax: Computes the gradient of the wave function with respect to the particle coordinates using JAX.
            grad_wf: Wrapper for `grad_wf_closure_jax` to compute gradients with respect to coordinates.
            laplacian_closure_jax: Computes the Laplacian of the wave function using JAX.
            laplacian: Wrapper for `laplacian_closure_jax` to compute the Laplacian with respect to the particle coordinates.
            grad_params_closure_jax: Computes the gradients of the wave function with respect to the neural network parameters using JAX.
            grad_params: Wrapper for `grad_params_closure_jax` to compute gradients with respect to the neural network parameters.
            pdf: Computes the probability density function.
            logprob_closure: Computes the logarithm of the probability amplitude for given parameters.
            logprob_closure_pretrain: Similar to `logprob_closure` but used before training.
            logprob: Computes the logarithm of the probability amplitude.
            _initialize_vars: Initializes internal variables related to the neural network configuration.

        The FFNN class supports JAX for automatic differentiation to efficiently compute gradients required for quantum Monte Carlo simulations.
        """
        super().__init__(
            nparticles,
            dim,
            rng=rng,
            logger=logger,
            logger_level=logger_level,
            backend=backend,
        )

        """
        Initializes the FFNN neural network quantum state.

        Parameters:
            nparticles (int): Number of particles in the system.
            dim (int): Dimensionality of the system (e.g., 2 for a 2D system).
            layer_sizes (list of int): Specifies the size of each layer in the network, including the input and output layers.
            activations (list of callable): Activation functions for each layer of the network, excluding the input layer.
            factor (float, optional): A scaling factor for the network's output, determining the amplitude of the wave function. Defaults to 1.0.
            rng (optional): Random number generator instance for initializing network parameters.
            log (bool, optional): Flag to enable or disable logging. Defaults to False.
            logger (Logger, optional): Logger instance for logging information about the network initialization and training.
            logger_level (str, optional): Logging level. Defaults to "INFO".
            backend (str, optional): Specifies the computational backend to use. Currently, only "jax" is supported.
            particle (optional): Configures the network to use a particular particle. Not implemented in this snippet.
            correlation (optional): Configures the network to use correlations between particles. Not implemented in this snippet.

        Initializes network parameters, sets up the architecture, and prepares the initial state.
        """
        self._initialize_vars(nparticles, dim, layer_sizes, activations, factor)
        self.configure_particle(particle)  # need to be before correlation
        self.configure_correlation(correlation)  # NEED TO BE BEFORE CONFIGURE_BACKEND
        self.configure_backend(backend)

        self._initialize_layers(rng)

        logp = self.logprob(self.r0)
        self.state = State(self.r0, logp, 0, 0)

        if self.logger_level != "SILENT":
            msg = f"Neural Network Quantum State initialized with particle {self.particle} as FFNN with {self.__str__()}."  # noqa
            self.logger.info(msg)

    def reinit_positions(self):
        """
        Reinitializes the positions of the particles and updates the wave function state accordingly.

        This method is useful for resetting the simulation to a new starting point.
        """
        self._reinit_positions()
        self.state = State(self.r0, self.logprob(self.r0), 0, self.state.delta)

    def _initialize_layers(self, rng):
        """
        Initializes the neural network layers and parameters using the provided random number generator.

        This method sets up the weights and biases for each layer of the neural network according to
        the He initialization scheme, which is suited for layers followed by ReLU activation functions.

        Parameters:
            rng: A random number generator capable of producing uniform distributions used for initializing weights.
        """
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
        Computes the logarithm of the wave function prior to training, using the initial network parameters.

        This method serves as a forward pass through the neural network to evaluate the log wave function
        for a given set of particle positions `x` and initial network parameters `params`.

        Parameters:
            x: A batch of particle positions of shape (batch_size, nparticles * dim).
            params: The initial parameters of the neural network.

        Returns:
            The logarithm of the wave function evaluated at `x` using the initial parameters `params`.
        """
        output = self.log_wf0(x, params)
        return output

    # @WaveFunction.particle
    # @partial(jax.jit, static_argnums=(0,))
    def log_wf0(self, x, params):
        """
        The core implementation of the neural network's forward pass.

        This method sequentially applies the network's layers to the input `x`, using the weights
        and biases specified by `params`, to compute the network's output.

        Parameters:
            x: A batch of particle positions of shape (batch_size, nparticles * dim).
            params: The current parameters of the neural network.

        Returns:
            A tensor of shape (batch_size,) representing the logarithm of the wave function evaluated at each position in `x`.
        """

        weights = [params.get(f"W{i}") for i in range(len(self._layer_sizes) - 1)]
        biases = [params.get(f"b{i}") for i in range(len(self._layer_sizes) - 1)]

        for i, (w, b) in enumerate(zip(weights, biases)):
            x = self.backend.matmul(x, w) + b
            x = self.activation(self._activations[i])(x)

        return x.squeeze(-1)

    def grad_wf_closure_jax(self, r, params):
        """
        Computes the gradient of the logarithm of the wave function with respect to the particle coordinates using JAX.

        This method utilizes automatic differentiation provided by JAX to efficiently compute the gradients
        required for the Monte Carlo updates.

        Parameters:
            r: A batch of particle positions.
            params: The current parameters of the neural network.

        Returns:
            A tensor representing the gradients of the log wave function with respect to `r`.
        """

        grad_wf = jax.grad(self.log_wf, argnums=0)
        return vmap(grad_wf, in_axes=(0, None))(r, params)

    def grad_wf(self, r):
        """
        Wrapper method for `grad_wf_closure_jax` to compute the gradients of the wave function with respect to the coordinates.

        Parameters:
            r: A batch of particle positions.

        Returns:
            A tensor representing the gradients of the log wave function with respect to `r`.
        """
        return self.grad_wf_closure(r, self.params)

    def laplacian_closure_jax(self, r, params):
        """
        Computes the Laplacian of the logarithm of the wave function with respect to the particle coordinates using JAX.

        Utilizes the Hessian computation provided by JAX to calculate the Laplacian, which is the trace of the Hessian matrix,
        for each particle position in the batch.

        Parameters:
            r: A batch of particle positions.
            params: The current parameters of the neural network.

        Returns:
            A tensor representing the Laplacian of the log wave function with respect to `r`.
        """

        hessian_wf = vmap(jax.hessian(self.log_wf), in_axes=(0, None))
        trace_hessian = vmap(self.backend.trace)

        return trace_hessian(hessian_wf(r, params))

    def laplacian(self, r):
        """
        Wrapper method for `laplacian_closure_jax` to compute the Laplacian of the wave function with respect to the coordinates.

        Parameters:
            r: A batch of particle positions.

        Returns:
            A tensor representing the Laplacian of the log wave function with respect to `r`.
        """
        return self.laplacian_closure(r, self.params)

    def grad_params_closure_jax(self, r, params):
        """
        Computes the gradients of the wave function with respect to the neural network parameters using JAX.

        This method is crucial for the optimization of the neural network parameters during the learning process.

        Parameters:
            r: A batch of particle positions.
            params: The current parameters of the neural network.

        Returns:
            A `Parameter` object containing the gradients of the log wave function with respect to each parameter in `params`.
        """
        grad_fn = vmap(jax.grad(self.log_wf, argnums=1), in_axes=(0, None))
        grad_eval = grad_fn(r, params)  # still a parameter type
        return grad_eval

    def grad_params(self, r):
        """
        Wrapper method for `grad_params_closure_jax` to compute the gradients of the wave function with respect to the neural network parameters.

        Parameters:
            r: A batch of particle positions.

        Returns:
            A `Parameter` object containing the gradients of the log wave function with respect to the neural network parameters.
        """

        # jax.grad will return a function that computes the gradient of wf.
        # This function will expect a parameter input, which in our case are the neural network parameters.

        return self.grad_params_closure(r, self.params)

    def pdf(self, r):
        """
        Computes the probability density function (PDF) for a given set of particle positions.

        The PDF is proportional to the square of the wave function, i.e., |Ψ(r)|^2, which is necessary
        for sampling particle positions in quantum Monte Carlo simulations.

        Parameters:
            r: A batch of particle positions.

        Returns:
            A tensor representing the PDF evaluated at each position in `r`.
        """

        return self.backend.exp(self.logprob(r))

    def logprob_closure(self, r, params):
        """
        Computes the logarithm of the probability amplitude for given particle positions and network parameters.

        This method directly relates to the computation of observables and the evaluation of the Monte Carlo acceptance ratio.

        Parameters:
            r: A batch of particle positions.
            params: The current parameters of the neural network.

        Returns:
            A tensor representing the log of the probability amplitude evaluated at each position in `r`.
        """

        return 2.0 * self.log_wf(r, params)

    def logprob_closure_pretrain(self, r, params):
        """
        Similar to `logprob_closure`, but intended for use before the training process begins, using initial network parameters.

        Parameters:
            r: A batch of particle positions.
            params: The initial parameters of the neural network.

        Returns:
            A tensor representing the log of the probability amplitude evaluated at each position in `r`, using the initial parameters.
        """

        return 2.0 * self.log_wf_pretrain(r, params)

    def logprob(self, r):
        """
        Computes the logarithm of the probability amplitude for a given set of particle positions using the current network parameters.

        This is the primary method used during simulations to evaluate the wave function's contribution to observables and the Monte Carlo algorithm.

        Parameters:
            r: A batch of particle positions.

        Returns:
            A tensor representing the log of the probability amplitude evaluated at each position in `r`.
        """

        return self.logprob_closure(r, self.params)

    def _initialize_vars(self, nparticles, dim, layer_sizes, activations, factor):
        """
        Initializes internal variables related to the neural network configuration.

        This method sets up essential attributes that define the neural network's architecture, including
        the number of particles, dimensionality, layer sizes, activation functions, and a scaling factor for the network output.

        Parameters:
            nparticles (int): The number of particles in the system.
            dim (int): The dimensionality of the system (e.g., 2 for a 2D system).
            layer_sizes (list of int): Specifies the sizes of each layer in the network, including the input and output layers.
            activations (list of callable): Activation functions for each layer of the network, excluding the input layer.
            factor (float): A scaling factor for the network's output, influencing the amplitude of the wave function. It's used to adjust the range of the network's output to match the expected range of the wave function.

        This method validates the provided configuration, ensuring that the number of layer sizes matches the number of activations plus one for the output layer.
        It raises a ValueError if the configuration is invalid. Additionally, it initializes attributes that represent the network's structure and parameters.
        """
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
