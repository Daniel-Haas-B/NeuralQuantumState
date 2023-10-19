import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from nqs.utils import Parameter
from nqs.utils import State
from scipy.special import expit

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class FFNN:
    def __init__(
        self,
        nparticles,
        dim,
        layer_sizes,
        activations,
        factor=1.0,  # not sure about this value
        sigma2=1.0,
        rng=None,
        log=False,
        logger=None,
        logger_level="INFO",
        backend="jax",
    ):
        """
        Initializes the FFNN Neural Network Quantum State.
        We here assume Ψ(x) = exp(FFNN(x)) so logΨ(x) = FFNN(x).
        Note that we are only going to use JAX backend for this.

        Args:
        - nparticles (int): Number of particles.
        - dim (int): Dimensionality.
        ...
        """
        self._initialize_vars(nparticles, dim, layer_sizes, activations, factor, sigma2)
        self._configure_backend(backend)
        if logger:
            self.logger = logger
        else:
            import logging

            self.logger = logging.getLogger(__name__)

        self.log = log
        self.rng = rng if rng else np.random.default_rng()
        r = rng.standard_normal(size=self._nvisible)

        self._initialize_layers(rng)

        logp = self.logprob(r)
        self.state = State(r, logp, 0, 0)

        if self.log:
            msg = f"Neural Network Quantum State initialized as FFNN with {self.__str__()}"
            self.logger.info(msg)

    def _initialize_layers(self, rng):
        """Always initialize all as rectangle for now"""

        self.params = {
            "b0": rng.standard_normal(size=(self._N * self._dim)) * 0.01,
            "W0": rng.standard_normal(size=(self._N * self._dim, self._layer_sizes[0]))
            * 0.01,
        }

        for i in range(1, len(self._layer_sizes)):
            # The number of units in the layers
            input_size = self._layer_sizes[i - 1]
            output_size = self._layer_sizes[i]

            # Glorot initialization, considering the size of layers
            limit = np.sqrt(6 / (input_size + output_size))

            # Initialize weights and biases
            self.params[f"W{i}"] = np.random.uniform(
                -limit, limit, (input_size, output_size)
            )
            self.params[f"b{i}"] = np.zeros((output_size,))

        param_obj = Parameter()
        param_obj.set(self.params)
        self.params = param_obj

    def _initialize_vars(
        self, nparticles, dim, layer_sizes, activations, factor, sigma2
    ):
        self._sigma2 = sigma2
        self._factor = factor
        self._layer_sizes = layer_sizes
        self._activations = activations
        self._ffnn_psi_repr = 2 * self._factor
        self._N = nparticles
        self._dim = dim
        self._nvisible = self._N * self._dim
        if len(layer_sizes) != len(activations):
            raise ValueError(
                f"num layers ({len(layer_sizes)}) != num activations ({len(activations)})"
            )
        self._precompute()

    def activation(self, activation_str):
        match activation_str:
            case "tanh":
                return jnp.tanh
            case "sigmoid":
                return jnp.sigmoid
            case "relu":
                return jnp.maximum
            case "softplus":
                return jnp.logaddexp
            case "gelu":
                return jax.nn.gelu
            case _:  # default
                raise ValueError(f"Invalid activation function {activation_str}")

    def _configure_backend(self, backend):
        if backend != "jax":
            raise ValueError(
                f"Invalid backend ({backend}) for FFNN. Only JAX is supported."
            )

        self.backend = jnp
        self.la = jnp.linalg
        self.sigmoid = expit  # jax.nn.sigmoid
        # self._convert_constants_to_jnp()
        # self._jit_functions()

    def __str__(self):
        """
        Construct a string representation of the neural network architecture.
        """
        # Start the network representation with the 'input layer'
        net_repr = "Neural Network Architecture\n"
        net_repr += "-" * 30 + "\n"  # Divider for clarity

        # Representing neurons as "o"
        neuron_repr = "o"

        for i, num_neurons in enumerate(self._layer_sizes):
            layer_type = (
                "Input"
                if i == 0
                else ("Output" if i == len(self._layer_sizes) - 1 else "Hidden")
            )
            layer_info = f"{layer_type} Layer (Layer {i}): {num_neurons} neurons\n"

            # Create a visual representation of neurons
            neurons = (neuron_repr * num_neurons).strip()

            # If you have many neurons, you might want to cap the number for display
            max_display_neurons = 10  # Adjust as needed
            if num_neurons > max_display_neurons:
                neurons = (
                    neuron_repr * max_display_neurons + "..."
                ).strip()  # Show that there are more neurons

            # Compile layer information and visual representation
            net_repr += layer_info
            net_repr += neurons + "\n\n"  # Add spacing between layers for clarity

        net_repr += "-" * 30  # Ending divider

        return net_repr

    def _jit_functions(self):
        functions_to_jit = [
            "_log_wf",
            "wf",
            "grad_wf_closure",
            "grads_closure",
            "laplacian_closure",
            "_precompute",
        ]
        for func_name in functions_to_jit:
            setattr(self, func_name, jax.jit(getattr(self, func_name)))
        return self

    def _precompute(self):
        self._sigma4 = self._sigma2 * self._sigma2
        self._sigma2_factor = 1.0 / self._sigma2
        self._sigma2_factor2 = 0.5 / self._sigma2

    def _apply_layers(self, x, params):
        """
        Applies the neural network layers and activations to input x.
        Will loop through each layer, applying the weights, bias, and activation function
        """

        for i in range(len(self._layer_sizes)):
            print("params.get([fW{i}]) type", type(params.get([f"W{i}"])))

            x = params.get([f"W{i}"]) @ x + params.get([f"b{i}"])
            x = self.activation(self._activations[i])(
                x
            )  # assuming self._activations stores strings

        return x

    def forward(self, x, params):
        """The forward method for computing the output of the FFNN."""
        return self._apply_layers(x, params)

    def _log_wf(self, r, params):
        """FFNN forqard pass sinse the log of the wave function is the FFNN"""

        return self.forward(
            r, params
        )  # Assuming the representation is Psi(x) = exp(FFNN(x))

    def wf(self, r, params):
        """Compute the wave function from the neural network output."""
        return self.forward(r, params)

    # @partial(jax.jit, static_argnums=(0,))
    def grad_wf_closure(self, r):
        """
        This is the autograd version of the gradient of the logarithm of the wave function w.r.t. the coordinates
        """
        grad_wf = jax.grad(self.wf, argnums=0)
        return grad_wf(r)

    def grad_wf(self, r):
        """
        (∇_r) Ψ(r) = ∑_i (∇_r) Ψ(r_i)
        """
        grad_fun = grad(
            self._log_wf
        )  # grad of log(wf) due to the property of exponential
        return grad_fun(r) * self.wf(r)  # Chain rule to get grad of wf itself

    # @partial(jax.jit, static_argnums=(0,))
    def laplacian_closure(self, r, params):
        """
        (∇_r)^2 Ψ(r) = ∑_i (∇_r)^2 Ψ(r_i)
        """
        grad_fun = grad(self._log_wf)
        laplacian_fun = grad(grad_fun)

        return (laplacian_fun(r) + grad_fun(r) ** 2) * self.wf(
            r, params
        )  # Using product rule and the fact that Ψ = exp(FFNN)

    def laplacian(self, r):
        return self.laplacian_closure(r)

    # @partial(jax.jit, static_argnums=(0,))
    def grads_closure(self, r, params):
        """
        This is the autograd version of the gradient of the logarithm of the wave function w.r.t. the parameters
        # This is obsolete and does not work well with out current layer implementation
        """

        grad_fn = jax.grad(self.wf, argnums=2)

        return grad_fn(r, params)

    def grads(self, r):
        """
        Gradients of the wave function with respect to the neural network parameters.
        """

        # jax.grad will return a function that computes the gradient of wf.
        # This function will expect a parameter input, which in our case are the neural network parameters.

        # The grad_fn function now computes the gradients of the wave function with respect to each parameter of
        # the neural network. We need to pass the current parameters to this function.

        params = self.params.get(
            self.params.keys()
        )  # Assuming self.params is a dict-like structure.

        # 'gradients' is now a structure (like a dictionary) holding the gradients with respect to each parameter.
        return self.grads_closure(r, params)

    # def pdf(self, r, params):
    #     """
    #     Probability amplitude
    #     """
    #     return self.backend.exp(self.logprob(r, params))

    def logprob_closure(self, r, params):
        """Log probability amplitude"""
        return self._ffnn_psi_repr * self._log_wf(r, params).sum()

    def logprob(self, r):
        """Log probability amplitude"""

        return self.logprob_closure(r, self.params)
