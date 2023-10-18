import jax
import jax.numpy as jnp
import numpy as np
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
        backend="numpy",
    ):
        """
        Initializes the FFNN Neural Network Quantum State.
        We here assume Ψ(x) = exp(FFNN(x)) so logΨ(x) = FFNN(x).
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
            "W0": rng.standard_normal(size=(self._N * self._dim, self._layer_sizes[1]))
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

        self.params = Parameter().set(self.params)

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

    def _configure_backend(self, backend):
        if backend == "numpy":
            self.backend = np
            self.la = np.linalg
            self.sigmoid = expit

        elif backend == "jax":
            self.backend = jnp
            self.la = jnp.linalg
            self.sigmoid = expit  # jax.nn.sigmoid
            self.grad_wf_closure = self.grad_wf_closure_jax
            self.grads_closure = self.grads_closure_jax
            self.laplacian_closure = self.laplacian_closure_jax
            # self._convert_constants_to_jnp()
            self._jit_functions()
        else:
            raise ValueError(f"Invalid backend: {backend}")

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
            "logprob_closure",
            "grad_wf_closure",
            "grads_closure",
            "laplacian_closure",
            "_precompute",
            "_softplus",
        ]
        for func_name in functions_to_jit:
            setattr(self, func_name, jax.jit(getattr(self, func_name)))
        return self

    def _precompute(self):
        self._sigma4 = self._sigma2 * self._sigma2
        self._sigma2_factor = 1.0 / self._sigma2
        self._sigma2_factor2 = 0.5 / self._sigma2

    def _softplus(self, x):
        """Softplus activation function.

        Computes the element-wise function
                softplus(x) = log(1 + e^x)
        """
        return self.backend.logaddexp(x, 0)

    def _log_wf(self, r, v_bias, h_bias, kernel):
        """FFNN forqard pass sinse the log of the wave function is the FFNN"""

        return self.forward(r)

    def wf(self, r, v_bias, h_bias, kernel):
        """Evaluate the wave function"""
        return self._factor * self._log_wf(r, v_bias, h_bias, kernel).sum()

    def pdf(self, r, v_bias, h_bias, kernel):
        """
        Probability amplitude
        """
        return self.backend.exp(self.logprob(r, v_bias, h_bias, kernel))

    def logprob_closure(self, r, v_bias, h_bias, kernel):
        """Log probability amplitude"""
        return self._ffnn_psi_repr * self._log_wf(r, v_bias, h_bias, kernel).sum()

    def logprob(self, r):
        """Log probability amplitude"""
        pass
        # v_bias, h_bias, kernel = self.params.get(["v_bias", "h_bias", "kernel"])
        # return self.logprob_closure(r, v_bias, h_bias, kernel)

    # @partial(jax.jit, static_argnums=(0,))
    def grad_wf_closure(self, r):
        """
        This is the autograd version of the gradient of the logarithm of the wave function w.r.t. the coordinates
        """

        grad_wf = jax.grad(self.wf, argnums=0)

        return grad_wf(r)

    def grad_wf(self, r):
        """
        grad of the wave function w.r.t. the coordinates
        """
        params = self.params.get(self.params.keys())  # this is stupid but it works
        print(params)
        return self.grad_wf_closure(r, params)

    # @partial(jax.jit, static_argnums=(0,))
    def laplacian_closure(self, r, v_bias, h_bias, kernel):
        """
        nabla^2 of the wave function w.r.t. the coordinates
        """

        def wrapped_wf(r_):
            return self.wf(r_, v_bias, h_bias, kernel)

        grad_wf = jax.grad(wrapped_wf)
        hessian_wf = jax.jacfwd(
            grad_wf
        )  # This computes the Jacobian of the gradient, which is the Hessian

        hessian_at_r = hessian_wf(r)

        # If you want the Laplacian, you'd sum the diagonal of the Hessian.
        # This assumes r is a vector and you want the Laplacian w.r.t. each element.
        laplacian = jnp.trace(hessian_at_r)

        return laplacian

    def laplacian(self, r):
        v_bias, h_bias, kernel = self.params.get(["v_bias", "h_bias", "kernel"])
        return self.laplacian_closure(r, v_bias, h_bias, kernel)

    # @partial(jax.jit, static_argnums=(0,))
    def grads_closure(self, r, v_bias, h_bias, kernel):
        """
        This is the autograd version of the gradient of the logarithm of the wave function w.r.t. the parameters
        """

        grad_v_bias = jax.grad(self.wf, argnums=1)
        grad_h_bias = jax.grad(self.wf, argnums=2)
        grad_kernel = jax.grad(self.wf, argnums=3)

        return (
            grad_v_bias(r, v_bias, h_bias, kernel),
            grad_h_bias(r, v_bias, h_bias, kernel),
            grad_kernel(r, v_bias, h_bias, kernel),
        )

    def grads(self, r):
        """Gradients of the wave function w.r.t. the parameters"""
        params = self.params.get(self.params.keys())

        return self.grads_closure(r, params)
