import jax
import jax.numpy as jnp
import numpy as np
from nqs.utils import Parameter
from nqs.utils import State
from scipy.special import expit

# from jax import grad

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
        self.params = Parameter()  # Initialize empty Parameter object

        # do not change this order
        self.params.set(
            "W0",
            jnp.array(
                rng.standard_normal(size=(self._N * self._dim, self._layer_sizes[0]))
            ),
        )
        self.params.set(
            "b0", jnp.array(rng.standard_normal(size=(self._layer_sizes[0])))
        )

        for i in range(1, len(self._layer_sizes)):
            # The number of units in the layers
            input_size = self._layer_sizes[i - 1]
            output_size = self._layer_sizes[i]

            # Glorot initialization, considering the size of layers
            limit = np.sqrt(6 / (input_size + output_size))

            # Initialize weights and biases
            self.params.set(
                f"W{i}",
                jnp.array(rng.uniform(-limit, limit, (input_size, output_size))),
            )
            self.params.set(f"b{i}", jnp.zeros((output_size,)))

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
                return expit  # jax.nn.sigmoid
            case "relu":
                return jax.nn.relu
            case "softplus":
                return jax.nn.softplus
            case "gelu":
                return jax.nn.gelu
            case "linear":
                return lambda x: x
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
        self._jit_functions()

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

    def forward(self, x, params):
        if isinstance(params, Parameter):
            # Custom Parameter instance
            for i in range(0, len(self._layer_sizes)):
                x = params.get(f"W{i}").T @ x + params.get(f"b{i}")
                x = self.activation(self._activations[i])(x)
        elif isinstance(params, tuple):
            # Tuple of JAX arrays from grad
            for i in range(0, len(self._layer_sizes)):
                # Assuming params are ordered as W0, b0, W1, b1, ..., Wn, bn
                W = params[i * 2]  # Even indices are weights
                b = params[i * 2 + 1]  # Odd indices are biases
                x = W.T @ x + b
                x = self.activation(self._activations[i])(x)
        else:
            raise TypeError("Unexpected parameter format")
        return x

    # def log_wf(self, r, params):
    #     """Compute the logarithm of the wave function from the neural network output.

    #     """
    #     return self.forward(
    #         r, params
    #     ).sum(axis=0)

    def wf(self, r, params):
        """Compute the wave function from the neural network output.
        this is actually exp(forward(r, params)

        """
        return jnp.exp(self.forward(r, params).sum(axis=0))

    # @partial(jax.jit, static_argnums=(0,))
    def grad_wf_closure(self, r, params):
        """
        This is the autograd version of the gradient of the logarithm of the wave function w.r.t. the coordinates

        we can optimize this by
        grad_log_fun = grad(
            self._log_wf
        )  # grad of log(wf) due to the property of exponential
        return grad_log_fun(r) * self.wf(r)  # Chain rule to get grad of wf itself like we are doing in the laplacian
        """
        grad_wf = jax.grad(self.wf, argnums=0)
        return grad_wf(r, params)

    def grad_wf(self, r):
        """
        (∇_r) Ψ(r) = ∑_i (∇_r) Ψ(r_i)
        """
        return self.grad_wf_closure(r, self.params)

    # @partial(jax.jit, static_argnums=(0,))
    def laplacian_closure(self, r, params):
        """
        (∇_r)^2 Ψ(r) = ∑_i (∇_r)^2 Ψ(r_i)
        This can be optimized by using the fact that Ψ = exp(FFNN)
        For now we will use the autograd version
        """

        def wrapped_wf(r_):
            return self.wf(r_, params)

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
        return self.laplacian_closure(r, self.params)

    def grads_closure(self, r, *param_values):
        """
        This is the autograd version of the gradient of the logarithm of the wave function w.r.t. the parameters
        Param values is a list of the current parameter values
        we need to differentiate self. wf w.r.t each array in param_values
        # NOTE: this is not analytical gradient. This is the autograd version of the gradient of the logarithm of the wave function w.r.t. the parameters
        # FIXME: maybe we can use analytical expression from backpropagation
        """
        grad_values = []
        grad_fn = jax.grad(self.wf, argnums=1)
        for i in range(len(param_values)):
            grad_values.append(grad_fn(r, param_values)[i])

        return tuple(grad_values)

    def grads(self, r):
        """
        Gradients of the wave function with respect to the neural network parameters.
        """

        # jax.grad will return a function that computes the gradient of wf.
        # This function will expect a parameter input, which in our case are the neural network parameters.

        # The grad_fn function now computes the gradients of the wave function with respect to each parameter of
        # the neural network. We need to pass the current parameters to this function.

        # 'gradients' is now a structure (like a dictionary) holding the gradients with respect to each parameter.

        param_keys = self.params.keys()
        param_values = [self.params.get(key) for key in param_keys]
        grad_values = self.grads_closure(r, *param_values)  # will this change order?

        grads_dict = {key: value for key, value in zip(param_keys, grad_values)}
        # print(grads_dict["W1"].sum())
        return grads_dict

    def pdf(self, r):
        """Probability density function"""
        psi2 = jnp.abs(self.wf(r, self.params)) ** 2

        return psi2

    def logprob_closure(self, r, params):
        """Log probability amplitude"""

        return jnp.log(jnp.abs(self.wf(r, params)) ** 2)

    def logprob(self, r):
        """Log probability amplitude"""

        return self.logprob_closure(r, self.params)

    def compute_sr_matrix(self, expval_grads, grads, shift=1e-4):
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
            grad_value = self.backend.array(
                grad_value
            )  # this should be done outside of the function

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
