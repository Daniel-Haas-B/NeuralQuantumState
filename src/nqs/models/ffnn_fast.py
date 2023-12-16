import jax
import jax.numpy as jnp
import numpy as np
from nqs.utils import Parameter
from nqs.utils import State
from scipy.special import expit

# from jax import grad

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class FFNNFAST:
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

        input_size_0 = self._N * self._dim
        output_size_0 = self._layer_sizes[0]
        limit_0 = np.sqrt(6 / (input_size_0 + output_size_0))  # * 0.1
        # do not change this order
        self.params.set(
            "W0",
            jnp.array(
                np.array(rng.uniform(-limit_0, limit_0, (input_size_0, output_size_0))),
                # rng.standard_normal(size=(input_size_0, output_size_0))
            ),
        )
        self.params.set("b0", jnp.zeros((output_size_0,)))

        for i in range(1, len(self._layer_sizes)):
            # The number of units in the layers
            input_size = self._layer_sizes[i - 1]
            output_size = self._layer_sizes[i]

            # Glorot initialization, considering the size of layers
            limit = jnp.sqrt(6 / (input_size + output_size))  # * 0.1

            # Initialize weights and biases
            self.params.set(
                f"W{i}",
                np.array(rng.uniform(-limit, limit, (input_size, output_size))),
                # rng.standard_normal(size=(input_size, output_size)),
            )
            self.params.set(
                f"b{i}",
                jnp.zeros((output_size,))
                # jnp.array(rng.standard_normal(size=(output_size,)))
            )

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

    def activation(self, activation_str):
        match activation_str:
            case "tanh":
                return jnp.tanh
            case "sigmoid":
                return jax.nn.sigmoid
            case "relu":
                return jax.nn.relu
            case "softplus":
                return jax.nn.softplus
            case "gelu":
                return jax.nn.gelu
            case "linear":
                return lambda x: x
            case "exp":
                return jnp.exp
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

    def _jit_functions(self):
        functions_to_jit = [
            "grad_wf_closure",
            "laplacian_closure",
            "grads_closure",
            "forward",
            "wf",
            "grad_wf_closure",
            "grads_closure",
            "laplacian_closure",
        ]
        for func_name in functions_to_jit:
            setattr(self, func_name, jax.jit(getattr(self, func_name)))
        return self

    def forward(self, x, params):
        # Custom Parameter instance
        for i in range(0, len(self._layer_sizes)):
            x = params.get(f"W{i}").T @ x + params.get(f"b{i}")
            x = self.activation(self._activations[i])(x)
        return x

    def log_wf(self, r, params):
        """Compute the logarithm of the wave function from the neural network output."""
        return self.forward(r, params).sum(axis=0)

    def wf(self, r, params):
        """Compute the wave function from the neural network output.
        this is actually
        """
        return self.log_wf(r, params)

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

    def grads_closure(self, r, params):
        grad_fn = jax.grad(self.wf, argnums=1)
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

    def logprob_closure(self, r, params):
        """Log probability amplitude
        that is log(|psi|^2).
        In our case, psi is real, so this is 2 log(|psi|) = 2 * forward(r, params) ?
        """

        return 2.0 * self.log_wf(r, params)

    def logprob(self, r):
        """Log probability amplitude"""

        return self.logprob_closure(r, self.params)

    def compute_sr_matrix(self, expval_grads, grads, shift=1e-5):
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
