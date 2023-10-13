import numpy as np
from nqs.utils import Parameter
from nqs.utils import State

from .base_rbm import BaseRBM


class RBM(BaseRBM):
    def __init__(
        self,
        nparticles,
        dim,
        nhidden=1,
        factor=1.0,  # not sure about this value
        sigma2=1.0,
        rng=None,
        log=False,
        logger=None,
        logger_level="INFO",
        backend="numpy",
    ):
        """RBM Neural Network Quantum State"""
        super().__init__(factor, sigma2, backend=backend)

        self._N = nparticles
        self._dim = dim
        self._nvisible = self._N * self._dim
        self._nhidden = nhidden
        self.logger = logger

        r = rng.standard_normal(size=self._nvisible)

        # Initialize visible bias
        v_bias = rng.standard_normal(size=self._nvisible) * 0.01
        h_bias = rng.standard_normal(size=self._nhidden) * 0.01
        kernel = rng.standard_normal(size=(self._nvisible, self._nhidden))
        kernel *= np.sqrt(1 / self._nvisible)

        self.params = Parameter()
        self.params.set(["v_bias", "h_bias", "kernel"], [v_bias, h_bias, kernel])
        self.log = log
        logp = self.logprob(r)
        self.state = State(r, logp, 0, 0)

        if self.log:
            neuron_str = "neurons" if self._nhidden > 1 else "neuron"
            msg = (
                f"Neural Network Quantum State initialized as RBM with "
                f"{self._nhidden} hidden {neuron_str}"
            )
            self.logger.info(msg)

    # def tune(
    #     self,
    #     tune_iter=20_000,
    #     tune_interval=500,
    #     early_stop=False,  # set to True later
    #     rtol=1e-05,
    #     atol=1e-08,
    #     seed=None,
    #     mcmc_alg=None,
    # ):
    #     """
    #     !! BROKEN NOW due to self.scale
    #     Tune proposal scale so that the acceptance rate is around 0.5.
    #     """

    #     state = self.state
    #     v_bias, h_bias, kernel = self.wf.params.get(["v_bias", "h_bias", "kernel"])

    #     scale = self.scale

    #     if mcmc_alg is not None:
    #         self._sampler = Sampler(self.mcmc_alg, self.rbm, self.rng, self._log)

    #     # Used to throw warnings if tuned alg mismatch chosen alg
    #     # in other procedures
    #     self._tuned_mcmc_alg = self.mcmc_alg

    #     # Config
    #     # did_early_stop = False
    #     seed_seq = generate_seed_sequence(seed, 1)[0]

    #     # Reset n_accepted
    #     state = State(state.positions, state.logp, 0, state.delta)

    #     if self._log:
    #         t_range = tqdm(
    #             range(tune_iter),
    #             desc="[Tuning progress]",
    #             position=0,
    #             leave=True,
    #             colour="green",
    #         )
    #     else:
    #         t_range = range(tune_iter)

    #     steps_before_tune = tune_interval

    #     for i in t_range:
    #         state = self._sampler.step(state, v_bias, h_bias, kernel, seed_seq)
    #         steps_before_tune -= 1

    #         if steps_before_tune == 0:
    #             # Tune proposal scale
    #             old_scale = scale
    #             accept_rate = state.n_accepted / tune_interval
    #             scale = self._sampler.tune_scale(old_scale, accept_rate)

    #             # Reset
    #             steps_before_tune = tune_interval
    #             state = State(state.positions, state.logp, 0, state.delta)

    #     # Update shared values
    #     self.state = state
    #     self.wf.params.set(["v_bias", "h_bias", "kernel"], [v_bias, h_bias, kernel])
    #     self.scale = scale
    #     self._is_tuned_ = True
