import numpy as np

from .sampler import Sampler
from nqs.state.utils import advance_PRNG_state


class MetropolisHastings(Sampler):
    def __init__(self, rng, scale, logger=None):
        super().__init__(rng, scale, logger)

    def _step(self, wf, state_batch, seed, batch_size):
        """One step of the Langevin Metropolis-Hastings algorithm

        Parameters
        ----------
        state : State
            Current state of the system. See state.py
        alpha :
            Variational parameter
        D : float
            Diffusion constant. Default: 0.5
        dt : float
            Scale of proposal distribution. Default: 1.0
        """

        # Precompute
        dt = self.scale**2
        Ddt = 0.5 * dt
        quarterDdt = 1 / (4 * Ddt)

        next_gens = [advance_PRNG_state(seed, state.delta) for state in state_batch]
        rngs = [self._rng(next_gen) for next_gen in next_gens]

        for i in range(batch_size):
            state = state_batch[i - 1]
            rng = rngs[i]

            # Compute drift force at current positions
            F = self.hamiltonian.drift_force(wf, state.positions)
            prev_sign = state.sign
            proposals = rng.normal(loc=state.positions, scale=self.scale) + F * Ddt
            # Compute proposal log density
            logp_proposal = wf.logprob(proposals)

            # Green's function conditioned on proposals
            F_prop = self.hamiltonian.drift_force(wf, proposals)
            G_prop = -((state.positions - proposals - Ddt * F_prop) ** 2) * quarterDdt

            # Green's function conditioned on current positions
            G_cur = -((proposals - state.positions - Ddt * F) ** 2) * quarterDdt

            # Metroplis-Hastings ratio
            ratio = logp_proposal + np.sum(G_prop) - state.logp - np.sum(G_cur)

            # Sample log uniform rvs
            log_unif = np.log(rng.random())

            # Metroplis acceptance criterion
            accept = log_unif < ratio

            # If accept is True, yield proposal, otherwise keep old state
            new_positions = proposals if accept else state.positions
            new_sign = wf.get_sign() if accept else prev_sign

            # Create new state
            new_logp = logp_proposal if accept else state.logp
            new_n_accepted = state.n_accepted + accept
            new_delta = state.delta + 1

            state_batch[i].positions = new_positions
            state_batch[i].logp = new_logp
            state_batch[i].n_accepted = new_n_accepted
            state_batch[i].delta = new_delta
            state_batch[i].sign = new_sign

        return state_batch

    def tune_scale(self, scale, acc_rate):
        """Proposal scale lookup table. (Original)

        Aims to obtain an acceptance rate between 20-50%.

        Retrieved from the source code of PyMC [1].

        Tunes the scaling parameter for the proposal distribution
        according to the acceptance rate over the last tune_interval:

                        Rate    Variance adaptation
                        ----    -------------------
                        <0.001        x 0.1
                        <0.05         x 0.5
                        <0.2          x 0.9
                        <0.5          x 0.95
                        >0.8          x 1.4
                        >0.75         x 2
                        >0.95         x 10

        References
        ----------
        [1] https://github.com/pymc-devs/pymc/blob/main/pymc/step_methods/metropolis.py#L263

        Arguments
        ---------
        scale : float
            Scale of the proposal distribution
        acc_rate : float
            Acceptance rate of the last tuning interval

        Returns
        -------
        scale : float
            Updated scale parameter
        """
        # print("acc_rate: ", acc_rate)
        # print("scale before: ", scale)
        if acc_rate < 0.001:
            # reduce by 90 percent
            scale *= 0.1
        elif acc_rate < 0.05:
            # reduce by 50 percent
            scale *= 0.5
        elif acc_rate < 0.2:
            # reduce by ten percent
            scale *= 0.9
        elif acc_rate < 0.5:
            # reduce by five percent
            scale *= 0.95
        elif acc_rate > 0.95:
            # increase by factor of ten
            scale *= 4.0
        elif acc_rate > 0.8:
            # increase by ten percent
            scale *= 2
        elif acc_rate > 0.75:
            # increase by double
            scale *= 1.1

        # print("scale after: ", scale)

        self.scale = scale
        return scale

    def step(self, wf, state, seed, batch_size):
        return self._step(wf, state, seed, batch_size)

    def reset_scale(self, scale):
        self.scale = scale
        return scale
