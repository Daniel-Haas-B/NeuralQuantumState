import numpy as np
from nqs.utils import advance_PRNG_state
from nqs.utils import State

from .sampler import Sampler


class Metropolis(Sampler):
    def __init__(self, rng, scale, logger):
        super().__init__(rng, scale, logger)
        self.test_p_proposals = []
        self.test_p_unif = []
        self.test_p_state = []

    def _step(self, wf, state, seed):
        """One step of the random walk Metropolis algorithm

        Parameters
        ----------
        state : nqs.State
            Current state of the system. See state.py

        scale : float
            Scale of proposal distribution. Default: 0.5

        Returns
        -------
        new_state : nqs.State
            The updated state of the system.
        """

        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)

        # Sample proposal positions, i.e., move walkers
        proposals = rng.normal(loc=state.positions, scale=self.scale)
        # Sample log uniform rvs
        log_unif = np.log(rng.random())

        # Compute proposal log density

        logp_proposal = wf.logprob(proposals)

        # Metroplis acceptance criterion
        accept = log_unif < logp_proposal - state.logp

        # If accept is True, yield proposal, otherwise keep old state
        new_positions = proposals if accept else state.positions

        # Create new state
        new_logp = wf.logprob(new_positions)
        new_n_accepted = state.n_accepted + accept
        new_delta = state.delta + 1
        new_state = State(new_positions, new_logp, new_n_accepted, new_delta)

        return new_state

    def _fixed_step(self, wf, state, seed, fixed_index=0):
        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)

        # Sample proposal positions, i.e., move walkers
        positions = state.positions
        proposals = rng.normal(loc=positions, scale=self.scale)
        proposals[fixed_index] = positions[fixed_index]  # Fix one particle
        # Sample log uniform rvs
        log_unif = np.log(rng.random())

        # Compute proposal log density
        logp_proposal = wf.logprob(proposals)

        # Metroplis acceptance criterion
        accept = log_unif < logp_proposal - state.logp

        # If accept is True, yield proposal, otherwise keep old state
        new_positions = proposals if accept else state.positions

        # Create new state
        new_logp = wf.logprob(new_positions)
        new_n_accepted = state.n_accepted + accept
        new_delta = state.delta + 1
        new_state = State(new_positions, new_logp, new_n_accepted, new_delta)

        return new_state

    def step(self, wf, state, seed):
        return self._step(wf, state, seed)

    def tune_scale(scale, acc_rate):
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
                        >0.5          x 1.1
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
        if acc_rate < 0.001:
            # reduce by 90 percent
            return scale * 0.1
        elif acc_rate < 0.05:
            # reduce by 50 percent
            scale *= 0.5
        elif acc_rate < 0.2:
            # reduce by ten percent
            scale *= 0.9
        elif acc_rate > 0.5:
            # increase by ten percent
            scale *= 1.1
        elif acc_rate > 0.75:
            # increase by double
            scale *= 2.0
        elif acc_rate > 0.95:
            # increase by factor of ten
            scale *= 10.0

        return scale
