import jax
import jax.numpy as jnp
import numpy as np
from line_profiler import LineProfiler  # noqa
from src.state.utils import advance_PRNG_state
from src.state.utils import State

from .sampler import Sampler


class Metropolis(Sampler):
    def __init__(self, rng, scale, logger=None):

        super().__init__(rng, scale, logger)

    def _step(self, wf, state_batch, seed, batch_size):
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

        # Advance RNG batch_size times
        # create empty array of states of size batch_size

        for i in range(batch_size):
            state = state_batch[i - 1]

            next_gen = advance_PRNG_state(seed, state.delta)
            rng = self._rng(next_gen)

            prev_pos = state.positions
            proposals_pos = rng.normal(loc=prev_pos, scale=self.scale)
            log_unif = np.log(rng.uniform())

            # Compute proposal log density
            logp_proposal = wf.logprob(proposals_pos)
            current_logp = wf.logprob(prev_pos)

            # Metroplis acceptance criterion
            accept = log_unif < logp_proposal - current_logp

            # If accept is True, yield proposal, otherwise keep old state
            new_positions = proposals_pos if accept else prev_pos

            # Create new state
            new_logp = logp_proposal if accept else current_logp
            # new_logp = p_proposal if accept else state.logp
            new_n_accepted = state.n_accepted + accept
            new_delta = state.delta + 1

            state_batch[i].positions = new_positions
            state_batch[i].logp = new_logp
            state_batch[i].n_accepted = new_n_accepted
            state_batch[i].delta = new_delta

        return state_batch

    def jax_step(self, wf, state_batch, seed, batch_size):
        """
        use jax rng to generate random numbers
        """
        self.seed += 1
        # print("seed: ", self.seed)
        rng = jax.random.PRNGKey(self.seed)
        sanity = []

        for i in range(batch_size):

            state = state_batch[i - 1]
            rng, rng_input = jax.random.split(rng)

            center = state.positions

            proposals_pos = (
                jax.random.normal(rng_input, shape=center.shape) * self.scale + center
            )
            sanity.append(proposals_pos)
            logp_proposal = wf.logprob(proposals_pos)

            log_unif = jnp.log(jax.random.uniform(rng_input))
            accept = log_unif < logp_proposal - state.logp

            new_positions = proposals_pos if accept else state.positions
            new_logp = logp_proposal if accept else state.logp
            new_n_accepted = state.n_accepted + accept
            new_delta = state.delta + 1
            state_batch[i].positions = new_positions
            state_batch[i].logp = new_logp
            state_batch[i].n_accepted = new_n_accepted
            state_batch[i].delta = new_delta

        return state_batch

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

    def step(self, wf, state, seed, batch_size=1):

        return self._step(wf, state, seed, batch_size)

    def reset_scale(self, scale):
        self.scale = scale
        print("scale reset to: ", scale)
        return scale

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
            scale *= 2.0
        elif acc_rate > 0.75:
            # increase by double
            scale *= 1.1

        # print("scale after: ", scale)

        self.scale = scale
        return scale  # if scale < max_scale else max_scale
