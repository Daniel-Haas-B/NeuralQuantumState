import numpy as np
from nqs.utils import advance_PRNG_state
from nqs.utils import State

from .sampler import Sampler


class MetroHastings(Sampler):
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
        dt = self._scale**2
        Ddt = 0.5 * dt
        quarterDdt = 1 / (4 * Ddt)

        for i in range(batch_size):
            state = state_batch[i - 1]

            sys_size = state.positions.shape

            # Advance RNG
            next_gen = advance_PRNG_state(seed, state.delta)
            rng = self._rng(next_gen)

            # Compute drift force at current positions
            F = self.hamiltonian.drift_force(wf, state.positions)
            print(F)
            exit()
            # Sample proposal positions, i.e., move walkers
            proposals = (
                state.positions
                + F * Ddt
                + rng.normal(loc=0, scale=self._scale, size=sys_size)
            )

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

            # Create new state
            new_logp = wf.logprob(new_positions) if accept else state.logp
            new_n_accepted = state.n_accepted + accept
            new_delta = state.delta + 1

            state_batch[i] = State(
                new_positions, new_logp, new_n_accepted, new_delta
            )  # could be slow

        return state_batch

    def tune_scale(self, scale, acc_rate):
        """Proposal dt (squared scale for importance sampler) lookup table.

        Aims to obtain an acceptance rate between 40-80%.

        Arguments
        ---------
        dt : float
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
        elif acc_rate < 0.4:
            # reduce by ten percent
            scale *= 0.9
        elif acc_rate > 0.8:
            # increase by ten percent
            scale *= 1.1
        elif acc_rate > 0.75:
            # increase by double
            scale *= 2.0
        elif acc_rate > 0.95:
            # increase by factor of ten
            scale *= 10.0

        return scale

    def step(self, wf, state, seed, batch_size):
        return self._step(wf, state, seed, batch_size)
