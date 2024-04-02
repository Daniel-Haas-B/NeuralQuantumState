import matplotlib.pyplot as plt
import numpy as np

seed = 42
nsamples = 10000


def advance_state_philox(seed, delta):
    return np.random.Philox(seed).advance(delta)


def advance_state_pcg64(seed, delta):
    return np.random.PCG64(seed).advance(delta)


pcg_positions = np.zeros((nsamples, 4))
philox_positions = np.zeros((nsamples, 4))


def log_ansatz(x):
    return -0.5 * np.sum(x**2)


def logprob(x):
    return 2 * log_ansatz(x)


for i in range(nsamples):
    delta = i

    next_gen_phil = advance_state_philox(seed, delta)
    next_gen_pcg = advance_state_pcg64(seed, delta)

    rng_pcg = np.random.default_rng(next_gen_pcg)
    rng_phil = np.random.default_rng(next_gen_phil)
    # generate normal distribution of x and y

    prev_pos_pcg = pcg_positions[-1]
    proposal_pcg = rng_pcg.normal(loc=prev_pos_pcg, scale=1)

    prev_pos_phil = philox_positions[-1]
    proposal_phil = rng_phil.normal(loc=prev_pos_phil, scale=1)

    # acceptance ratio
    log_unif_pcg = np.log(rng_pcg.uniform())
    logprob_proposal_pcg = logprob(proposal_pcg)
    logprob_current_pcg = logprob(prev_pos_pcg)

    log_unif_phil = np.log(rng_phil.uniform())
    logprob_proposal_phil = logprob(proposal_phil)
    logprob_current_phil = logprob(prev_pos_phil)

    accept_pcg = log_unif_pcg < logprob_proposal_pcg - logprob_current_pcg
    accept_phil = log_unif_phil < logprob_proposal_phil - logprob_current_phil

    pcg_positions[i] = proposal_pcg if accept_pcg else prev_pos_pcg
    philox_positions[i] = proposal_phil if accept_phil else prev_pos_phil


plt.figure(figsize=(10, 10))

plt.scatter(pcg_positions[:, 0], pcg_positions[:, 1], label="PCG64")
plt.scatter(philox_positions[:, 0], philox_positions[:, 1], label="Philox")


plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.legend()
plt.show()
