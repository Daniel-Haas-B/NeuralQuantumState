# import pytest
from src.nqs.utils import State
from src.samplers.metropolis import Metropolis
from src.samplers.sampler import Sampler


# Mocks for testing
class MockRBM:
    def local_energy(self, *args, **kwargs):
        return 1.0

    def logprob(self, *args, **kwargs):
        return 0.5


def test_sampler_init():
    rbm = MockRBM()
    sampler = Sampler(rbm, None)
    assert sampler.scale is None


def test_metropolis_init():
    rbm = MockRBM()
    metropolis = Metropolis(rbm, None, 0.5)
    assert metropolis.scale == 0.5


def test_metropolis_step():
    rbm = MockRBM()
    metropolis = Metropolis(rbm, lambda x: 0.5, 0.5)  # Using lambda for RNG
    state = State(None, 0.5, 0, 0)
    new_state = metropolis.step(state, None, None, None, None)
    assert isinstance(new_state, State)


# WIP
# def test_metropolis_tune_scale():
#    scale = Metropolis.tune_scale(0.5, 0.05)
#    assert scale == 0.5 * 0.5
