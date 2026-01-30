import numpy as np

from . import _UP, _DOWN, _UNDEF
from .simulate import Simulator, MemorylessMarkovChain

class LocalExternalDisturbance(MemorylessMarkovChain):
    """
    If a local external disturbance is a Poisson process, then
    it can be modeled as a memoryless Markov chain of Bernoulli trials.

    When the state is _UP, it means that the disturbance has been triggered,
    while _DOWN means no disturbance.
    
    Therefore, the transition probability from _DOWN to _UP
    = \lambda / nmax, where lambda is the rate of the Poisson process,
    and dt is the time step of the simulation.

    The transition probability from _UP to _DOWN = 1, since we assume
    that the disturbance only lasts for one time step.
    """
    pass

class GlobalExternalDisturbance(MemorylessMarkovChain):
    pass

class TeleseismicActivity(GlobalExternalDisturbance):
    """
    NOTE: For this one, we need to have a way to compute time delays
    between detectors based on their geographical locations. But if
    the time delays are negligible compared to the time resolution
    of the simulation, we can ignore them.
    """
    pass
