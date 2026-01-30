import numpy as np

from .simulate import Simulator, MemorylessMarkovChain

class GlobalExternalDisturbance(MemorylessMarkovChain):
    pass

class ScheduledMaintenance(Simulator):
    pass

class TeleseismicActivity(GlobalExternalDisturbance):
    """
    NOTE: For this one, we need to have a way to compute time delays
    between detectors based on their geographical locations. But if
    the time delays are negligible compared to the time resolution
    of the simulation, we can ignore them.
    """
    pass
