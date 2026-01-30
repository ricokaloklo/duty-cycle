import numpy as np

class ExternalDisturbance:
    pass

class GlobalExternalDisturbance(ExternalDisturbance):
    pass

class ScheduledMaintenance(ExternalDisturbance):
    """
    NOTE: It requires the object to have knowledge of
    the 'absolute' time in the simulation.
    """
    pass

class TeleseismicActivity(GlobalExternalDisturbance):
    """
    NOTE: For this one, we need to have a way to compute time delays
    between detectors based on their geographical locations. But if
    the time delays are negligible compared to the time resolution
    of the simulation, we can ignore them.
    """
    pass
