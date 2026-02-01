import numpy as np
import pandas as pd
from obspy.geodetics import gps2dist_azimuth

from . import _UP, _DOWN, _UNDEF
from .simulate import MemorylessMarkovChain
from .utils import load_earthquake_catalog

class LocalExternalDisturbance:
    pass

class GlobalExternalDisturbance:
    pass

class PoissonProcessExternalDisturbance(MemorylessMarkovChain):
    """
    If an external disturbance is a Poisson process, then
    it can be modeled as a memoryless Markov chain of Bernoulli trials.

    When the state is _UP, it means that the disturbance has been triggered,
    while _DOWN means no disturbance.
    
    Therefore, the transition probability from _DOWN to _UP
    = \lambda / nmax, where lambda is the rate of the Poisson process,
    and dt is the time step of the simulation.

    The transition probability from _UP to _DOWN = 1, since we assume
    that the disturbance only lasts for one time step.
    """
    param_names = ["rate"]
    param_labels = [r"$r$"]
    truncate_output: bool = False
    default_initial_state: int = _DOWN

    def transition_prob_utd(self, idx, idx_lastchange, dt, cont_up_time):
        return 1.0

    def transition_prob_dtu(self, idx, idx_lastchange, dt, cont_down_time):
        return self.params["rate"] / self.nmax

class TeleseismicActivity(PoissonProcessExternalDisturbance, GlobalExternalDisturbance):
    wave_speed_km_per_dt = np.nan

    def __init__(self, **kwargs):
        if "catalog_path" in kwargs:
            catalog_path = kwargs.pop("catalog_path")
        else:
            catalog_path = None
        super().__init__(**kwargs)

        # Load the earthquake catalog
        if catalog_path is None:
            self.earthquake_catalog = load_earthquake_catalog()
        else:
            self.earthquake_catalog = pd.read_csv(catalog_path)

    def sample_origin_coords_from_file(self):
        """
        Draw a random sample of origin coordinates of teleseismic disturbances
        from a pandas DataFrame containing historical earthquake data.

        Parameters
        ----------
        None

        Returns
        -------
        origins : a tuple
            A sample of (latitude, longitude) tuple.
        """
        sampled_origin = self.earthquake_catalog.sample(n=1)[["latitude", "longitude"]].iloc[0]
        return sampled_origin

    def compute_time_delay_steps(self, origin_coords, components_coords, speed=None):
        """
        Compute the time delay in number of steps for each component
        based on their geographical coordinates.

        We assume that the seismic wave is a surface wave propagating
        at a constant speed.

        Parameters
        ----------
        origin_coords : tuple
            A tuple of (latitude, longitude) for the origin of the disturbance.
        components_coords : dict
            A dictionary where keys are component names and values are
            tuples of (latitude, longitude) for each component.
        speed : float or None, optional
            The speed of the seismic wave in km/[dt]. If None, use
            the default wave speed defined in the class, by default None.

        Returns
        -------
        time_delays : dict
            A dictionary where keys are component names and values are
            the time delay in number of steps for each component.
        """
        if speed is None:
            speed = self.wave_speed_km_per_dt

        time_delays = {}
        for name, coords in components_coords.items():
            distance, _, _ = gps2dist_azimuth(
                origin_coords[0], origin_coords[1],
                coords[0], coords[1]
            )
            time_delay = distance / (speed * 1000) # in whatever unit of dt
            time_delay_steps = int(np.round(time_delay / self.dt))
            time_delays[name] = time_delay_steps
        return time_delays
