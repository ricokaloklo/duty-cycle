import numpy as np
import pandas as pd
from obspy.geodetics import gps2dist_azimuth

from . import _UP, _DOWN, _UNDEF
from .simulate import MemorylessMarkovChain
from .utils import load_earthquake_catalog

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
    truncate_output:bool = False

    def transition_prob_utd(self, idx, idx_lastup, dt, cont_up_time):
        return 1.0

    def transition_prob_dtu(self, idx, idx_lastup, dt, cont_down_time):
        return self.params["rate"] / self.nmax

class TeleseismicActivity(PoissonProcessExternalDisturbance):
    def sample_origin_coords_from_file(self, filepath=None):
        """
        Draw a random sample of origin coordinates of teleseismic disturbances
        from a CSV file containing historical earthquake data.

        Parameters
        ----------
        filepath : str, optional
            Path to the CSV file. If None, the default earthquake catalog is used.

        Returns
        -------
        origins : a tuple
            A sample of (latitude, longitude) tuple.
        """
        if filepath is None:
            df = load_earthquake_catalog()
        else:
            df = pd.read_csv(filepath)

        origins = list(zip(df['latitude'], df['longitude']))
        sampled_origin = origins[np.random.randint(0, len(origins))]
        return sampled_origin

    def compute_time_delay_steps(self, origin_coords, components_coords, speed):
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
        speed : float
            The speed of the seismic wave in km/[dt].

        Returns
        -------
        time_delays : dict
            A dictionary where keys are component names and values are
            the time delay in number of steps for each component.
        """
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
