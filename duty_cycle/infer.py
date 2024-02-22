import numpy as np
from KDEpy import NaiveKDE

from .simulate import *

def make_density_estimator_for_cont_up_down_times(
    simulator,
    simulation_params,
    nsample=1000,
    kernel="box",
    bw="silverman",
):
    """
    Make a density estimator for the contiguous up and down times of a detector.

    Parameters
    ----------
    simulator : Simulator
        The simulator object.
    simulation_params : array-like
        The parameters of the duty cycle model.
    nsample : int, optional
        The minimum number of samples to simulate.
    kernel : str, optional
        The kernel to use for the density estimator.
    bw : str, optional
        The bandwidth to use for the density estimator.

    Returns
    -------
    cont_up_times_kde : NaiveKDE
        The density estimator for the contiguous up times.
    cont_down_times_kde : NaiveKDE
        The density estimator for the contiguous down times.
    """
    # Simulate the contiguous up and down times
    cont_up_times, cont_down_times = simulator.simulate_cont_up_down_times(
        simulation_params, nsample=nsample
    )

    # Make the density estimators
    cont_up_times_kde = NaiveKDE(bw=bw, kernel=kernel).fit(cont_up_times)
    cont_down_times_kde = NaiveKDE(bw=bw, kernel=kernel).fit(cont_down_times)

    return cont_up_times_kde, cont_down_times_kde
