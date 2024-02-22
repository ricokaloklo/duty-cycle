import numpy as np
from KDEpy import NaiveKDE
import torch
from torch import distributions as dist

from .simulate import *

# From https://github.com/pytorch/pytorch/issues/11412
class LogUniform(dist.TransformedDistribution):
    def __init__(self, lower_bound, upper_bound):
        super(LogUniform, self).__init__(
            dist.Uniform(
                lower_bound.log(),
                upper_bound.log()
            ),
            dist.ExpTransform()
        )

def make_density_estimator_from_data(
        cont_up_times,
        cont_down_times,
        kernel="box",
        bw="silverman",
):
    """
    Make a density estimator for the contiguous up and down times
    of a detector using data.

    Parameters
    ----------
    cont_up_times : array-like
        The contiguous up times.
    cont_down_times : array-like
        The contiguous down times.
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
    # Make the density estimators
    cont_up_times_kde = NaiveKDE(bw=bw, kernel=kernel).fit(cont_up_times)
    cont_down_times_kde = NaiveKDE(bw=bw, kernel=kernel).fit(cont_down_times)

    return cont_up_times_kde, cont_down_times_kde

def make_density_estimator_from_simulation(
    simulator,
    simulation_params,
    nsample=1000,
    kernel="box",
    bw="silverman",
):
    """
    Make a density estimator for the contiguous up and down times
    of a detector using simulated data.

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
    return make_density_estimator_from_data(
        cont_up_times,
        cont_down_times,
        kernel=kernel,
        bw=bw,
    )
