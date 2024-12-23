from KDEpy import NaiveKDE

from .simulate import *

def make_kdes_from_data(
        cont_up_times,
        cont_down_times,
        kernel="box",
        bw="silverman",
):
    """
    Make kernel density estimators for the contiguous up and down times
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

def make_kdes_from_simulation(
    simulator,
    simulation_params,
    nsample=1000,
    kernel="box",
    bw="silverman",
):
    """
    Make kernel density estimators for the contiguous up and down times
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
    return make_kdes_from_data(
        cont_up_times,
        cont_down_times,
        kernel=kernel,
        bw=bw,
    )

def make_histograms_from_data(
    cont_up_times,
    cont_down_times,
    bin_edges,
):
    """
    Make normalized histograms for the continuous up and down times
    of a detector using data.

    Parameters
    ----------
    cont_up_times : array-like
        The contiguous up times.
    cont_down_times : array-like
        The contiguous down times.
    bin_edges : array-like
        The bin edges to use for the histograms.

    Returns
    -------
    cont_up_times_hist : array
        The normalized histogram for the contiguous up times.
    cont_down_times_hist : array
        The normalized histogram for the contiguous down times.
    """
    cont_up_time_hist_heights, _ = np.histogram(cont_up_times, bins=bin_edges, density=True)
    cont_down_time_hist_heights, _ = np.histogram(cont_down_times, bins=bin_edges, density=True)

    return cont_up_time_hist_heights, cont_down_time_hist_heights

def make_histograms_from_simulation(
    simulator,
    simulation_params,
    nsample=1000,
    bin_edges=None,
):
    """
    Make normalized histograms for the continuous up and down times
    of a detector using simulated data.

    Parameters
    ----------
    simulator : Simulator
        The simulator object.
    simulation_params : array-like
        The parameters of the duty cycle model.
    nsample : int, optional
        The minimum number of samples to simulate.
    bin_edges : array-like, optional
        The bin edges to use for the histograms.

    Returns
    -------
    cont_up_times_hist : array
        The normalized histogram for the contiguous up times.
    cont_down_times_hist : array
        The normalized histogram for the contiguous down times.
    """
    # Simulate the contiguous up and down times
    cont_up_times, cont_down_times = simulator.simulate_cont_up_down_times(
        simulation_params, nsample=nsample
    )

    # Make the density estimators
    return make_histograms_from_data(
        cont_up_times,
        cont_down_times,
        bin_edges=bin_edges,
    )