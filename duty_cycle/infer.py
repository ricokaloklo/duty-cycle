import gzip
import multiprocessing as mp
import os
import pickle
import numpy as np
import torch
import tqdm
from sbi.inference import infer as sbi_infer
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import PermutationInvariantEmbedding

from .density import (
    make_kdes_from_simulation,
    make_kdes_from_data,
    make_histograms_from_simulation,
    make_histograms_from_data,
)
from .simulate_network import NetworkSimulator
from .visualize import visualize_posterior


class SimulationBasedInference:
    @classmethod
    def load_from_file(cls, filename):
        open_fn = gzip.open if str(filename).endswith(".gz") else open
        with open_fn(filename, "rb") as f:
            return pickle.load(f)

    def save_to_file(self, filename):
        open_fn = gzip.open if str(filename).endswith(".gz") else open
        with open_fn(filename, "wb") as f:
            pickle.dump(self, f)

    def plot_corner(
        self,
        posterior_samples,
        filename="corner.png",
        use_tex=False,
        truths=None,
    ):
        """
        Make a corner plot of the posterior samples.

        Parameters
        ----------
        posterior_samples : array_like
            The posterior samples.
        filename : str, optional
            The name of the file to save the corner plot to. Default is "corner.png".
        use_tex : bool, optional
            Whether to use TeX for the labels. Default is False.
        truths : array_like, optional
            The true values of the parameters. Default is None.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The corner plot.
        """
        fig = visualize_posterior(
            posterior_samples.numpy(),
            labels=self.simulator.param_labels,
            truths=truths,
            use_tex=use_tex,
        )

        if filename is not None:
            fig.savefig(filename)

        return fig

class SummaryStatisticInference(SimulationBasedInference):
    # NOTE: This does not work for NetworkSimulator. Use EmbeddingNetworkInference instead.
    def __init__(
            self,
            simulator,
            prior,
            density_estimator="kde",
            density_estimator_kwargs={},
            nsample=1000,
            ngridpoint=50,
            grid_range=(1e-2, 1),
            grid_spacing="log",
        ):
        """
        Inference using summary statistics (histograms or KDEs) of the continuous up and down times.

        Parameters
        ----------
        simulator : Simulator
            The simulator to use for generating data.
        prior : torch.distributions.Distribution
            The prior distribution over the parameters.
        density_estimator : str, optional
            The density estimator to use for the summary statistics. Options are "histogram" or "kde". Default is "kde".
        density_estimator_kwargs : dict, optional
            The keyword arguments to pass to the density estimator. Default is {}.
        nsample : int, optional
            The number of samples to draw from the simulator for each parameter set. Default is 1000.
        ngridpoint : int, optional
            The number of grid points to use for the summary statistics. Default is 50.
        grid_range : tuple, optional
            The range of the grid for the summary statistics. Default is (1e-2, 1).
        grid_spacing : str, optional
            The spacing of the grid for the summary statistics. Options are "linear" or "log". Default is "log".
        """
        self.simulator = simulator
        self.simulator.truncate_output = True # To avoid artifact in the summary statistics due to truncation
        assert not isinstance(simulator, NetworkSimulator), "SummaryStatisticInference does not work for NetworkSimulator. Use EmbeddingNetworkInference instead."
        self.prior = prior

        assert density_estimator in ["histogram", "kde"], "Invalid density estimator."
        self.density_estimator = density_estimator
        self.density_estimator_kwargs = density_estimator_kwargs

        self.nsample = nsample
        assert ngridpoint > 0 and type(ngridpoint) is int, "ngridpoint must be a positive integer."
        self.ngridpoint = ngridpoint

        # Set up the grid for evaluation
        assert grid_spacing in ["linear", "log"], "Invalid grid spacing."

        if grid_spacing == "linear":
            # Figure out the bin edges for histograms first
            self.bin_edges = np.histogram_bin_edges([], bins=self.ngridpoint, range=grid_range)
            # Grid points are the midpoints of the bin edges
            self.grid = (self.bin_edges[1:] + self.bin_edges[:-1])/2
        else:
            self.grid = np.geomspace(*grid_range, num=self.ngridpoint, endpoint=False)
            # Figure out the corresponding "bin edges" for histograms
            self.bin_edges = np.r_[
                grid_range[0],
                0.5*(self.grid[:-1] + self.grid[1:]),
                grid_range[1]
            ]

        self.trained_posterior = None

    def train(
            self,
            method="SNPE",
            nsimulation=5000,
            ncore=1,
        ):
        """
        Train the posterior using the simulator.

        Parameters
        ----------
        method : str, optional
            The inference method to use. Default is "SNPE".
        nsimulation : int, optional
            The number of simulations to use for training the posterior.
        ncore : int, optional
            The number of cores to use for training the posterior.
        """

        def simulator_for_sbi(simulation_params):
            if self.density_estimator == "kde":
                return np.concatenate([kde(self.grid) for kde in make_kdes_from_simulation(self.simulator, simulation_params, self.nsample, **self.density_estimator_kwargs)])
            else:
                return np.concatenate(make_histograms_from_simulation(self.simulator, simulation_params, self.nsample, self.bin_edges))

        self.trained_posterior = sbi_infer(
            simulator_for_sbi,
            self.prior,
            method=method,
            num_simulations=nsimulation,
            num_workers=ncore,
        )

    def infer(
            self,
            observations,
            nposterior=10000,
        ):
        """
        Make inferences on the parameters of the duty cycle model.

        Parameters
        ----------
        observations : tuple of array_like
            The observed data, as a tuple of (cont_up_times, cont_down_times).
        nposterior : int, optional
            The number of posterior samples to draw.
        
        Returns
        -------
        posterior_samples : array_like
            The posterior samples.
        log_probs : array_like
            The log probabilities of the posterior samples.
        """
        if self.trained_posterior is None:
            raise ValueError("The posterior has not been trained yet.")
        
        assert nposterior > 0 and type(nposterior) is int, "nposterior must be a positive integer."

        cont_up_times, cont_down_times = observations

        if self.density_estimator == "kde":
            obs = np.concatenate([kde(self.grid) for kde in make_kdes_from_data(cont_up_times, cont_down_times, **self.density_estimator_kwargs)])
        else:
            obs = np.concatenate(make_histograms_from_data(cont_up_times, cont_down_times, self.bin_edges))

        # NOTE Our prior is bounded
        posterior_samples = self.trained_posterior.sample((nposterior,), x=obs)
        log_probs = self.trained_posterior.log_prob(posterior_samples, x=obs)

        return posterior_samples, log_probs

class EmbeddingNetworkInference(SimulationBasedInference):
    pass
