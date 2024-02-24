import pickle
import numpy as np
from sbi.inference.base import infer as sbi_infer

from .density import (
    make_kdes_from_simulation,
    make_kdes_from_data,
    make_histograms_from_simulation,
    make_histograms_from_data,
)

class Inference:
    def __init__(
            self,
            simulator,
            prior,
            density_estimator="histogram",
            density_estimator_kwargs={},
            nsample=1000,
            ngridpoint=50,
            grid_range=(1e-2, 1),
        ):
        self.simulator = simulator
        self.prior = prior

        assert density_estimator in ["histogram", "kde"], "Invalid density estimator."
        self.density_estimator = density_estimator
        self.density_estimator_kwargs = density_estimator_kwargs

        self.nsample = nsample
        assert ngridpoint > 0 and type(ngridpoint) is int, "ngridpoint must be a positive integer."
        self.ngridpoint = ngridpoint
        # Set up the grid for evaluation
        self.grid = np.geomspace(*grid_range, num=self.ngridpoint, endpoint=False)
        # Figure out the corresponding "bin edges" for histograms
        self.bin_edges = np.r_[
            grid_range[0],
            0.5*(self.grid[:-1] + self.grid[1:]),
            grid_range[1]
        ]

        if self.density_estimator == "kde":
            self.simulator_for_sbi = lambda simulation_params: np.concatenate([kde(self.grid) for kde in make_kdes_from_simulation(self.simulator, simulation_params, self.nsample, **self.density_estimator_kwargs)])
        else:
            self.simulator_for_sbi = lambda simulation_params: np.concatenate(make_histograms_from_simulation(self.simulator, simulation_params, self.nsample, self.bin_edges))
        self.trained_posterior = None

        def train(
                self,
                method="SNPE",
                nsimulation=5000,
                ncore=1,
            ):
            self.trained_posterior = sbi_infer(
                self.simulator_for_sbi,
                self.prior,
                method=method,
                num_simulations=nsimulation,
                num_workers=ncore,
            )

        def infer(
                self,
                cont_up_times,
                cont_down_times,
                nposterior=10000,
            ):
            """
            Make inferences on the parameters of the duty cycle model.

            Parameters
            ----------
            cont_up_times : array_like
                The contiguous up times.
            cont_down_times : array_like
                The contiguous down times.
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

            if self.density_estimator == "kde":
                obs = np.concatenate(make_kdes_from_data(cont_up_times, cont_down_times, **self.density_estimator_kwargs))
            else:
                obs = np.concatenate(make_histograms_from_data(cont_up_times, cont_down_times, self.bin_edges))

            posterior_samples = self.trained_posterior.sample((nposterior,), x=obs)
            log_probs = self.trained_posterior.log_prob(posterior_samples, x=obs)

            return posterior_samples, log_probs
            