import pickle
import numpy as np
import torch
from sbi.inference import infer as sbi_infer
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

from .density import (
    make_kdes_from_simulation,
    make_kdes_from_data,
    make_histograms_from_simulation,
    make_histograms_from_data,
)
from .simulate_network import NetworkSimulator
from .utils import MultivariateTimeSeriesTransformerEmbedding
from .visualize import visualize_posterior

class SimulationBasedInference:
    @classmethod
    def load_from_file(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def save_to_file(self, filename):
        with open(filename, "wb") as f:
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
    def __init__(
            self,
            simulator,
            prior,
            embedding_net_kwargs={},
        ):
        self.simulator = simulator
        self.simulator.truncate_output = False # Such that the simulator always outputs the same length of time series
        self.prior = prior
        self.embedding_net_kwargs = embedding_net_kwargs
        self.embedding_net_kwargs = dict(
            vit=False,                # time-series mode
            is_causal=True,           # Causal mask for time series
            num_hidden_layers=3,      # Number of transformer layers
            num_attention_heads=4,    # Number of attention heads
            num_key_value_heads=4,    # Number of key-value heads (can be different from num_attention_heads)
            feature_space_dim=64,     # Internal transformer dimension
            final_emb_dimension=32,   # Final embedding dimension passed to the density estimator
        )
        self.embedding_net_kwargs.update(embedding_net_kwargs)

        self.ncomponent = 1
        if isinstance(simulator, NetworkSimulator):
            self.ncomponent = len(simulator.components)

        self.embedding_net = MultivariateTimeSeriesTransformerEmbedding(
            input_dim=self.ncomponent,
            transformer_cfg=self.embedding_net_kwargs,
        )

        self.trained_posterior = None

    def train(
            self,
            method="SNPE",
            nsimulation=5000,
            ncore=1,
            device="cpu",
        ):
        # Move embedding_net and prior to device
        self.embedding_net.to(device)
        self.prior.to(device)

        # Sample parameters from the prior and simulate data
        thetas = self.prior.sample((nsimulation,)).to(device)
        xs_list = []
        for theta in thetas:
            bit_ts_dict = self.simulator.simulate_duty_cycle(theta)
            component_bit_ts = torch.stack([torch.Tensor(bit_ts_dict[name]) for name in self.simulator.components.keys()], dim=-1).to(device)
            xs_list.append(component_bit_ts)

        # Stack into a single tensor: (nsimulation, T, ncomponent)
        xs = torch.stack(xs_list, dim=0).to(device)

        if method == "SNPE":
            density_estimator = posterior_nn(
                model="maf",
                embedding_net=self.embedding_net,
                z_score_x="none",
                z_score_y="none",
            )

            if device == "cpu":
                self.trained_posterior = NPE(prior=self.prior, density_estimator=density_estimator, device="cpu").append_simulations(thetas, xs).train()
            else:
                self.trained_posterior = NPE(prior=self.prior, density_estimator=density_estimator, device=device).append_simulations(thetas, xs).train()
        else:
            raise NotImplementedError(f"Method {method} not implemented yet.")

    def infer(
            self,
            observations,
            device="cpu",
            nposterior=10000,
        ):
        x_obs = torch.Tensor(observations).unsqueeze(0).to(device) # (1, T, ncomponent)
        posterior_samples = self.trained_posterior.sample((nposterior,), x_obs).detach()
        log_probs = self.trained_posterior.log_prob(posterior_samples, x_obs).detach()

        return posterior_samples.cpu().numpy(), log_probs.cpu().numpy()
