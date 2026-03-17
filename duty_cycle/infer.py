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
from .utils import (
    FlattenedTrialTimeSeriesEmbedding,
    RunLengthEventTransformerEmbedding,
)
from .visualize import visualize_posterior

_MP_SIMULATOR = None
_MP_COMPONENT_NAMES = None


def _init_duty_cycle_worker(simulator, component_names):
    global _MP_SIMULATOR
    global _MP_COMPONENT_NAMES
    _MP_SIMULATOR = simulator
    _MP_COMPONENT_NAMES = component_names


def _simulate_component_bit_ts(simulator, component_names, theta):
    bit_ts_dict = simulator.simulate_state_time_series(torch.as_tensor(theta))
    return np.stack(
        [np.asarray(bit_ts_dict[name], dtype=np.float32) for name in component_names],
        axis=-1,
    )


def _simulate_duty_cycle_worker(theta):
    return _simulate_component_bit_ts(_MP_SIMULATOR, _MP_COMPONENT_NAMES, theta)


def _simulate_iid_trials_worker(task):
    theta, ntrial, max_ntrial = task
    ntrial = int(ntrial)
    max_ntrial = int(max_ntrial)
    ntrial = min(max(1, ntrial), max_ntrial)

    first_trial = _simulate_component_bit_ts(_MP_SIMULATOR, _MP_COMPONENT_NAMES, theta)
    flat_dim = first_trial.shape[0] * first_trial.shape[1]

    trials = np.full((max_ntrial, flat_dim), np.nan, dtype=np.float32)
    trials[0] = first_trial.reshape(-1)
    for idx in range(1, ntrial):
        trial = _simulate_component_bit_ts(_MP_SIMULATOR, _MP_COMPONENT_NAMES, theta)
        trials[idx] = trial.reshape(-1)
    return trials


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
    def __init__(
            self,
            simulator,
            prior,
            embedding_net_kwargs={},
            iid_embedding_kwargs={},
            event_embedding_kwargs={},
        ):
        """
        Inference using an embedding network to learn summary statistics from time series data.

        Parameters
        ----------
        simulator : Simulator
            The simulator to use for generating data.
        prior : torch.distributions.Distribution
            The prior distribution over the parameters.
        embedding_net_kwargs : dict, optional
            The keyword arguments to pass to the embedding network.
        iid_embedding_kwargs : dict, optional
            Keyword arguments for permutation-invariant aggregation used when
            training with multiple iid trials. Supported keys are
            aggregation_fn, num_layers, num_hiddens, and output_dim.
        event_embedding_kwargs : dict, optional
            Keyword arguments controlling the per-trial run-length event
            transformer. Supported keys are max_events.
        """
        self.simulator = simulator
        self.simulator.truncate_output = False # Such that the simulator always outputs the same length of time series
        self.prior = prior
        self.embedding_net_kwargs = embedding_net_kwargs
        self.embedding_net_kwargs = dict(
            vit=False,                # time-series mode
            is_causal=True,           # Causal mask for time series
            num_hidden_layers=4,      # Number of transformer layers
            num_attention_heads=8,    # Number of attention heads
            num_key_value_heads=8,    # Number of key-value heads (can be different from num_attention_heads)
            feature_space_dim=128,     # Internal transformer dimension
            final_emb_dimension=64,   # Final embedding dimension passed to the density estimator
        )
        self.embedding_net_kwargs.update(embedding_net_kwargs)

        self.ncomponent = 1
        if isinstance(simulator, NetworkSimulator) or (hasattr(simulator, "components") and len(simulator.components) > 0):
            self.ncomponent = len(simulator.components)

        self.event_embedding_kwargs = dict(
            max_events=None,
        )
        self.event_embedding_kwargs.update(event_embedding_kwargs)
        self.trial_embedding_net = RunLengthEventTransformerEmbedding(
            ntime=self.simulator.nmax,
            ncomponent=self.ncomponent,
            transformer_cfg=self.embedding_net_kwargs,
            max_events=self.event_embedding_kwargs["max_events"],
        )
        self.embedding_net = self.trial_embedding_net
        self.iid_embedding_kwargs = dict(
            aggregation_fn="mean",
            num_layers=2,
            num_hiddens=128,
            output_dim=self.embedding_net_kwargs["final_emb_dimension"],
        )
        self.iid_embedding_kwargs.update(iid_embedding_kwargs)
        self.iid_mode = False
        self.max_ntrial = 1

        self.posterior_net = None

    def _build_iid_embedding_net(self):
        trial_wrapper = FlattenedTrialTimeSeriesEmbedding(
            trial_embedding_net=self.trial_embedding_net,
            ntime=self.simulator.nmax,
            ncomponent=self.ncomponent,
        )
        return PermutationInvariantEmbedding(
            trial_net=trial_wrapper,
            trial_net_output_dim=self.embedding_net_kwargs["final_emb_dimension"],
            aggregation_fn=self.iid_embedding_kwargs["aggregation_fn"],
            num_layers=self.iid_embedding_kwargs["num_layers"],
            num_hiddens=self.iid_embedding_kwargs["num_hiddens"],
            output_dim=self.iid_embedding_kwargs["output_dim"],
        )

    def train(
            self,
            method="SNPE",
            nsimulation=5000,
            device="cpu",
            batch_size=128,
            ncore=1,
            max_ntrial=10,
        ):
        """
        Train the posterior using the simulator.

        Parameters
        ----------
        method : str, optional
            The inference method to use. Default is "SNPE".
        nsimulation : int, optional
            The number of simulations to use for training the posterior.
        device : str, optional
            The device to use for training the posterior. Default is "cpu". Use "cuda" for GPU acceleration if available.
        batch_size : int, optional
            The batch size to use for training the posterior. Default is 128.
        ncore : int, optional
            The number of CPU processes to use for simulation. Set to 1 to disable multiprocessing.
        max_ntrial : int, optional
            Maximum number of iid trials per parameter set. For each simulated
            parameter vector, the number of trials is drawn uniformly from
            1..max_ntrial and missing trials are padded with NaNs.
            If max_ntrial > 1, iid mode with permutation-invariant embedding
            is enabled.
        """
        self.max_ntrial = max(1, int(max_ntrial))
        self.iid_mode = self.max_ntrial > 1
        if self.iid_mode:
            self.embedding_net = self._build_iid_embedding_net()
        else:
            self.embedding_net = self.trial_embedding_net

        # Move embedding_net and prior to device
        self.embedding_net.to(device)
        self.prior.to(device)

        # Sample parameters from the prior and simulate data
        # NOTE: We always perform the simulation on CPU
        thetas = self.prior.sample((nsimulation,)).to(device="cpu")
        component_names = tuple(self.simulator.components.keys())
        ncore = max(1, int(ncore))
        theta_values = thetas.detach().cpu().numpy()

        if self.iid_mode:
            flat_dim = self.simulator.nmax * self.ncomponent
            ntrials = np.random.randint(1, self.max_ntrial + 1, size=nsimulation)
            if ncore == 1 or nsimulation < 2:
                xs_values = np.full(
                    (nsimulation, self.max_ntrial, flat_dim),
                    np.nan,
                    dtype=np.float32,
                )
                for idx, theta in enumerate(tqdm.tqdm(theta_values)):
                    ntrial_i = int(ntrials[idx])
                    for trial_idx in range(ntrial_i):
                        component_bit_ts = _simulate_component_bit_ts(self.simulator, component_names, theta)
                        xs_values[idx, trial_idx] = component_bit_ts.reshape(-1)
            else:
                nworker = min(ncore, nsimulation, os.cpu_count() or ncore)
                chunksize = max(1, nsimulation // (nworker * 8))
                tasks = [
                    (theta_values[idx], int(ntrials[idx]), self.max_ntrial)
                    for idx in range(nsimulation)
                ]
                start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
                mp_ctx = mp.get_context(start_method)

                with mp_ctx.Pool(
                    processes=nworker,
                    initializer=_init_duty_cycle_worker,
                    initargs=(self.simulator, component_names),
                ) as pool:
                    xs_values = list(
                        tqdm.tqdm(
                            pool.imap(_simulate_iid_trials_worker, tasks, chunksize=chunksize),
                            total=nsimulation,
                        )
                    )
                xs_values = np.stack(xs_values, axis=0)
            xs = torch.from_numpy(xs_values)
        else:
            if ncore == 1 or nsimulation < 2:
                xs_values = np.empty((nsimulation, self.simulator.nmax, self.ncomponent), dtype=np.float32)
                for idx, theta in enumerate(tqdm.tqdm(theta_values)):
                    component_bit_ts = _simulate_component_bit_ts(self.simulator, component_names, theta)
                    xs_values[idx] = component_bit_ts
            else:
                nworker = min(ncore, nsimulation, os.cpu_count() or ncore)
                chunksize = max(1, nsimulation // (nworker * 8))
                start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
                mp_ctx = mp.get_context(start_method)

                with mp_ctx.Pool(
                    processes=nworker,
                    initializer=_init_duty_cycle_worker,
                    initargs=(self.simulator, component_names),
                ) as pool:
                    xs_values = list(
                        tqdm.tqdm(
                            pool.imap(_simulate_duty_cycle_worker, theta_values, chunksize=chunksize),
                            total=nsimulation,
                        )
                    )
                xs_values = np.stack(xs_values, axis=0)
            xs = torch.from_numpy(xs_values)

        if method == "SNPE":
            density_estimator = posterior_nn(
                model="maf",
                embedding_net=self.embedding_net,
                z_score_theta="independent",
                z_score_x="none",
            )
            # NPE is identical to SNPE = SNPE_C, which is the one we used also in SummaryStatisticInference
            self.inference = NPE(prior=self.prior, density_estimator=density_estimator, device=device)
            self.posterior_net = self.inference.append_simulations(
                thetas,
                xs,
                exclude_invalid_x=False,
            ).train(training_batch_size=batch_size)
        else:
            raise NotImplementedError(f"Method {method} not implemented yet.")

    def infer(
            self,
            observations,
            device="cpu",
            nposterior=10000,
            batch_size=8,
        ):
        """
        Make inferences on the parameters of the duty cycle model.
        
        Parameters
        ----------
        observations : tuple of array_like
            If max_ntrial=1 during training, shape (T, ncomponent) is treated as
            a single observation.
            If max_ntrial>1 during training, shape can be (T, ncomponent) or
            (ntrial, T, ncomponent) with 1 <= ntrial <= max_ntrial.
        device : str, optional
            The device to use for inference. Default is "cpu". Use "cuda" for GPU acceleration if available.
        nposterior : int, optional
            The number of posterior samples to draw for each observation. Default is 10000.
        batch_size : int, optional
            The batch size to use for computing log probabilities. Default is 8. This is only
            used when device is "cuda" to avoid using too much GPU memory.
        
        Returns
        -------
        posterior_samples : array_like
            The posterior samples, with shape (batch_size, nposterior, parameter_dim).
        log_probs : array_like
            The log probabilities of the posterior samples, with shape (batch_size, nposterior).
        """
        observations = torch.as_tensor(observations, dtype=torch.float32)
        if self.iid_mode:
            if observations.ndim == 2:
                observations = observations.unsqueeze(0)
            if observations.ndim != 3:
                raise ValueError(
                    "In iid mode, observations must have shape (T, ncomponent) "
                    "or (ntrial, T, ncomponent)."
                )
            ntrial_obs = int(observations.shape[0])
            if ntrial_obs < 1 or ntrial_obs > self.max_ntrial:
                raise ValueError(
                    f"Expected number of trials in [1, {self.max_ntrial}], got {ntrial_obs}."
                )
            if observations.shape[1] != self.simulator.nmax:
                raise ValueError(
                    f"Expected time dimension {self.simulator.nmax}, got {observations.shape[1]}."
                )
            if observations.shape[2] != self.ncomponent:
                raise ValueError(
                    f"Expected component dimension {self.ncomponent}, got {observations.shape[2]}."
                )
            flat_dim = self.simulator.nmax * self.ncomponent
            x_obs = torch.full(
                (1, self.max_ntrial, flat_dim),
                float("nan"),
                dtype=torch.float32,
                device=device,
            )
            x_obs[0, :ntrial_obs] = observations.reshape(ntrial_obs, flat_dim).to(device)
        else:
            if observations.ndim == 2:
                x_obs = observations.unsqueeze(0).to(device)
            else:
                x_obs = observations.to(device)
        self.posterior_net.to(device)
        trained_posterior = self.inference.build_posterior()

        with torch.no_grad():
            posterior_samples = trained_posterior.sample((nposterior,), x_obs).detach()
        # To avoid using too much GPU memory, we compute the log_probs in batches
        def batched_log_prob(posterior, samples, x_obs, batch_size=1024):
            logps = []
            with torch.no_grad():
                for start in range(0, samples.shape[0], batch_size):
                    end = start + batch_size
                    lp = posterior.log_prob(samples[start:end], x_obs)
                    logps.append(lp.detach().cpu())
                    torch.cuda.empty_cache()
            return torch.cat(logps, dim=0)
        log_probs = batched_log_prob(trained_posterior, posterior_samples, x_obs, batch_size=min(batch_size, nposterior))

        return posterior_samples.cpu().numpy(), log_probs
