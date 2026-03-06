import numpy as np
import pandas as pd
import torch
from torch import distributions as dist
from torch.distributions import constraints
from torch import nn
import os, pathlib
import json
from sbi.neural_nets import embedding_nets

from . import _UP, _DOWN

def load_detector_coordinates():
    det_locs_json_path = os.path.join(
        pathlib.Path(__file__).resolve().parent,
        "data/",
        "detector_loc.json"
    )
    det_locs_list = json.load(open(det_locs_json_path, "r"))
    det_locs = {}
    for det in det_locs_list:
        det_locs[det["name"]] = det["location"]

    return det_locs

detector_coordinates = load_detector_coordinates()

def load_earthquake_catalog():
    eq_catalog_csv_path = os.path.join(
        pathlib.Path(__file__).resolve().parent,
        "data/",
        "USGS_EarthquakeCatalogFrom2025_MagnAbove5.csv"
    )
    eq_catalog = pd.read_csv(eq_catalog_csv_path)
    return eq_catalog

# From https://github.com/pytorch/pytorch/issues/11412
class LogUniform(dist.TransformedDistribution):
    arg_constraints = {
        "low": constraints.dependent,
        "high": constraints.dependent,
    }

    def __init__(self, low, high):
        if type(low) is not torch.Tensor:
            low = torch.tensor(low)
        if type(high) is not torch.Tensor:
            high = torch.tensor(high)

        self.low = low
        self.high = high

        super(LogUniform, self).__init__(
            dist.Uniform(
                self.low.log(),
                self.high.log()
            ),
            dist.ExpTransform()
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

def find_contiguous_up_and_down_segments(bit_ts):
    """
    Find contiguous UP and DOWN segments in the simulation output.

    Parameters
    ----------
    bit_ts : array_like
        A time series of 'bits' of UP and DOWN as an array.
    
    Returns
    -------
    cont_up_segments : list
        A list of lists where each sublist contains the start and end indices of a contiguous UP segment.
    cont_down_segments : list
        A list of lists where each sublist contains the start and end indices of a contiguous DOWN segment.
    """
    cont_up_segments = []
    cont_down_segments = []

    _idx = 0 # A flag
    was_up = True

    # Loop over the simulation output one-by-one
    for idx in range(len(bit_ts)):
        if bit_ts[idx] == _UP:
            if not was_up:
                # The detector was DOWN in the previous time step
                cont_down_segments.append([_idx, idx-1])
                _idx = idx # Move the flag to the current index
                was_up = True
        else:
            if was_up:
                # The detector was UP in the previous time step
                cont_up_segments.append([_idx, idx-1])
                _idx = idx # Move the flag to the current index
                was_up = False

    # Append the last segment
    if was_up:
        cont_up_segments.append([_idx, len(bit_ts)-1])
    else:
        cont_down_segments.append([_idx, len(bit_ts)-1])

    return cont_up_segments, cont_down_segments

def generate_state_bit_timeseries_from_dataframe(df, state, start_time=None, end_time=None, dt=1):
    """
    Generate a time series of UP and DOWN states from a dataframe.

    Parameters
    ----------
    df: pandas.DataFrame
        A dataframe containing the start and end times of UP and DOWN states.
    state: int
        The state to generate the time series for. Must be either _UP or _DOWN.
    start_time: float, optional
        The start time of the time series. If None, the start time of the dataframe is used.
    end_time: float, optional
        The end time of the time series. If None, the end time of the dataframe is used.
    dt: float, optional
        The time step of the time series.
    
    Returns
    -------
    output : array_like
        The time series of UP and DOWN states.
    """
    assert state in [_UP, _DOWN], "state must be either _UP or _DOWN"
    rev_state = _DOWN if state == _UP else _UP

    if start_time is None:
        start_time = df.iloc[0]["start_time"] # Start time for the entire dataframe
    if end_time is None:
        end_time = df.iloc[-1]["end_time"] # End time for the entire dataframe

    output = np.ones(int((end_time - start_time)/dt)+1)*rev_state
    start_idx = df[df["start_time"] >= start_time].index[0]
    end_idx = df[df["end_time"] <= end_time].index[-1]

    for i in range(start_idx, end_idx+1):
        _st_idx = int((df.iloc[i]["start_time"]-start_time)/dt)
        _et_idx = int((df.iloc[i]["end_time"]-start_time)/dt)+1
        output[_st_idx:_et_idx] = state

    return output

def convert_start_end_indices_to_duration(start_idx, end_idx, dt):
    """
    Convert a start and end index to a duration.

    Parameters
    ----------
    start_idx : int
        The start index.
    end_idx : int
        The end index.
    dt : float
        The time step.
    
    Returns
    -------
    output : float
        The duration.
    """
    return (end_idx-start_idx)*dt

def calculate_duty_factor(bit_ts):
    return len(bit_ts[bit_ts == _UP])/len(bit_ts)

class MultivariateTimeSeriesTransformerEmbedding(nn.Module):
    """
    Wrapper: project N-D time series to d_model, then run TransformerEmbedding.

    Input to forward: x with shape (batch, T, N) or (T, N).
    Output: (batch, emb_dim)
    """

    def __init__(self, input_dim: int, transformer_cfg: dict):
        super().__init__()

        # Base transformer embedding (time-series mode: vit=False)
        self.transformer = embedding_nets.TransformerEmbedding(transformer_cfg)

        # Linear projection from N channels -> d_model per time step
        self.proj = nn.Linear(input_dim, transformer_cfg["feature_space_dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, T, N) or (T, N)
        returns: (batch, emb_dim)
        """
        if x.ndim == 2:
            # (T, N) -> (1, T, N)
            x = x.unsqueeze(0)

        # Project feature dimension N -> d_model
        # Result: (batch, T, d_model)
        x = self.proj(x)

        # TransformerEmbedding in time-series mode expects (batch, seq_len, feature_space_dim)
        # and returns (batch, final_emb_dimension).
        z = self.transformer(x)
        return z


class RunLengthEventSetEmbedding(nn.Module):
    """
    Encode a binary multivariate time series as a set of run-length events.

    Each event row contains the component id, state, normalized start index,
    and normalized duration. The event rows are embedded independently with
    an MLP and then pooled across the set.
    """

    def __init__(
        self,
        ntime: int,
        ncomponent: int,
        num_hiddens: int = 64,
        num_layers: int = 2,
        output_dim: int = 64,
        max_events: int | None = None,
    ):
        super().__init__()
        self.ntime = int(ntime)
        self.ncomponent = int(ncomponent)
        self.max_events = int(max_events) if max_events is not None else self.ntime * self.ncomponent
        self.event_input_dim = self.ncomponent + 3
        self.row_embedding = self._build_mlp(
            input_dim=self.event_input_dim,
            hidden_dim=num_hiddens,
            output_dim=num_hiddens,
            num_layers=num_layers,
        )
        self.set_projection = self._build_mlp(
            input_dim=num_hiddens + 1,
            hidden_dim=num_hiddens,
            output_dim=output_dim,
            num_layers=num_layers,
        )

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
        num_layers = max(1, int(num_layers))
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
            return nn.Sequential(*layers)

        layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def _build_event_table(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        batch_size, ntime, ncomponent = x.shape
        if ntime != self.ntime or ncomponent != self.ncomponent:
            raise ValueError(
                f"Expected input shape (*, {self.ntime}, {self.ncomponent}), got {tuple(x.shape)}."
            )

        x_binary = (x > 0.5).to(dtype=torch.long)
        event_table = x.new_zeros((batch_size, self.max_events, self.event_input_dim))
        event_mask = torch.zeros((batch_size, self.max_events), dtype=torch.bool, device=x.device)
        component_eye = torch.eye(self.ncomponent, dtype=x.dtype, device=x.device)
        max_start = max(1, self.ntime - 1)

        for batch_idx in range(batch_size):
            write_idx = 0
            for component_idx in range(self.ncomponent):
                if write_idx >= self.max_events:
                    break

                seq = x_binary[batch_idx, :, component_idx]
                change_points = torch.nonzero(seq[1:] != seq[:-1], as_tuple=False).flatten() + 1
                starts = torch.cat((seq.new_zeros(1), change_points))
                stops = torch.cat((change_points, seq.new_tensor([self.ntime]))) - 1
                durations = stops - starts + 1
                states = seq.index_select(0, starts)

                num_events = starts.numel()
                num_store = min(num_events, self.max_events - write_idx)
                if num_store == 0:
                    continue

                event_rows = torch.cat(
                    (
                        component_eye[component_idx].unsqueeze(0).expand(num_store, -1),
                        states[:num_store].unsqueeze(-1).to(dtype=x.dtype),
                        starts[:num_store].unsqueeze(-1).to(dtype=x.dtype) / max_start,
                        durations[:num_store].unsqueeze(-1).to(dtype=x.dtype) / self.ntime,
                    ),
                    dim=-1,
                )
                event_table[batch_idx, write_idx : write_idx + num_store] = event_rows
                event_mask[batch_idx, write_idx : write_idx + num_store] = True
                write_idx += num_store

        return event_table, event_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        event_table, event_mask = self._build_event_table(x)
        row_embeddings = self.row_embedding(event_table.reshape(-1, self.event_input_dim))
        row_embeddings = row_embeddings.reshape(event_table.shape[0], self.max_events, -1)
        row_embeddings = row_embeddings * event_mask.unsqueeze(-1)

        valid_counts = event_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = row_embeddings.sum(dim=1) / valid_counts.to(dtype=row_embeddings.dtype)
        normalized_counts = valid_counts.to(dtype=row_embeddings.dtype) / self.max_events
        return self.set_projection(torch.cat((pooled, normalized_counts), dim=-1))


class HybridBitSequenceTrialEmbedding(nn.Module):
    """
    Fuse raw-sequence and run-length-event embeddings for a single trial.
    """

    def __init__(
        self,
        raw_embedding_net: nn.Module,
        event_embedding_net: nn.Module,
        raw_output_dim: int,
        event_output_dim: int,
        fusion_hidden_dim: int,
    ):
        super().__init__()
        self.raw_embedding_net = raw_embedding_net
        self.event_embedding_net = event_embedding_net
        self.fusion_net = nn.Sequential(
            nn.Linear(raw_output_dim + event_output_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, raw_output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_embedding = self.raw_embedding_net(x)
        event_embedding = self.event_embedding_net(x)
        fused = self.fusion_net(torch.cat((raw_embedding, event_embedding), dim=-1))
        return raw_embedding + fused


class FlattenedTrialTimeSeriesEmbedding(nn.Module):
    """
    Adapter for trial-based iid data.

    Input is expected as a flattened per-trial tensor with shape
    (batch, ntrial, T * ncomponent). The adapter reshapes each trial back to
    (T, ncomponent), applies the trial embedding network, and returns
    (batch, ntrial, emb_dim).
    """

    def __init__(self, trial_embedding_net: nn.Module, ntime: int, ncomponent: int):
        super().__init__()
        self.trial_embedding_net = trial_embedding_net
        self.ntime = int(ntime)
        self.ncomponent = int(ncomponent)
        self.flat_dim = self.ntime * self.ncomponent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            if x.shape[-1] != self.flat_dim:
                raise ValueError(f"Expected flattened size {self.flat_dim}, got {x.shape[-1]}.")
            x_reshaped = x.reshape(x.shape[0], self.ntime, self.ncomponent)
            return self.trial_embedding_net(x_reshaped)

        if x.ndim != 3:
            raise ValueError(f"Expected tensor with 2 or 3 dims, got {x.ndim}.")
        if x.shape[-1] != self.flat_dim:
            raise ValueError(f"Expected flattened size {self.flat_dim}, got {x.shape[-1]}.")

        batch_size, ntrial, _ = x.shape
        x_reshaped = x.reshape(batch_size * ntrial, self.ntime, self.ncomponent)
        embedded = self.trial_embedding_net(x_reshaped)
        return embedded.reshape(batch_size, ntrial, -1)
