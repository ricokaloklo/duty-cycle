import numpy as np
import torch

from . import _UP, _DOWN
from .utils import (
    sigmoid,
    find_contiguous_up_and_down_segments,
    convert_start_end_indices_to_duration,
)

class Simulator:
    param_names = []
    param_labels = []

    def __init__(self, N=1000, random_seed=None):
        self.N = N

        # Set the random seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def unpack_params(self, simulation_params, use_torch=False):
        if use_torch:
            return dict(zip(self.param_names, [p.item() for p in simulation_params]))
        else:
            return dict(zip(self.param_names, simulation_params))

    def simulate_duty_cycle(self, simulation_params):
        """
        Simulate the duty cycle of a detector.

        Parameters
        ----------
        simulation_params : array-like
            The parameters of the duty cycle model.

        Returns
        -------
        output : array-like
            The simulated duty cycle.
        """
        raise NotImplementedError

    def simulate_cont_up_down_times(self, simulation_params, nsample=1000):
        """
        Simulate the contiguous up and down times of a detector.

        Parameters
        ----------
        simulation_params : array-like
            The parameters of the duty cycle model.
        nsample : int, optional
            The minimum number of samples to simulate.

        Returns
        -------
        cont_up_times : array-like
            The simulated contiguous up times.
        cont_down_times : array-like
            The simulated contiguous down times.
        """
        cont_up_times = []
        cont_down_times = []

        while len(cont_up_times) < nsample or len(cont_down_times) < nsample:
            simulated_bit_ts = self.simulate_duty_cycle(simulation_params)
            cont_up_time_idxs, cont_down_time_idxs = find_contiguous_up_and_down_segments(simulated_bit_ts)

            cont_up_times += [convert_start_end_indices_to_duration(*idxs, dt=1./self.N) for idxs in cont_up_time_idxs]
            cont_down_times += [convert_start_end_indices_to_duration(*idxs, dt=1./self.N) for idxs in cont_down_time_idxs]

        return cont_up_times, cont_down_times

class SigmoidDropOffModel(Simulator):
    param_names = [
        "mean_cont_up_time",
        "std_cont_up_time",
        "mean_cont_down_time",
        "std_cont_down_time",
        "k_cont_up",
        "k_cont_down",
    ]
    param_labels = [
        r"$\mu_{\tau_{\rm cont;up}}$",
        r"$\sigma_{\tau_{\rm cont;up}}$",
        r"$\mu_{\tau_{\rm cont;down}}$",
        r"$\sigma_{\tau_{\rm cont;down}}$",
        r"$k_{\rm cont;up}$",
        r"$k_{\rm cont;down}$",
    ]

    def simulate_duty_cycle(self, simulation_params):
        _use_torch = True if type(simulation_params) is torch.Tensor else False
        params = self.unpack_params(simulation_params, use_torch=_use_torch)

        output = np.ones(self.N, dtype=int)*_DOWN
        # NOTE We start a simulation with the detector is the UP state
        output[0] = _UP
        idx_lastup = 0
        dt = 1./self.N # Time step

        # NOTE We generate the simulated duty cycle sequentially
        for idx in range(1, self.N):
            # Draw a random number ~U(0,1)
            u = np.random.rand()

            if output[idx-1] == _UP:
                # In the previous time step, the detector is UP

                # Compute p_cont_up
                # NOTE Here idx_lastup means the first index where the detector was UP
                # Make a draw from the normal distribution
                _cont_up_time = np.random.normal(
                    params["mean_cont_up_time"],
                    params["std_cont_up_time"],
                )
                if _cont_up_time < 0:
                    p_cont_up = 0
                elif _cont_up_time > 1:
                    p_cont_up = 1
                else:
                    p_cont_up = 1 - sigmoid(
                        (idx-idx_lastup)*dt,
                        x0=_cont_up_time,
                        k=params["k_cont_up"],
                    )

                if u < p_cont_up:
                    # In this time step, the detector remains UP
                    output[idx] = _UP
                else:
                    # In this time step, the detector is no longer UP
                    output[idx] = _DOWN
                    idx_lastup = idx # This is the LAST index that the detector is UP
            else:
                # In the previous time step, the detector is DOWN
                
                # Compute p_cont_down
                # NOTE Here idx_lastup means the last index where the detector was UP
                # Make a draw from the normal distribution
                _cont_down_time = np.random.normal(
                    params["mean_cont_down_time"],
                    params["std_cont_down_time"],
                )
                if _cont_down_time < 0:
                    p_cont_down = 0
                elif _cont_down_time > 1:
                    p_cont_down = 1
                else:
                    p_cont_down = 1 - sigmoid(
                        (idx-idx_lastup)*dt,
                        x0=_cont_down_time,
                        k=params["k_cont_down"],
                    )

                if u < p_cont_down:
                    # In this time step, the detector remains DOWN
                    output[idx] = _DOWN
                else:
                    # In this time step, the detector is no longer DOWN
                    output[idx] = _UP
                    idx_lastup = idx

        return output
