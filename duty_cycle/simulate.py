import numpy as np

from . import _UP, _DOWN, _UNDEF
from .utils import (
    find_contiguous_up_and_down_segments,
    convert_start_end_indices_to_duration,
)

class Simulator:
    param_names = []
    param_labels = []

    def __init__(self, dt, nmax=1000, random_seed=None):
        """
        Parameters
        ----------
        dt : float
            The time step of the simulation.
        nmax : int, optional
            The maximum number of time steps to simulate.
        random_seed : int or None, optional
            The random seed to use for the simulation. If None, no seed is set.
        """
        self.dt = dt
        self.nmax = nmax
        self.params = None

        # Set the random seed
        if random_seed is not None:
            np.random.seed(random_seed)

    @property
    def duration(self):
        return self.dt * self.nmax

    def unpack_params(self, simulation_params, use_torch=True):
        import torch
        if type(simulation_params) is torch.Tensor:
            if use_torch:
                return dict(zip(self.param_names, [p.item() for p in simulation_params]))
            else:
                # Move the tensor to CPU and detach it
                return dict(zip(self.param_names, simulation_params.detach().cpu().numpy()))
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

            cont_up_times += [convert_start_end_indices_to_duration(*idxs, dt=self.dt) for idxs in cont_up_time_idxs]
            cont_down_times += [convert_start_end_indices_to_duration(*idxs, dt=self.dt) for idxs in cont_down_time_idxs]

        return cont_up_times, cont_down_times

class IndependentUpDownSegments(Simulator):
    def _simulate_step(
        self,
        output,
        dt,
        idx,
        idx_lastup,
        cont_up_time,
        cont_down_time,
    ):
        _state = _UNDEF
        # Draw a random number ~U(0,1)
        u = np.random.rand()
        if output[idx-1] == _UP:
            # In the previous time step, the detector was UP

            # Compute p_cont_up
            # NOTE Here idx_lastup means the first index where the detector was UP
            if cont_up_time < 0:
                p_cont_up = 0
            else:
                p_cont_up = 1 - self.transition_prob_utd(idx, idx_lastup, dt, cont_up_time)
            if u < p_cont_up:
                # In this time step, the detector remains UP
                _state = _UP
            else:
                # In this time step, the detector is no longer UP
                _state = _DOWN
                idx_lastup = idx # This is the LAST index that the detector was UP
                # Make a draw from the normal distribution
                cont_down_time = self.realize_cont_down_timescale()
        else:
            # In the previous time step, the detector is DOWN

            # Compute p_cont_down
            # NOTE Here idx_lastup means the last index where the detector was UP
            if cont_down_time < 0:
                p_cont_down = 0
            else:
                p_cont_down = 1 - self.transition_prob_dtu(idx, idx_lastup, dt, cont_down_time)
            if u < p_cont_down:
                # In this time step, the detector remains DOWN
                _state = _DOWN
            else:
                # In this time step, the detector is no longer DOWN
                _state = _UP
                idx_lastup = idx
                # Make a draw from the normal distribution
                cont_up_time = self.realize_cont_up_timescale()

        return _state, idx_lastup, cont_up_time, cont_down_time

    def _simulate(
            self,
            initial_state=_UP,
            idx_lastup=0,
            cont_up_time=None,
            cont_down_time=None
        ):
        """
        Simulate the up and down segments of the duty cycle.

        Parameters
        ----------
        initial_state : int, optional
            The initial state of the detector. Either _UP or _DOWN.
        idx_lastup : int, optional
            The index of the last time the detector was up.
        cont_up_time : float or None, optional
            The contiguous up time to use. If None, a new contiguous up time is drawn.
        cont_down_time : float or None, optional
            The contiguous down time to use. If None, a new contiguous down time is drawn.
        
        Returns
        -------
        output : array-like
            The simulated duty cycle.
        """

        output = np.ones(self.nmax, dtype=int)*_UNDEF
        output[0] = initial_state

        # Make a draw from the normal distribution if needed
        if initial_state == _UP:
            if cont_up_time is None:
                cont_up_time = self.realize_cont_up_timescale()
        elif initial_state == _DOWN:
            if cont_down_time is None:
                cont_down_time = self.realize_cont_down_timescale()
        else:
            raise ValueError("previous_state must be either _UP or _DOWN")

        dt = self.dt # Time step

        # NOTE We generate the simulated duty cycle sequentially
        for idx in range(1, self.nmax):
            output[idx], idx_lastup, cont_up_time, cont_down_time = self._simulate_step(
                output,
                dt,
                idx,
                idx_lastup,
                cont_up_time,
                cont_down_time,
            )

        # Truncate the output to the last UP/DN state change
        last_change_idx = np.where(np.diff(output) != 0)[0]
        if len(last_change_idx) > 0:
            last_change_idx = last_change_idx[-1] + 1
            output = output[:last_change_idx]

        return output

class MemorylessMarkovChain(IndependentUpDownSegments):
    """
    Memoryless Markov chain simulation for duty cycles.

    The model is defined by two parameters:
    - p_utd: Probability of transitioning from UP to DOWN in each time step.
    - p_dtu: Probability of transitioning from DOWN to UP in each time step.

    The transitions are memoryless, meaning the probability of transitioning does not depend on how long the system has been in the current state.
    """
    param_names = [
        "p_utd",
        "p_dtu",
    ]
    param_labels = [
        r"$p_{\rm up\;to\;down}$",
        r"$p_{\rm down\;to\;up}$",
    ]

    # Define the functions needed for the simulation specifically for this model
    def realize_cont_up_timescale(self):
        return np.nan # Not used in this model
    
    def realize_cont_down_timescale(self):
        return np.nan # Not used in this model

    def transition_prob_utd(self, idx, idx_lastup, dt, cont_up_time):
        return self.params["p_utd"]

    def transition_prob_dtu(self, idx, idx_lastup, dt, cont_down_time):
        return self.params["p_dtu"]

    def simulate_duty_cycle(self, simulation_params, initial_state=_UP, idx_lastup=0, cont_up_time=None, cont_down_time=None):
        self.params = self.unpack_params(simulation_params, use_torch=True)

        # Sanity check
        assert 0 <= self.params["p_utd"] <= 1, "p_utd must be between 0 and 1"
        assert 0 <= self.params["p_dtu"] <= 1, "p_dtu must be between 0 and 1"

        return self._simulate(
            initial_state=initial_state,
            idx_lastup=idx_lastup,
            cont_up_time=cont_up_time,
            cont_down_time=cont_down_time,
        )

class WeibullVLMC(IndependentUpDownSegments):
    """
    Variable length Markov chain simulation for duty cycles that gives Weibull-distributed contiguous up and down times.

    The model is defined by six parameters:
    - scale_up: Scale parameter for the Weibull distribution of contiguous up times.
    - shape_up: Shape parameter for the Weibull distribution of contiguous up times.
    - initial_deadtime_up: Initial dead time after transitioning to UP state during which no further transitions can occur.
    - scale_down: Scale parameter for the Weibull distribution of contiguous down times.
    - shape_down: Shape parameter for the Weibull distribution of contiguous down times.
    - initial_deadtime_down: Initial dead time after transitioning to DOWN state during which no further transitions can occur.

    The transitions depend on the time since the last state change, with probabilities defined by the discrete Weibull distribution.
    When shape=1, the distribution reduces to the exponential distribution, corresponding to a memoryless process.

    """
    param_names = [
        "scale_up",
        "shape_up",
        "initial_deadtime_up",
        "scale_down",
        "shape_down",
        "initial_deadtime_down",
    ]
    param_labels = [
        r"$\lambda_{\rm up}$",
        r"$k_{\rm up}$",
        r"$t_{\rm dead,up}$",
        r"$\lambda_{\rm down}$",
        r"$k_{\rm down}$",
        r"$t_{\rm dead,down}$",
    ]

    # Define the functions needed for the simulation specifically for this model
    def realize_cont_up_timescale(self):
        return np.nan # Not used in this model

    def realize_cont_down_timescale(self):
        return np.nan # Not used in this model

    def transition_prob_utd(self, idx, idx_lastup, dt, cont_up_time):
        idx_lastdead = int(np.floor(self.params["initial_deadtime_up"]/dt)) # This last index (relative to idx_lastup) where the detector is still dead
        if (idx - idx_lastup) <= idx_lastdead:
            return 0 # No transition possible during dead time
        else:
            return 1 - np.exp(
                -( ((idx-idx_lastup-idx_lastdead)*dt) / self.params["scale_up"] )**self.params["shape_up"] + ( ((idx-idx_lastup-idx_lastdead-1)*dt) / self.params["scale_up"] )**self.params["shape_up"]
            )
    
    def transition_prob_dtu(self, idx, idx_lastup, dt, cont_down_time):
        idx_lastdead = int(np.floor(self.params["initial_deadtime_down"]/dt)) # This last index (relative to idx_lastup) where the detector is still dead
        if (idx - idx_lastup) <= idx_lastdead:
            return 0 # No transition possible during dead time
        else:
            return 1 - np.exp(
                -( ((idx-idx_lastup-idx_lastdead)*dt) / self.params["scale_down"] )**self.params["shape_down"] + ( ((idx-idx_lastup-idx_lastdead-1)*dt) / self.params["scale_down"] )**self.params["shape_down"]
            )

    def simulate_duty_cycle(self, simulation_params, initial_state=_UP, idx_lastup=0, cont_up_time=None, cont_down_time=None):
        self.params = self.unpack_params(simulation_params, use_torch=True)

        # Sanity check
        assert self.params["scale_up"] > 0, "scale_up must be positive"
        assert self.params["shape_up"] > 0, "shape_up must be positive"
        assert self.params["initial_deadtime_up"] >= 0, "initial_deadtime_up must be non-negative"
        assert self.params["scale_down"] > 0, "scale_down must be positive"
        assert self.params["shape_down"] > 0, "shape_down must be positive"
        assert self.params["initial_deadtime_down"] >= 0, "initial_deadtime_down must be non-negative"

        return self._simulate(
            initial_state=initial_state,
            idx_lastup=idx_lastup,
            cont_up_time=cont_up_time,
            cont_down_time=cont_down_time,
        )
