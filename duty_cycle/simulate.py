import numpy as np
import torch

from . import _UP, _DOWN

_param_names = [
    "mean_cont_up_time",
    "std_cont_up_time",
    "mean_cont_down_time",
    "std_cont_down_time",
    "k_cont_up",
    "k_cont_down",
]

def sigmoid(x, x0=0.5, k=1):
    """
    A sigmoid function where the input x is bounded between 0 and 1.

    Parameters
    ----------
    x : float or array-like
        The input to the sigmoid function.
    x0 : float, optional
        The point of which sigmoid(x0)=1/2.
    k : float, optional
        The steepness of the sigmoid function.

    Returns
    -------
    output : float or array-like
        The output of the sigmoid function.

    Raises
    ------
    ValueError
        If the input x is not between 0 and 1.
    
    Notes
    -----
    This sigmoid function is basically applying an inverse logistic
    function to the input x, then feeding the output to the standard
    logistic function.
    """
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("Input x must be between 0 and 1.")
    r = np.log(0.5)/np.log(x0)
    xpowr = x**r
    return xpowr**k/(xpowr**k + (1.-xpowr)**k)

def simulate_duty_cycle(simulation_params, N=1000, random_seed=None):
    """
    Simulate the duty cycle of a detector.

    Parameters
    ----------
    simulation_params : array-like
        The parameters of the duty cycle model.
    N : int, optional
        The number of time steps to simulate.
    random_seed : int, optional
        The random seed to use for the simulation.

    Returns
    -------
    output : array-like
        The simulated duty cycle.
    """
    # Figure out whether we should cast the output to a torch tensor
    _use_torch = True if type(simulate_duty_cycle) is torch.Tensor else False

    # Set the random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    # Unpack the simulation parameters
    if _use_torch:
        params = dict(zip(_param_names, [p.item() for p in simulation_params]))
    else:
        params = dict(zip(_param_names, simulation_params))

    output = np.ones(N, dtype=int)*_DOWN
    # NOTE We start a simulation with the detector is the UP state
    output[0] = _UP
    idx_lastup = 0
    dt = 1./N # Time step

    # NOTE We generate the simulated duty cycle sequentially
    for idx in range(1, N):
        # Draw a random number ~U(0,1)
        u = np.random.rand()

        if output[idx-1] == _UP:
            # In the previous time step, the detector is UP

            # Compute p_cont_up
            # NOTE Here idx_lastup means the first index where the detector was UP
            p_cont_up = 1 - sigmoid(
                (idx-idx_lastup)*dt,
                x0=np.random.normal(
                    params["mean_cont_up_time"],
                    params["std_cont_up_time"],
                ),
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
            p_cont_down = 1 - sigmoid(
                (idx-idx_lastup)*dt,
                x0=np.random.normal(
                    params["mean_cont_down_time"],
                    params["std_cont_down_time"],
                ),
                k=params["k_cont_down"],
            )

            if u < p_cont_down:
                # In this time step, the detector remains DOWN
                output[idx] = _DOWN
            else:
                # In this time step, the detector is no longer DOWN
                output[idx] = _UP
                idx_lastup = idx

    if _use_torch:
        # Pack the output into a torch Tensor
        output = torch.from_numpy(output)
    return output
