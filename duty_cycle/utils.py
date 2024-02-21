import numpy as np

from . import _UP, _DOWN

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
