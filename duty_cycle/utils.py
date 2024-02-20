import numpy as np

from . import _UP, _DOWN

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

def generate_state_bit_timeseries_from_dataframe(df, state, dt=1):
    assert state in [_UP, _DOWN], "state must be either _UP or _DOWN"
    inv_state = _DOWN if state == _UP else _UP

    _st = df.iloc[0]["start_time"] # Start time for the entire dataframe
    _et = df.iloc[-1]["end_time"] # End time for the entire dataframe

    output = np.ones(int((_et - _st)/dt))*state
    for i in range(len(df)-1):
        _this_st = df.iloc[i]["end_time"]
        _this_et = df.iloc[i+1]["start_time"]
        output[int((_this_st-_st)/dt):int((_this_et-_st)/dt)] = inv_state

    return output
