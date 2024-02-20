from .simulate import _UP, _DOWN

def find_contiguous_up_and_down_segments(simulation_output):
    """
    Find contiguous UP and DOWN segments in the simulation output.

    Parameters
    ----------
    simulation_output : array_like
        The simulation output.
    
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
    for idx in range(len(simulation_output)):
        if simulation_output[idx] == _UP:
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
        cont_up_segments.append([_idx, len(simulation_output)-1])
    else:
        cont_down_segments.append([_idx, len(simulation_output)-1])

    return cont_up_segments, cont_down_segments