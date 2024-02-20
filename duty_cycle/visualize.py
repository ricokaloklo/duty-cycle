import numpy as np
from matplotlib import pyplot as plt

from .simulate import _UP, _DOWN
from .utils import find_contiguous_up_and_down_segments

def visualize_duty_cycle(simulation_output, use_tex=True):
    if use_tex:
        plt.rcParams.update({
            "text.usetex": True,
        })
    else:
        plt.rcParams.update({
            "text.usetex": False,
        })

    fig = plt.figure(dpi=150)
    ax = fig.gca()

    cont_up_segments, cont_down_segments = find_contiguous_up_and_down_segments(simulation_output)
    N = len(simulation_output)
    
    # Plot the UP segments
    for up_segment in cont_up_segments:
        ax.plot(
            np.arange(up_segment[0], up_segment[1]+1, 1)/N, 
            simulation_output[up_segment[0]:up_segment[1]+1], 
            color="tab:green",
        )
    # Plot the DOWN segments
    for down_segment in cont_down_segments:
        ax.plot(
            np.arange(down_segment[0], down_segment[1]+1, 1)/N, 
            simulation_output[down_segment[0]:down_segment[1]+1], 
            color="tab:red",
        )

    ax.set_ylabel(r"${\rm detector\;state}$")
    ax.set_yticks([_UP, _DOWN], ["UP", "DOWN"])

    ax.grid(alpha=0.5)
    ax.set_xlabel(r"$t/T$")
    
    return fig