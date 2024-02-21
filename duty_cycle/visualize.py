import numpy as np
from matplotlib import pyplot as plt

from . import _UP, _DOWN
from .utils import find_contiguous_up_and_down_segments

def visualize_duty_cycle(bit_ts, use_tex=True):
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

    cont_up_segments, cont_down_segments = find_contiguous_up_and_down_segments(bit_ts)
    N = len(bit_ts)
    
    # Plot the UP segments
    for up_segment in cont_up_segments:
        ax.plot(
            np.arange(up_segment[0], up_segment[1]+1, 1)/N, 
            bit_ts[up_segment[0]:up_segment[1]+1], 
            color="tab:green",
        )
    # Plot the DOWN segments
    for down_segment in cont_down_segments:
        ax.plot(
            np.arange(down_segment[0], down_segment[1]+1, 1)/N, 
            bit_ts[down_segment[0]:down_segment[1]+1], 
            color="tab:red",
        )

    ax.set_ylabel(r"${\rm detector\;state}$")
    ax.set_yticks([_UP, _DOWN], ["UP", "DOWN"])

    ax.grid(alpha=0.5)
    ax.set_xlabel(r"$t/T$")
    
    return fig

def visualize_binned_data(binned_data, bin_edges, histtype="step", xlabel=r"$\tau/T$", use_tex=True):
    if use_tex:
        plt.rcParams.update({
            "text.usetex": True,
        })
    else:
        plt.rcParams.update({
            "text.usetex": False,
        })

    assert histtype in ["step", "stepfilled"], "Only support histtype='step' or histtype='stepfilled'"
    _filled = True if histtype == "stepfilled" else False

    fig = plt.figure(dpi=150)
    ax = fig.gca()

    ax.stairs(
        binned_data,
        edges=bin_edges,
        fill=_filled,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"${\rm density}$")

    return fig