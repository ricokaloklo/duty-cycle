import numpy as np
from matplotlib import pyplot as plt

from . import _UP, _DOWN
from .utils import find_contiguous_up_and_down_segments

def visualize_duty_cycle(bit_ts, dt, use_tex=True):
    original_rcParams = plt.rcParams.copy()
    if use_tex:
        plt.rcParams.update({
            "text.usetex": True,
        })
    else:
        plt.rcParams.update({
            "text.usetex": False,
        })

    fig = plt.figure(dpi=150, figsize=(6, 2))
    ax = fig.gca()

    cont_up_segments, cont_down_segments = find_contiguous_up_and_down_segments(bit_ts)
    
    # Plot the UP segments
    for up_segment in cont_up_segments:
        ax.plot(
            np.arange(up_segment[0], up_segment[1]+1, 1)*dt, 
            bit_ts[up_segment[0]:up_segment[1]+1], 
            color="tab:green",
            linewidth=2,
        )
        ax.fill_between(
            np.arange(up_segment[0], up_segment[1]+1, 1)*dt,
            0, 1,
            color="tab:green",
            alpha=0.3,
        )
    # Plot the DOWN segments
    for down_segment in cont_down_segments:
        ax.plot(
            np.arange(down_segment[0], down_segment[1]+1, 1)*dt, 
            bit_ts[down_segment[0]:down_segment[1]+1], 
            color="tab:red",
            linewidth=2,
        )
        ax.fill_between(
            np.arange(down_segment[0], down_segment[1]+1, 1)*dt,
            0, 1,
            color="tab:red",
            alpha=0.3,
        )

    ax.set_ylabel(r"${\rm detector\;state}$", fontsize=12)
    ax.set_yticks([_UP, _DOWN], [r"${\rm UP}$", r"${\rm DOWN}$"])

    ax.grid(alpha=0.5)
    ax.set_xlabel(r"$t$", fontsize=12)

    plt.rcParams = original_rcParams
    return fig

def visualize_posterior(
    posterior_samples,
    labels=None,
    use_tex=True,
    truths=None,
    **kwargs,
):
    import corner
    _default_kwargs = {
        "label_kwargs": {"fontsize": 8},
        "labelpad": 0.25,
    }
    _default_kwargs.update(kwargs)

    if use_tex:
        plt.rcParams.update({
            "text.usetex": True,
        })
    else:
        plt.rcParams.update({
            "text.usetex": False,
        })

    fig = plt.figure(dpi=300)
    fig = corner.corner(
        posterior_samples,
        labels=labels,
        truths=truths,
        fig=fig,
        **_default_kwargs,
    )
    for ax in fig.get_axes():
        ax.tick_params(axis="both", which="major", labelsize=_default_kwargs["label_kwargs"]["fontsize"])

    return fig
