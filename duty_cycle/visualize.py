import numpy as np
from matplotlib import pyplot as plt

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
    N = len(simulation_output)
    ax.plot(np.arange(N)/N, simulation_output)
    ax.grid(alpha=0.5)
    ax.set_xlabel(r"$t/T$")
    
    return fig