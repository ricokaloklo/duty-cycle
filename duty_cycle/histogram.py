import numpy as np

from .utils import find_contiguous_up_and_down_segments

def make_histogram_bin_edges(nbin):
    # For now we only allow equal-width bins
    return np.histogram_bin_edges([], bins=nbin)

def make_histograms(data, dt, bin_edges):
    binned_cont_up_times = np.zero(len(bin_edges) - 1)
    binned_cont_down_times = np.zero(len(bin_edges) - 1)

    for bit_ts in data:
        cont_up_time_idxs, cont_down_time_idxs = find_contiguous_up_and_down_segments(bit_ts)
        for idxs in cont_up_time_idxs:
            cont_up_times = (idxs[1] - idxs[0])*dt
        for idxs in cont_down_time_idxs:
            cont_down_times = (idxs[1] - idxs[0])*dt

        binned_cont_up_times += np.histogram(cont_up_times, bins=bin_edges)[0]
        binned_cont_down_times += np.histogram(cont_down_times, bins=bin_edges)[0]

    # Re-normalize the histograms
    binned_cont_up_times /= binned_cont_up_times.sum()
    binned_cont_down_times /= binned_cont_down_times.sum()

    return binned_cont_up_times, binned_cont_down_times