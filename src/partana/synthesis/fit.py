import numpy as np
from scipy.stats import linregress

def compute_histogram_and_centers(data, start=10, end=200, bin1=1, bin2=5, breakpoint=100):
    """
    Computes histogram counts and bin centers for a given data array using custom bin sizes.

    Parameters:
    - data: array-like, the input data for which the histogram is computed.
    - start: int, the starting value for the first bin (default is 25).
    - end: int, an approximate ending value for binning (default is 130).

    Returns:
    - counts: array, the counts of data points in each bin.
    - bin_centers: array, the center values of each bin.
    """

    if data.max() > end:
        end = data.max()

    # Define bin edges: 1-width bins from start to breakpoint, and 5-width bins from breakpoint onwards
    bins_1 = np.arange(start, breakpoint, bin1)  # Bins from start to breakpoint with width 1
    bins_2 = np.arange(breakpoint, end + 5, bin2)  # Bins from breakpoint to end with width 5
    bins = np.concatenate((bins_1, bins_2))  # Combine the two bin arrays

    # Calculate histogram data
    counts, bin_edges = np.histogram(data, bins=bins)

    # Calculate bin centers for plotting
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return counts/(bin_edges[1:]-bin_edges[:-1]), bin_centers

def fit_log_log_line(bin_centers, counts):
    # Filter out zero counts to avoid log(0) issues
    valid_indices = counts > 0
    log_bin_centers = np.log(bin_centers[valid_indices])
    log_counts = np.log(counts[valid_indices])

    # Perform linear regression on the log-log data
    slope, intercept, r_value, p_value, std_err = linregress(log_bin_centers, log_counts)

    # Compute the fitted line in the original space
    fitted_line = np.exp(intercept) * bin_centers ** slope

    return slope, intercept, fitted_line
