import numpy as np
import random
import pandas as pd
from scipy import fftpack
from collections import defaultdict


def compute_spectral_entropy_global(time_series, n_fft=None):
    """
    Compute the spectral entropy of a time series to estimate its periodicity strength.
    Lower entropy → stronger periodicity; higher entropy → weaker periodicity.

    Args:
        time_series (array-like): Raw timestamps of interactions.
        n_fft (int, optional): Number of FFT points. Defaults to min(len(time_series), 2048).

    Returns:
        float: Normalized spectral entropy in [0, 1].
    """
    if len(time_series) < 2:
        return 0.5  # Default medium periodicity

    # Compute time intervals
    time_diffs = np.diff(time_series)

    # Normalize intervals
    time_diffs = (time_diffs - np.mean(time_diffs)) / (np.std(time_diffs) + 1e-10)

    # Compute FFT and power spectral density
    if n_fft is None:
        n_fft = min(len(time_diffs), 2048)

    fft_vals = np.fft.fft(time_diffs, n=n_fft)
    psd = np.abs(fft_vals) ** 2
    psd_norm = psd / psd.sum()

    # Spectral entropy
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

    # Normalize entropy to [0, 1]
    max_entropy = np.log(n_fft)
    normalized_entropy = entropy / max_entropy
    return normalized_entropy


def analyze_periodicity_global(node_interact_times):
    """
    Analyze global periodicity of interaction timestamps and compute a recommended p value.

    Args:
        node_interact_times (array-like): Interaction timestamps for all nodes.

    Returns:
        tuple: (spectral_entropy, recommended_p)
    """
    sorted_times = np.sort(node_interact_times)

    entropy = compute_spectral_entropy_global(sorted_times)

    # Map entropy → p (more periodic → higher p)
    recommended_p = 1 - entropy
    recommended_p = max(0.1, min(0.9, recommended_p))

    return entropy, recommended_p


def compute_spectral_entropy_per_node(time_series, n_fft=None):
    """
    Compute spectral entropy for a single node's interaction sequence.
    Returns NaN for invalid sequences (constant timestamps or too short).

    Args:
        time_series (array-like): Sorted timestamps for a node.

    Returns:
        float or np.nan: Spectral entropy in [0,1] or NaN if invalid.
    """
    if len(time_series) < 3:
        return np.nan

    time_diffs = np.diff(time_series)

    # Skip nodes with constant timestamps
    if np.allclose(time_diffs, 0):
        return np.nan

    # Normalize intervals
    time_diffs = (time_diffs - np.mean(time_diffs)) / (np.std(time_diffs) + 1e-10)

    if n_fft is None:
        n_fft = min(len(time_diffs), 2048)

    fft_vals = np.fft.fft(time_diffs, n=n_fft)
    psd = np.abs(fft_vals) ** 2
    psd_sum = np.sum(psd)

    if psd_sum == 0 or np.isnan(psd_sum):
        return np.nan

    psd_norm = psd / psd_sum
    psd_norm = np.clip(psd_norm, 1e-12, 1.0)

    entropy = -np.sum(psd_norm * np.log(psd_norm))
    normalized_entropy = entropy / np.log(len(psd_norm))

    if np.isnan(normalized_entropy) or np.isinf(normalized_entropy):
        return np.nan

    return float(np.clip(normalized_entropy, 0, 1))


def analyze_periodicity_per_node(src_node_ids, dst_node_ids, node_interact_times):
    """
    Node-level periodicity analysis.
    Computes spectral entropy for each node separately and aggregates.

    Nodes with invalid timestamps or constant-time interactions are skipped.

    Args:
        src_node_ids (array-like): Source nodes in interactions.
        dst_node_ids (array-like): Destination nodes in interactions.
        node_interact_times (array-like): Interaction timestamps.

    Returns:
        tuple: (average_entropy, recommended_p)
    """
    node_times = defaultdict(list)

    # Collect timestamps per node
    for u, v, t in zip(src_node_ids, dst_node_ids, node_interact_times):
        node_times[u].append(t)
        node_times[v].append(t)

    node_entropies = []
    for node, times in node_times.items():
        times = np.sort(np.array(times))
        entropy = compute_spectral_entropy_per_node(times)

        if np.isnan(entropy):
            continue

        node_entropies.append(entropy)

    avg_entropy = np.mean(node_entropies) if len(node_entropies) > 0 else 0.5

    recommended_p = 1 - avg_entropy
    recommended_p = max(0.1, min(0.9, recommended_p))

    return avg_entropy, recommended_p
