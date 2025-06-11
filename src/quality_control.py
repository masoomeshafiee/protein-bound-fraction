import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def _plot_histogram_with_fit(data_array, title, xlabel, intermediate_dir, filename_base, bins=50, color='skyblue', fit_gaussian=True, xlim=None, density=True):
    """Helper to plot histogram (optionally with Gaussian fit) and save it."""
    sample_size = len(data_array)
    mu, std = norm.fit(data_array) if fit_gaussian else (None, None)

    plt.figure(figsize=(8, 5))
    count, bins_edge, _ = plt.hist(data_array, bins=bins, density=density, color=color, alpha=0.6, label='Data')

    if fit_gaussian and density:
        x = np.linspace(min(data_array), max(data_array), 1000)
        plt.plot(x, norm.pdf(x, mu, std), 'r--', label=f'Gaussian Fit\nμ={mu:.2f}, σ={std:.2f}, n={sample_size}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density' if density else 'Count')
    plt.legend()

    if xlim is not None:
        plt.xlim(xlim)

    plt.tight_layout()

    # Save in both PNG and SVG formats
    os.makedirs(intermediate_dir, exist_ok=True)
    for ext in ['png', 'svg']:
        path = os.path.join(intermediate_dir, f"{filename_base}.{ext}")
        plt.savefig(path)

    plt.show()
    plt.close()

    return {
        "mean": mu,
        "std": std,
        "n": sample_size,
    }

def plot_track_length_distribution(data, intermediate_dir):
    """
    Plot and save distribution of track lengths (no Gaussian fit).
    """
    track_lengths = []
    for sample in data.values():
        if 'tracks' in sample:
            track_lengths.extend(sample['tracks']['track_duration'].dropna().values)

    return _plot_histogram_with_fit(
        np.array(track_lengths),
        title="Histogram of Track Lengths",
        xlabel="Track Duration (frames)",
        intermediate_dir=intermediate_dir,
        filename_base="track_length_distribution",
        fit_gaussian=False,  
        xlim=(0, 60), density=False          # Manually cap x-axis for better visibility
    )


def plot_intensity_distribution(data, intermediate_dir):
    """
    Plot and save distribution of spot intensities with Gaussian fit.
    """
    intensities = []
    for sample in data.values():
        if 'spots' in sample:
            intensities.extend(sample['spots']['intensity'].dropna().values)

    return _plot_histogram_with_fit(
        data_array=np.array(intensities),
        title="Distribution of Spot Intensities",
        xlabel="Spot Intensity",
        intermediate_dir=intermediate_dir,
        filename_base="spot_intensity_distribution",
        fit_gaussian=True
    )
