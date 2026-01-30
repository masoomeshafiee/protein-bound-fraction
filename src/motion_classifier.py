import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import joblib
import os
import logging
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===========================
# Core Classification Logic
# ===========================

def gmm_fit(data, n_components=2, random_state=42):
    """
    Fit a Gaussian Mixture Model to the data.
    
    Parameters:
    - data: np.ndarray, the data to fit the GMM to.
    - n_components: int, number of mixture components.
    - random_state: int, random state for reproducibility.
    
    Returns:
    - gmm: GaussianMixture object fitted to the data.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(data)
    return gmm

def compute_threshold(gmm_model):
    """Compute intersection points (thresholds) between each adjacent pair of Gaussian components."""
    means = gmm_model.means_.flatten()
    stds = np.sqrt(gmm_model.covariances_.flatten())
    weights = gmm_model.weights_.flatten()

    sorted_indices = np.argsort(means)
    thresholds = []

    for i in range(len(sorted_indices) - 1):
        idx1, idx2 = sorted_indices[i], sorted_indices[i + 1]

        def diff(x):
            p1 = weights[idx1] * norm.pdf(x, means[idx1], stds[idx1])
            p2 = weights[idx2] * norm.pdf(x, means[idx2], stds[idx2])
            return p1 - p2

        try:
            t = brentq(diff, means[idx1], means[idx2])
            thresholds.append(t)
            logging.info(f"Threshold between component {idx1} and {idx2}: {t:.4f}")

        except ValueError as e:
            thresholds.append((means[idx1] + means[idx2]) / 2)
            logging.warning(f"Could not compute threshold between components {idx1} and {idx2}, using mean. Error: {e}")
    return thresholds


def assign_classes(feature_df, gmm, feature = 'log_rg', confidence_level=0.9):
    """
    Assigns each track to a class based on a fitted multi-component GMM.

    Parameters:
    - feature_df: pd.DataFrame with chosen feature column.
    - gmm: Fitted GaussianMixture model (with ≥2 components)
    - feature: str, column name in feature_df to use for classification (default: 'log_rg')
    - confidence_level: Float, e.g. 0.9 — minimum probability to confidently assign class

    Returns:
    - classified_df: copy of feature_df with added coloumns ['predicted_class', 'confidence', 'status']
    - component_to_class: dict mapping component indices to class labels
    """
    if feature not in feature_df.columns:
        logging.error(f"Missing feature column: '{feature}' in input dataframe.")
        raise ValueError(f"Input DataFrame must contain {feature} column.")

    X = feature_df[feature].to_numpy().reshape(-1, 1)

    logging.info(f"Classifying {len(X)} data points using feature '{feature}'.")
    probs = gmm.predict_proba(X)
    hard_labels = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    # Map components to class labels based on ascending mean
    component_order = np.argsort(gmm.means_.flatten())
    class_labels = [f"class_{i}" for i in range(len(component_order))]  # customizable
    component_to_class = {int(comp_idx): class_labels[i] for i, comp_idx in enumerate(component_order)}

    logging.info(f"Component-to-class mapping: {component_to_class}")

    predicted_class = []
    status = []

    for i, comp_idx in enumerate(hard_labels):
        confidence = confidences[i]
        if confidence >= confidence_level:
            predicted_class.append(component_to_class[comp_idx])
            status.append("confident")
        else:
            predicted_class.append("uncertain")
            status.append("low_confidence")

    classified_df = feature_df.copy()
    classified_df['predicted_class'] = predicted_class
    classified_df['confidence'] = confidences
    classified_df['status'] = status
    
    logging.info(f"Completed class assignment. {sum(np.array(status)=='confident')} confident assignments.")
    return classified_df, component_to_class

def summarize_classification(classified_df, class_column="predicted_class"):
    """
    Computes summary statistics of classification results.

    Parameters:
    - classifed_df: DataFrame with a predicted_class column.
    - class_column: name of the column with class labels.

    Returns:
    - summary_df: DataFrame with sample-size and percentages.
    """
    if class_column not in classified_df.columns:
        logging.error(f"Column '{class_column}' not found in classified_df.")
        raise ValueError(f"Column '{class_column}' not found in input DataFrame.")

    total = len(classified_df)
    counts = classified_df[class_column].value_counts()
    summary_df = pd.DataFrame({
        'sample_size': total,
        'class': counts.index,
        'count': counts.values,
        'percent': (counts.values / total * 100).round(2)
    })
    logging.info(f"Classification summary:\n{summary_df}")
    return summary_df


def plot_classification_results(data, gmm, plot_settings, summary_df, thresholds=None, save_path=None):
    """
    Plot the histogram of feature values with GMM components and classification summary.

    Parameters:
    - data: np.array of feature values to plot.
    - gmm: Fitted GaussianMixture model.
    - plot_settings: dict with plot settings like title, xlabel, ylabel, etc.
    - summary_df: pd.DataFrame from summarize_classification with 'class' and 'percent' columns.
    - thresholds: List of floats (optional), threshold(s) between components.
    - save_path: Path to save the figure (optional). If None, plot is shown.
    """
    logging.info("Plotting classification results...")

    plt.figure(figsize=(10, 6))

    # Histogram
    counts, bins, patches = plt.hist(
        data,
        bins=plot_settings['bins'],
        density=True,
        alpha=0.5,
        color=plot_settings['hist_color'],
        label=plot_settings['hist_label']
    )

    # Overlay GMM components using ordered means (component with lowest mean = class_0)
    x = np.linspace(data.min(), data.max(), 1000)
    sorted_indices = np.argsort(gmm.means_.flatten())

    for i, comp_idx in enumerate(sorted_indices):
        mean = gmm.means_[comp_idx, 0]
        std = np.sqrt(gmm.covariances_[comp_idx, 0])
        weight = gmm.weights_[comp_idx]
        if len(sorted_indices)==2:
            plt.plot(
                x,
                weight * norm.pdf(x, mean, std),
                label=f"class_{i}" + (f"(bound)" if i==0 else "(diffusive)" )+ f" mean(log)={mean:.2f}" + f" mean={10 ** mean:.3f}" if plot_settings.get('log_scale', False) else "",
                linewidth=2
            )
        else:
            plt.plot(
                x,
                weight * norm.pdf(x, mean, std),
                label=f"class_{i}" + f" mean(log)={mean:.2f}" + f" mean={10 ** mean:.3f}" if plot_settings.get('log_scale', False) else "",
                linewidth=2
            )

    # Optional threshold lines
    if thresholds is not None:
        for t in thresholds:
            plt.axvline(t, color='red', linestyle='--', label=f"Threshold @ {t:.2f}")

    # Title with sample size and class proportions
    if summary_df is not None:
        total = summary_df['sample_size'].iloc[0]  # same across rows
        class_summary = ', '.join(
            f"{row['class']}: {row['percent']:.1f}%" for _, row in summary_df.iterrows()
        )
        title = f"{plot_settings['title']} (n = {total}): {class_summary}"
        plt.title(title, fontsize=plot_settings['title_fontsize'])
    else:
        plt.title(plot_settings['title'], fontsize=plot_settings['title_fontsize'])

    plt.xlabel(plot_settings['xlabel'], fontsize=plot_settings['label_fontsize'])
    plt.ylabel(plot_settings['ylabel'], fontsize=plot_settings['label_fontsize'])
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=300)
            logging.info(f"Plot saved to: {save_path}")
        except Exception as e:
            logging.warning(f"Failed to save plot: {e}")
    
    plt.show()

def calculate_bound_fraction_per_cell(classified_df, min_tracks=20, class_column="predicted_class",
                                      cell_column="cell_id", file_column="file_name", bound_class="class_0"):
    """
    Calculates the fraction of 'bound' tracks per cell, conditioned on a minimum track count.

    Parameters:
    - classified_df: pd.DataFrame with at least [file_column, cell_column, class_column]
    - min_tracks: int, minimum number of tracks per cell to be included
    - class_column: str, name of column with predicted classes (e.g. 'predicted_class')
    - cell_column: str, name of column with cell IDs
    - file_column: str, name of column with file names
    - bound_class: str, the class considered "bound" (e.g. 'class_0')

    Returns:
    - pd.DataFrame with [file_name, cell_id, n_tracks, bound_fraction]
    """
    logging.info("Calculating bound fraction per cell...")
    required_cols = {file_column, cell_column, class_column}
    missing = required_cols - set(classified_df.columns)
    if missing:
        logging.error(f"Missing required columns: {missing}")
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    grouped = classified_df.groupby([file_column, cell_column])

    records = []
    for (fname, cid), group in grouped:
        n_tracks = len(group)
        if n_tracks >= min_tracks:
            n_bound = (group[class_column] == bound_class).sum()
            bound_frac = n_bound / n_tracks
            records.append({
                file_column: fname,
                cell_column: cid,
                "n_tracks": n_tracks,
                "bound_fraction": bound_frac
            })
        else:
            logging.info(f"Skipped cell {cid} in file {fname}: only {n_tracks} tracks found(min required = {min_tracks})")

    result_df = pd.DataFrame(records)        
    logging.info(f"Calculated bound fractions for {len(result_df)} cells (with >= {min_tracks} tracks out of {len(grouped)} cells).")
    return result_df



def plot_bound_fraction_per_cell(cell_df, save_path=None):
    """
    Plot a single boxplot (with individual points) of bound fraction across all cells.

    Parameters:
    - cell_df: DataFrame with columns ['file_name', 'cell_id', 'bound_fraction']
    - save_path: Path to save the figure (optional)
    """
    logging.info("Plotting bound fraction per cell...")

    if cell_df.empty:
        logging.warning("Input DataFrame for plotting is empty.")
        return
    # Add a dummy column for unified x-axis label
    cell_df = cell_df.copy()
    cell_df['label'] = "Condition"

    plt.figure(figsize=(6, 6))
    ax = sns.boxplot(data=cell_df, x="label", y="bound_fraction", color='lightblue')
    sns.stripplot(data=cell_df, x="label", y="bound_fraction", color='gray', size=4, jitter=True)

    plt.title("Bound Fraction per Cell")
    plt.ylabel("Bound Fraction")
    plt.xlabel("")
    plt.ylim(0, 1)
    plt.tight_layout()
    
    if save_path:
        try: 
            plt.savefig(save_path, dpi=300)
            logging.info(f"Bound fraction plot saved to: {save_path}")
        except Exception as e:
            logging.warning(f"Failed to save plot: {e}")
    
    plt.show()


    # ===============================
    # Public API
    # ===============================
    
def classify_tracks(processed_data, classification_config, plot_config, output_config, single_cell_config, random_state=42, single_cell_analysis=False):
    """
    Classify tracks using GMM.

    Parameters:
    - processed_data: pd.DataFrame with required classification features. Example:'radius_of_gyration' and 'log_rg' column.
    - classification_config: dict with keys 'n_components' (int) and 'confidence_level' (float).
    - plot_config: dict with keys 'plot_results' (bool), 'plot setting' (str, optional).
    - output_config: 'save_plot'(bool), output_path

    Returns:
    - gmm: Fitted GaussianMixture model.
    - classified_df: pd.DataFrame with additional columns ['predicted_class', 'confidence', 'status'].
    _ summary_df: pd.DataFrame with classification summary statistics.
    - cell_records: pd.DataFrame with bound fraction per cell if single_cell_analysis is True.

    """
    logging.info("Starting GMM fiting and classification ...")
    fit_new_gmm = classification_config.get('fit_new_gmm', True)
    n_components = classification_config['n_components']
    feature = classification_config['feature_to_classify']
    confidence_level = classification_config['confidence_level']
    plot_results = plot_config.get('plot_results', True)
    plot_settings = plot_config['plot_settings']
    output_dir = output_config['output_dir']

    # Fit GMM to log-transformed of selected feature data
    try:
        feature_data = processed_data[feature].to_numpy()
    except KeyError:
        logging.error(f"'{feature}' column not found in processed_data.")
        raise
    
    if fit_new_gmm:
        gmm = gmm_fit(feature_data.reshape(-1, 1), n_components, random_state)
        logging.info(f"GMM model fitted with {n_components} components.")
    else:
        # Load a gmm model
        gmm_path = classification_config['gmm_model_path']
        gmm = joblib.load(gmm_path)
        logging.info("Loaded existing GMM model.")
        
        # Check if the GMM has the expected number of components
        if gmm.n_components != n_components:
            logging.error(f"Expected GMM with {n_components} components, but got {gmm.n_components}.")
            raise ValueError(f"Expected GMM with {n_components} components, but got {gmm.n_components}.")
        

    # Assign classes to tracks based on GMM
    classified_df, component_to_class = assign_classes(processed_data, gmm, feature, confidence_level)

    # Summarize classification results
    summary_df = summarize_classification(classified_df, "predicted_class")

    # Plot results if specified
    if plot_results:
        save_path = os.path.join(output_dir, output_config['classification_plot_name'])
        if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        if plot_config['plot_thresholds']:
            thresholds = compute_threshold(gmm)
            plot_classification_results(feature_data.flatten(), gmm, plot_settings, summary_df, thresholds=thresholds, save_path=save_path)
        else:
            plot_classification_results(feature_data.flatten(), gmm, plot_settings, summary_df, save_path=save_path)
        

    # single cell analysis
    # Dynamically resolve bound class
    bound_class = component_to_class[np.argmin(gmm.means_)]
    if single_cell_analysis:
        logging.info("Performing single-cell analysis...")
        min_tracks_per_cell = single_cell_config['min_tracks_per_cell']
        cell_records = calculate_bound_fraction_per_cell(classified_df, min_tracks=min_tracks_per_cell, class_column="predicted_class",
                                      cell_column="cell_id", file_column="file_name", bound_class=bound_class)
        if single_cell_config['box_plot']:
            save_path = os.path.join(output_dir, output_config['single_cell_plot_name'])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plot_bound_fraction_per_cell(cell_records, save_path=save_path)
        
        return gmm, classified_df, summary_df, cell_records
    
    return gmm, classified_df, summary_df


