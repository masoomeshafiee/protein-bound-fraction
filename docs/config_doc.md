# Configuration File Documentation

This document describes the structure and parameters of the configuration file used in the protein-bound fraction analysis pipeline. The configuration controls various aspects of data input, filtering, feature extraction, classification, plotting, and output generation.

---

## 1. `quality_control`
- **Type**: `bool`
- **Description**: If set to `true`, the pipeline will generate quality control plots (e.g., track length distribution, spot intensity distribution) before analysis begins. This is to ensure that the quality of data is consistent accross different batches or experiments. 

---

## 2. `path`
Defines all input/output paths and file suffixes used by the pipeline.

| Key            | Type     | Description |
|----------------|----------|-------------|
| `input_dir`    | `str`    | Path to the directory containing input track and spot CSV files. |
| `masks_dir`    | `str`    | Path to the directory containing cell segmentation mask images. |
| `spot_suffix`  | `str`    | Filename suffix (end of file name) used to identify spot CSV files (both for the pipeline and the `file_name_handling.py`). |
| `track_suffix` | `str`    | Filename suffix (end of file name) used to identify track CSV files (both for the pipeline and the `file_name_handling.py`). |
| `mask_suffix`  | `str`    | Filename suffix (end of file name) used to match corresponding mask files (both for the pipeline and the `file_name_handling.py`). |
| `dry_run`      | `bool`   | If `true`, the `file_name_handling.py` script simulates the file name changing without actually changing the file name. Useful to make sure the file names will be changed as desired.

---

## 3. `columns_names`
Specifies expected column names with the correct order in the spot and track CSV files.

| Dataset | Columns |
|---------|---------|
| `spots` | `["track_id", "frame_number", "x_loc", "y_loc", "intensity"]` |
| `tracks` | `["track_id", "spt_tr", "spt_widt", "mean_sp", "max_sp", "min_sp", "med_sp", "std_sp", "mean_q", "max_q", "min_q", "med_q", "std_q", "track_duration", "tr_start", "tr_fin", "x_loc", "y_loc"]` |

Ensure these match the format of your TrackMate or tracking output files.

---

## 4. `filtering`
Criteria for filtering tracks prior to analysis.

| Key                 | Type  | Description |
|---------------------|-------|-------------|
| `min_track_duration` | `int` | Minimum duration (in frames) a track must have to be included in the analysis. |

---

## 5. `feature_extraction`
Specifies which features should be extracted from tracks.

| Key           | Type     | Description |
|---------------|----------|-------------|
| `feature_list` | `list[str]` | List of features to extract. Supported: `"radius_of_gyration"` |

---

## 6. `classification`
Settings for classifying molecular motion using a Gaussian Mixture Model (GMM).

| Key                   | Type      | Description |
|------------------------|-----------|-------------|
| `n_components`         | `int`     | Number of Gaussian components to fit. Typically 2 for "bound" vs "diffusive". |
| `confidence_level`     | `float`   | Confidence threshold for classification. Values range from 0.0 to 1.0. If 0.0 the confidence is not considered.  |
| `feature_to_classify`  | `str`     | Feature to use for classification. Typically `"log_rg"` (log radius of gyration). |
| `fit_new_gmm`  | `bool`     | If `true`, the data will be fitted to a new gmm, if `false` the script loads an existing gmm model for classification . |
| `gmm_model_path`  | `str`     | path to an existing model that you want to use for current classification ( if `fit_new_gmm` = `false`). otherwise you can leave it with an arbitrary string such as "NA", it does not affect the pipeline. |

---

## 7. `plot`
Configuration for plotting results.

| Key               | Type        | Description |
|--------------------|-------------|-------------|
| `plot_results`     | `bool`      | Whether to generate plots for classification results. |
| `plot_thresholds`  | `bool`      | Whether to display threshold lines on the plot for interpretability.  |
| `plot_settings`    | `dict`      | Detailed appearance settings for histogram and classification plots. See below. |
**Note**: The `plot_thresholds` option is intended purely for visualization and interpretability. It displays threshold lines based on the Gaussian mixture model but **does not** affect the actual classification, which is determined by the posterior probabilities of the fitted GMM components.

### `plot_settings`
- `bins`: Number of bins in the histogram.
- `hist_color`: Color of the histogram bars.
- `hist_label`: Label for histogram data.
- `title`: Title of the classification plot.
- `xlabel`: Label for the x-axis (e.g., log radius of gyration).
- `ylabel`: Label for the y-axis (e.g., density).
- `title_fontsize`: Font size of the plot title.
- `label_fontsize`: Font size for axis labels.

---

## 8. `output_config`
Controls the output location and filenames for results.

| Key                         | Type     | Description |
|-----------------------------|----------|-------------|
| `output_dir`                | `str`    | Directory where results (CSV files, plots) will be saved. |
| `single_cell_plot_name`     | `str`    | Filename for the per-cell classification box plot. |
| `classification_plot_name`  | `str`    | Filename for the GMM classification plot. |

---

## 9. `single_cell`
Settings for analyzing and visualizing single-cell classification results.

| Key                   | Type   | Description |
|------------------------|--------|-------------|
| `box_plot`             | `bool` | If `true`, generates a single box plot of bound fractions across cells. |
| `min_tracks_per_cell`  | `int`  | Minimum number of tracks per cell required to be included in single-cell analysis. |

---

## 10. `analyze_single_cell`
- **Type**: `bool`
- **Description**: If `true`, the pipeline performs analysis at the single-cell level using segmentation masks and computes per-cell bound fractions.

---

## Notes
- Ensure that input file names follow consistent naming conventions that match the provided suffixes. You can use the file_name_handling.py to change the spots and tracks base name to match the corresponding mask base name. 
- The configuration file must be saved as JSON (`.json`) and loaded at the beginning of the pipeline execution. You can use the config.json file and change it as you wish. A copy of your config file will be saved in the output directory for reproducibility. 

