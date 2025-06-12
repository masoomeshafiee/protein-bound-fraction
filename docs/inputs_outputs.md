# Input and Output File Documentation

This document provides an overview of the required inputs and the generated outputs for the protein-bound fraction analysis pipeline. It includes expected formats, conventions, and a breakdown of key output CSV files.

---

## Inputs

### 1. Config File (`.json`)
- Specifies file paths, parameters, and module toggles for the pipeline.
- See [`config_doc.md`]

---

### 2. TrackMate Spot Files (`*_spots.csv`)
- One file per video, typically exported from Fiji/ [TrackMate](https://www.sciencedirect.com/science/article/pii/S1046202316303346)).
- Expected columns (no headers required):

| Column Name    | Description                         |
|----------------|-------------------------------------|
| `TRACK_ID`     | ID of the spot's track              |
| `"frame_number"`   | Time point in the corresponding track         |
| `x_loc`   | X coordinate in pixels              |
| `y_loc`   | Y coordinate in pixels              |
| `intensity`    | Spot intensity                      |

---

### 3. TrackMate Track Files (`*_tracks.csv`)
- One file per video, typically exported from Fiji/ [TrackMate](https://www.sciencedirect.com/science/article/pii/S1046202316303346)).
- Usually the track files have these coloumns: ["track_id", "spt_tr", "spt_widt", "mean_sp", "max_sp", "min_sp", "med_sp", "std_sp", "mean_q", "max_q", "min_q", "med_q", "std_q", "track_duration", "tr_start", "tr_fin", "x_loc", "y_loc"]`
The file could have those coloumns but the required columns are(no headers required):

| Column Name       | Description                          |
|-------------------|--------------------------------------|
| `track_id`        | Unique identifier for the track      |
| `track_duration`  | Total number of frames in the track  |
| `x_loc`           | Mean X coordinate of the track       |
| `y_loc`           | Mean Y coordinate of the track       |

> ⚠️ **Note**: Input CSV files do not need to include column headers. The pipeline assigns them based on `columns_names` specified in the config. Make sure the column **order** matches the expected structure.

---

### 4. Segmentation Masks (`.png`)
- Grayscale images with unique integer labels for each cell (0 = background).
- Used for per-cell analysis by mapping tracks to specific cell regions.
- Recommended tool: [Cellpose](https://www.cellpose.org/).

---

## Outputs

All outputs are saved in the directory specified by `output_config["output_dir"]`.

---

### Intermediate Results (`intermediate_results/`)

#### `processed_data.csv`
Filtered tracks with extracted motion features.

| Column Name            | Description                                  |
|------------------------|----------------------------------------------|
| `file_name`            | Name of the source input file                |
| `cell_id`              | Associated cell ID (if single-cell analysis enabled)        |
| `track_id`             | Unique ID of the track                       |
| `radius_of_gyration`   | Raw radius of gyration           |
| `log_rg`               | Log-transformed radius of gyration           |

---

#### `issues.csv`
Tracks or cells skipped due to validation or processing issues.

| Column Name  | Description                              |
|--------------|------------------------------------------|
| `file_name`  | Name of the affected input file          |
| `cell_id`    | Cell ID where the issue occurred (if single-cell analysis enabled)   |
| `issue`      | Description of the encountered issue     |

---

#### `spot_intensity_distribution.svg`
- Histogram of spot intensities across all files with Gaussian fit.

#### `track_length_distribution.svg`
- Histogram of track durations (in frames) with Gaussian fit.

---

### Final Outputs

#### `classified_tracks.csv`
Track-level motion classification.

| Column Name          | Description                                                |
|----------------------|------------------------------------------------------------|
| `file_name`          | Source input file                                          |
| `cell_id`            | Associated cell ID (if applicable)                         |
| `track_id`           | Unique ID of the track                                     |
| `radius_of_gyration` | Raw radius of gyration                                     |
| `log_rg`             | Log-transformed radius of gyration                         |
| `predicted_class`    | Motion class (e.g., 0 = bound, 1 = diffusive)              |
| `confidence`         | Posterior probability of assigned class                    |
| `status`             | Classification status (e.g., "classified", "low_confidence") |
**Note:** 
if the confidence threshold is set to low, all the tracks willbe assigned to a class. So if you dont want any undetermined track, set the confidence threshold to 0 in the config file. 
---

#### `classification_summary.csv`
Aggregate summary of classification across all files.

| Column Name    | Description                                  |
|----------------|----------------------------------------------|
| `sample_size`  | Total number of tracks analyzed              |
| `class`        | Motion class label (e.g., 0, 1)              |
| `count`        | Number of tracks in this class               |
| `percent`      | Percentage (fraction) of tracks in this class           |

---

#### `bound_fraction_per_cell.csv` *(if single-cell analysis is enabled)*
Bound fraction per segmented cell.

| Column Name        | Description                              |
|--------------------|------------------------------------------|
| `file_name`        | Name of the input file                   |
| `cell_id`          | Segmented cell ID                        |
| `n_tracks`         | Number of tracks mapped to this cell     |
| `bound_fraction`   | Estimated fraction of bound molecules    |

---

#### `gmm_model.joblib`
- A trained scikit-learn GaussianMixture model instance used for classifying .
This file contains the trained parameters of the Gaussian Mixture Model (GMM) that was fit to the extracted feature(s), such as the log-transformed radius of gyration (log_rg). It encapsulates the means, variances, weights, and convergence status of the model.

This allows you to:
- Reuse the trained model to classify new data without retraining.
- Inspect model parameters for publication or validation.
- Debug or retrace the classification logic in a transparent way.

The GMM is used in the pipeline to:
- Fit a mixture of Gaussian distributions to the input feature.
- Estimate the posterior probability of each data point belonging to each class.
- Assign motion classes based on the most likely Gaussian component.
---

#### Classification Plots
- Visualizations of GMM classification results, showing:
  - Histogram of the feature (e.g., `log_rg`)
  - Fitted Gaussian components
  - Decision thresholds (if enabled)
 
 **Note**: The `plot_thresholds` option is intended purely for visualization and interpretability. It displays threshold lines based on the Gaussian mixture model but **does not** affect the actual classification, which is determined by the posterior probabilities of the fitted GMM components.

---

### Metadata (`metadata/`)

| File Name             | Description                                                              |
|-----------------------|--------------------------------------------------------------------------|
| `config_used.json`    | Copy of the exact config file used for the run                           |
| `environment_used.txt`| Snapshot of the Python environment (`conda list` or `pip freeze`)        |
| `input_files.json`    | Mapping of all loaded input files                                        |

---

### Logging

#### `pipeline_log_<timestamp>.log`
- Full log of the pipeline run:
  - File loading status
  - Module progress
  - Any warnings or errors
  - Saved in the **main output directory**

---

## Tips
- Use consistent naming across spot, track, and mask files to ensure proper file matching. Consult the `file_name_handling_doc.md`.


