## `pipeline.py` 
This is the main orchestration script for the protein bound fraction pipeline. It performs data loading, quality control, filtering, feature extraction, classification, and saves all intermediate and final results along with metadata for reproducibility.

For inputs and outouts details consult `inputs_outputs.md`

**Usage**
``` bash
python pipeline.py --config path/to/config.json
```
##### **Main Function: main(config_path)**
Workflow Steps:
1. Load Configuration
2. Initialize Logging
3. Load Raw Data using load_data_from_config
4. (Optional) Quality Control Plots:
5. Filter Tracks & Spots
6. Feature Extraction (e.g., radius of gyration)
7. Classify Tracks using a Gaussian Mixture Model
8. Save Results (CSV, GMM model)
9. Save Metadata (config, environment, input files)

| Module               | Purpose                                                 |
| -------------------- | ------------------------------------------------------- |
| `load_data`          | Loads raw TrackMate CSVs and masks                               |
| `filter_data`        | Filters short tracks and non-masked areas |
| `feature_extraction` | Computes features like radius of gyration               |
| `motion_classifier`  | GMM-based classification of molecular motion            |
| `save_metadata`      | Stores pipeline config, pip environment, and input list |
| `quality_control`    | Generates QC plots for spot intensity and track length  |
| `data_validation`    | Validate the data strucutre and integriy during the analysis  |

**Logs are saved to:**
```
<output_dir>/pipeline_log_YYYYMMDD_HHMMSS.log
```
They include:
- Status of each pipeline step
- Errors and warnings
- Paths to saved files

**Error Handling**
Any exception during pipeline execution is logged with a full traceback. If an error occurs, the pipeline stops and raises the exception.

---
## `load_data.py` — Load Tracking, Spot, and Mask Data

This module is responsible for loading and organizing input data required by the pipeline, including:

- **Spot files** (CSV) from TrackMate
- **Track files** (CSV) from TrackMate
- **Segmentation masks** (PNG) generated by tools like Cellpose

It performs rigorous validation and logging to ensure that the required input files exist, match in naming, and are correctly formatted.
This module is typically called early in the pipeline to load all necessary input data into memory, structured for downstream modules like `filter_data.py`, `feature_extraction.py`, and `motion_classifier.py`.

### Function: 
`load_data_from_config(path, columns_names)`

 **Purpose**
Loads input datasets (spots, tracks, and masks) from specified directories using file name suffix conventions. It associates matching file triplets (spot, track, mask) by shared base name. If the dont share the same base name, it will raise an error. So make sure you use the 'file_name_handling.py` in advance. 


### Parameters

| Name            | Type   | Description |
|-----------------|--------|-------------|
| `path`          | `dict` | Dictionary containing keys such as: `input_dir`, `masks_dir`, `spot_suffix`, `track_suffix`, and `mask_suffix`. |
| `columns_names` | `dict` | Dictionary with two keys: `"spots"` and `"tracks"`, each containing an ordered list of column names for their respective CSVs. |

### Related config example
{
  "path": {
    "input_dir": "data/raw/",
    "masks_dir": "data/masks/",
    "spot_suffix": "_spots.csv",
    "track_suffix": "_tracks.csv",
    "mask_suffix": "_mask.png"
  },
  "columns_names": {
    "spots": ["TRACK_ID", "POSITION_T", "POSITION_X", "POSITION_Y", "INTENSITY"],
    "tracks": ["track_id", "track_duration", "x_loc", "y_loc"]
  }
}


### Returns

| Key          | Type                 | Description |
|--------------|----------------------|-------------|
| `data`       | `dict`               | A nested dictionary keyed by base file names. Each entry contains:  `spots`: DataFrame of spot data - `tracks`: DataFrame of track data - `mask`: 2D NumPy array (image) of the segmentation mask |

### Features

- Automatically matches `spots`, `tracks`, and `mask` files using base name convention.
- Assigns column names to raw CSVs (which may lack headers).
- Performs validation:
  - Checks for directory existence
  - Verifies file availability
  - Validates column name structure and type
- Logs all progress, file reads, and errors for transparency and debugging.


### Error Handling

Raises descriptive exceptions when:
- Required directories are missing
- Expected files are not found or mismatched
- Column name formatting is incorrect
- A matching file (e.g., mask for a given spot/track pair) is missing


### Example Output Structure

```python
{
  "sample01": {
    "spots": <pd.DataFrame>,
    "tracks": <pd.DataFrame>,
    "mask": <np.ndarray>
  },
  "sample02": {
    "spots": <pd.DataFrame>,
    "tracks": <pd.DataFrame>,
    "mask": <np.ndarray>
  },
  ...
}
```
---
## `quality_control.py` — Quality Control Visualization Module

This module provides standardized visualizations to assess data quality for microscopy tracking data before feature extraction or classification. It includes:

- Track duration distribution plotting
- Spot intensity distribution plotting with optional Gaussian fit


##### Internal Helper: `_plot_histogram_with_fit(...)`

**Purpose**
Creates and saves a histogram (optionally overlaid with a Gaussian fit) for any 1D data array.
**Returns**
- `dict`: Summary statistics of the data
  - `"mean"`: Mean of Gaussian fit (if applied)
  - `"std"`: Standard deviation of Gaussian fit
  - `"n"`: Sample size

- Saves plots to both `.png` and `.svg` in `intermediate_dir`
- Displays the plot via `matplotlib.pyplot.show`


##### Function: `plot_track_length_distribution(data, intermediate_dir)`
**Purpose**
Plots the distribution of track durations across all files in the dataset.

**Plot Details**
- Histogram of `"track_duration"` from all tracks
- X-axis limited to `(0, 60)` frames for visualization clarity
- File saved as:
  - `track_length_distribution.png`
  - `track_length_distribution.svg`


##### Function: `plot_intensity_distribution(data, intermediate_dir)`
**Purpose**
Plots the distribution of spot intensities with a Gaussian fit to check illumination consistency or signal-to-noise ratio.

**Parameters**
| Name               | Type         | Description                                  |
|--------------------|--------------|----------------------------------------------|
| `data`             | `dict`       | Dictionary of filtered data per file         |
| `intermediate_dir` | `str`        | Output directory for saving the histogram    |

**Returns**
- Summary `dict` with mean, std, and sample size from the Gaussian fit

**Plot Details**
- Histogram of `"intensity"` from all spots
- Gaussian curve overlaid with mean (`μ`) and standard deviation (`σ`)
- File saved as:
  - `spot_intensity_distribution.png`
  - `spot_intensity_distribution.svg`


##### Notes

- These plots help verify experimental consistency and guide parameter tuning.
- Gaussian fit is useful for identifying bimodal distributions, skew, or outliers.

---

## `data_validation.py` — Data and Mask Validation + Issue Logging

This module provides utility functions for validating input data structures (DataFrames and masks) and for logging and tracking data quality issues throughout the analysis pipeline.

It ensures that:
- Inputs conform to expected structure before downstream analysis.
- Data issues are logged both to the console and internally in a structured dictionary (`issue_dict`) for report generation or debugging.


### Function: `validate_dataframe(df, expected_columns)`

**Purpose**
Checks whether a given `pandas.DataFrame`:
- Is a valid DataFrame object
- Is not empty
- Contains all expected column names

**Parameters**
| Name               | Type           | Description                                |
|--------------------|----------------|--------------------------------------------|
| `df`               | `pd.DataFrame` | Input DataFrame to be validated            |
| `expected_columns` | `list`         | List of required column names              |

**Returns**
- `Tuple[bool, str]`: `(True, "DataFrame is valid")` or `(False, "Reason for failure")`

### Function: `validate_mask(mask)`

**Purpose**
Checks whether a given segmentation mask:
- Is a valid 2D NumPy array
- Is not empty
- Contains nonzero values (i.e., actual cell segmentation)

**Parameters**
| Name     | Type         | Description                       |
|----------|--------------|-----------------------------------|
| `mask`   | `np.ndarray` | Input segmentation mask to check  |

**Returns**
- `Tuple[bool, str]`: `(True, "Mask is valid")` or `(False, "Reason for failure")`


### Function: `log_issue(issue_dict, file_name, cell_id=None, message="", level='warning')`
**Purpose**
Logs an issue during data processing, both:
- To the Python logger (console)
- Into a nested issue_dict object for structured error tracking

**Example Structure of issue_dict**
```python
{
  "sample01.csv": {
    "file": "DataFrame is empty.",
    "cell_3": "Mask has no cells."
  },
  "sample02.csv": {
    "file": "Missing columns: {'track_id'}"
  }
}
```
**Integration**
**Upstream use**: Often used immediately after load_data.py to ensure files were read correctly.
**Downstream impact**: Prevents malformed data from entering critical modules like filter_data.py or feature_extraction.py.


**Example Usage**

```python
valid, msg = validate_dataframe(df, expected_columns=["track_id", "x", "y"])
if not valid:
    log_issue(issue_dict, file_name="sample1", message=msg, level='error')
```
---
## `filter_data.py` — Track and Spot Filtering Based on Masks and Duration

This module filters:
- Short tracks (below a user-defined duration)
- Tracks and spots that fall outside segmented cell masks

It supports both **population-level** and **single-cell-level** analysis and logs all issues for traceability.



##### Function: `filter_tracks_by_length(tracks, min_track_length)`

**Purpose**
Filters tracks based on a minimum duration.

**Parameters**
| Name               | Type           | Description                            |
|--------------------|----------------|----------------------------------------|
| `tracks`           | `pd.DataFrame` | Track data for a single            |
| `min_track_length` | `int`          | Minimum track duration to retain       |

**Returns**
- Filtered `DataFrame` of tracks



##### Function: `filter_tracks_by_mask(tracks, mask)`

**Purpose**
Filters tracks that fall within a given 2D mask (population or per-cell).

**Parameters**
| Name     | Type           | Description                               |
|----------|----------------|-------------------------------------------|
| `tracks` | `pd.DataFrame` | Track data containing `x_loc` and `y_loc` |
| `mask`   | `np.ndarray`   | Binary or labeled mask image              |

**Returns**
- Filtered `DataFrame` of in-mask tracks



##### Function: `filter_spots(spots, filtered_tracks)`

**Purpose**
Filters spot records to retain only those associated with the filtered tracks.

**Parameters**
| Name             | Type           | Description                          |
|------------------|----------------|--------------------------------------|
| `spots`          | `pd.DataFrame` | Original spot data                   |
| `filtered_tracks`| `pd.DataFrame` | Tracks after length/mask filtering   |

**Returns**
- Filtered `DataFrame` of spots



##### Function: `filter_population_level(...)`

**Purpose**
Performs end-to-end filtering of tracks and spots for each file using a single mask for all cells.

 **Returns**
- `dict` of filtered `tracks`, `spots`, and `mask` or empty dict on failure

**Common Issues Logged**
- Empty tracks after duration or mask filtering
- Empty spots corresponding to filtered tracks
- Missing or malformed data structures



##### Function: `filter_single_cell_level(...)`

**Purpose**
Filters tracks and spots for each **individual cell** using labeled mask values.

**Returns**
- `dict` in format `{cell_id: {'tracks': ..., 'spots': ..., 'mask': ...}}` or empty dict

**Additional Behavior**
- Iterates over unique `cell_id`s in the mask
- Skips cells with no qualifying tracks or spots
- Logs warnings for each empty cell individually



##### Public API: `filter_tracks_and_spots(...)`

**Purpose**
Main entry point to filter a dataset either at population or single-cell level.

**Parameters**
| Name                | Type    | Description                                                                 |
|---------------------|---------|-----------------------------------------------------------------------------|
| `data`              | `dict`  | Raw input data per file: `{file_name: {'spots', 'tracks', 'mask'}}`         |
| `min_track_length`  | `int`   | Minimum duration of tracks to keep                                          |
| `columns_names`     | `dict`  | Expected columns for validation: `{'spots': [...], 'tracks': [...]}`        |
| `analyze_single_cell`| `bool` | Whether to run per-cell (True) or population-level (False) filtering        |

**Returns**
- `filtered`: Filtered data dictionary (population or cell-level)
- `issue_dict`: Dictionary logging all issues, keyed by file and optionally cell ID

 **Raises**
- `ValueError` if no files contain valid data after filtering



#### Example Output Structure

```python
filtered = {
    'file01.csv': {
        1: {'tracks': df1, 'spots': df1s, 'mask': m1},
        2: {'tracks': df2, 'spots': df2s, 'mask': m2},
        ...
    },
    'file02.csv': {
        'tracks': df, 'spots': dfs, 'mask': m  # if population-level
    }
}
```
**Integration**
Upstream: Expects validated data structures loaded by `load_data.py` and checked by data_validation.py
Downstream: Provides clean input for `feature_extraction.py`, `motion_classifier.py`, etc.
---
## `feature_extraction.py` — Track-Level Feature Extraction Module

This module computes track-level features from spot data.
Currently focusing on the **radius of gyration**, but scalable to extract other features in the future.



###### `radius_of_gyration(spots_within_track, x='x_loc', y='y_loc')`

**Purpose**
Calculates the radius of gyration (Rg) for a track, a spatial dispersion metric defined as:

radius_of_gyration = sqrt(1/N * sum((x_i - x_c)^2 + (y_i - y_c)^2))

Where:
- \( x_i, y_i \): spot coordinates
- \( x_c, y_c \): center of mass of the track


###### `extract_rg(file_name, data_block, issue_dict, columns_names, cell_id=None)`

**Purpose**
Extracts radius of gyration and its log-transformed version for each track in a given data block (whole image or single cell).

**Parameters**
| Name          | Type           | Description                                                  |
|---------------|----------------|--------------------------------------------------------------|
| `file_name`   | `str`          | Filename associated with the data block                      |
| `data_block`  | `dict`         | Must contain a `'spots'` DataFrame                           |
| `issue_dict`  | `dict`         | Logs issues by file and (optionally) cell                    |
| `columns_names` | `dict`       | Column name mappings for validation                          |
| `cell_id`     | `str` or None  | Optional ID if analyzing individual cells                    |

**Returns**
- `pd.DataFrame` with:
  - `file_name`
  - `cell_id` (if applicable)
  - `track_id`
  - `radius_of_gyration`
  - `log_rg` (log-transformed Rg)

**Failure Handling**
- Logs and skips if:
  - No `spots` found
  - Validation fails
  - Track data is missing or malformed
  - Radius of gyration calculation fails



##### `extract_features(filtered_data, issue_dict, columns_names, feature_list, analyze_single_cell=False)`

**Purpose**
Top-level API to extract features (currently `radius_of_gyration`) from filtered tracking data, supporting both bulk and single-cell analysis.

**Parameters**
| Name                | Type             | Description                                                                 |
|---------------------|------------------|-----------------------------------------------------------------------------|
| `filtered_data`     | `dict`           | Preprocessed data: file → data block (or cell_id → data block)             |
| `issue_dict`        | `dict`           | Mutable issue logger                                                       |
| `columns_names`     | `dict`           | Expected column names for spots and tracks                                 |
| `feature_list`      | `list[str]`      | Features to extract (currently only `'radius_of_gyration'` supported)      |
| `analyze_single_cell` | `bool`         | If True, extract features per cell, otherwise per file                     |

**Returns**
- `pd.DataFrame`: Flattened table of extracted features
- `issue_dict`: Updated issue logger

**Notes**
- Filters out unsupported features automatically
- Logs and skips files or cells with missing/invalid structures
- Adds `log_rg` for better numerical stability in later modeling



####  Example Output Columns
| file_name | cell_id | track_id | radius_of_gyration | log_rg |
|-----------|---------|----------|---------------------|--------|
| exp1.tif  | 1      | 12       | 0.842                | -0.17  |



##### Integration

- **Input:** Output from `filter_data.py`
- **Output:** Feature table for use in classification or modeling (`motion_classifier.py`)

##### Design Highlights

- Pure and modular computation (`radius_of_gyration` is fully functional-style)
- Centralized issue tracking (`log_issue`, `validate_dataframe`)
- Compatible with both single-cell and population-level studies
- Scalable to future feature additions (via `feature_list`)

---
## `Motion_classifier.py` - Motion behavior Classification for the tracks
This module classifies single-molecule trajectories based on their motion patterns (e.g., bound vs. diffusive) using a Gaussian Mixture Model (GMM) applied to log-transformed features such as the radius of gyration. It has the option to use an already fitted gmm model instead of fitting a new model with the current data, if classification_config[`fit_new_gmm`] = false. 

- Fit a GMM to log-transformed track features (e.g., log_rg), or use an existing gmm model to keep the gmm parameters consistent accross different datasets. 
- Assign tracks to motion classes (e.g., class_0, class_1) with confidence levels
- Compute thresholds between GMM components
- Plot classification histograms and component distributions
- Optionally compute and visualize bound fraction per cell (for single-cell analyses)

##### Main Function: `classify_tracks(...)`

**Parameters**

| Name                    | Type           | Description                                                                                                                                                                            |
| ----------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `processed_data`        | `pd.DataFrame` | Must contain at least `log_rg` and optionally `cell_id` and `file_name` for single-cell analysis.                                                                                            |
| `classification_config` | `dict`         | GMM classification settings  |
| `plot_config`           | `dict`         | Plot settings                 |
| `output_config`         | `dict`         | Output file paths:`output_dir`: directory for saving files                                               |
| `single_cell_config`    | `dict`         | Single-cell summary options: `min_tracks_per_cell`: int`box_plot`: whether to plot bound fractions                                                                          |
| `random_state`          | `int`          | Random seed for reproducibility. Default is `42`.                                                                                                                                      |
| `single_cell_analysis`  | `bool`         | If `True`, calculates per-cell bound fractions. Default is `False`.                                                                                                                    |

**Returns**
- gmm: Fitted GaussianMixture object or the gmm that was loaded and used for the classification. 

- classified_df: DataFrame with assigned class, confidence, and status.

- summary_df: Summary DataFrame of class counts and percentages.

- cell_records: (optional) Per-cell bound fraction DataFrame (if single_cell_analysis=True)


#### Module Functions
- gmm_fit(data, n_components, random_state)
Fit a GMM to the input data if classification_config[`fit_new_gmm`] = true.

- compute_threshold(gmm_model)
Estimate the threshold (intersection point) between each adjacent GMM component.

- assign_classes(rg_df, gmm, feature, confidence_level)
Assigns tracks to classes using the GMM and applies a confidence filter.

- summarize_classification(classified_df)
Summarizes the number and percentage of tracks per class.

- plot_classification_results(data, gmm, plot_settings, summary_df, thresholds=None, save_path=None)
Visualizes the distribution of log(Rg) values, GMM components, and optional class thresholds.

- calculate_bound_fraction_per_cell(classified_df, ...)
Computes the per-cell fraction of tracks classified as "bound" (typically class_0), filtered by minimum number of tracks per cell.

- plot_bound_fraction_per_cell(cell_df, save_path=None)
Boxplot visualization of bound fractions across cells.


##### Example of configuration
```json
classification_config = {
    "n_components": 2,
    "confidence_level": 0.9,
    "feature_to_classify": "log_rg",
    "fit_new_gmm" : True,
    "gmm_model_path": "/path/to/an/existing/gmm_model.joblib"
}

plot_config = {
    "plot_results": True,
    "plot_thresholds": True,
    "plot_settings": {
        "title": "Track Classification",
        "xlabel": "log(Radius of Gyration)",
        "ylabel": "Density",
        "hist_color": "skyblue",
        "hist_label": "log(Rg)",
        "bins": 50,
        "title_fontsize": 14,
        "label_fontsize": 12
    }
}

output_config = {
    "output_dir": "results/",
    "classification_plot_name": "classification_histogram.png",
    "single_cell_plot_name": "bound_fraction_per_cell.png"
}

single_cell_config = {
    "min_tracks_per_cell": 20,
    "box_plot": True
}
```
##### **Notes**
- The component with the lowest mean is automatically labeled class_0, and is assumed to represent bound molecules.

- Tracks with a classification confidence below the threshold are marked "uncertain".
If you prefere to classify all the tracks, set the confidence threshold  in the config file to 0. 
- The GMM model is saved which allows you to reuse the trained model to classify new data without retraining. Potentially to keep the parameters of the model (mean, std, etc..) consistent. 

**Integration**
This module is intended to be used as part of a broader single-molecule analysis pipeline. It expects pre-processed data (e.g., filtered and log-transformed radius_of_gyration values).

**Output Files**
- classification_histogram.png — GMM histogram with class components.
- bound_fraction_per_cell.png (optional) — Boxplot of per-cell bound fractions.
- Structured DataFrames containing classification and per-cell stats, suitable for downstream analysis.

---
## `save_metadata.py `

This module provides functionality to save metadata for reproducibility in microscopy data processing pipelines. It stores the configuration used, Python environment, and input file list into a designated output directory.
**Parameters**
| Name      | Type   | Description                                                                   |
| --------- | ------ | ----------------------------------------------------------------------------- |
| `config`  | `dict` | The configuration dictionary used to run the pipeline.                        |
| `data`    | `dict` | Dictionary containing raw or processed input data. Keys should be file names. |
| `out_dir` | `str`  | Path to the root output directory where metadata will be saved.               |

**Saved Files**
| File Name              | Format     | Description                                                                   |
| ---------------------- | ---------- | ----------------------------------------------------------------------------- |
| `config_used.json`     | JSON       | The full configuration dictionary used for the pipeline run.                  |
| `environment_used.txt` | Plain Text | Output of `pip freeze`, capturing the Python environment for reproducibility. |
| `input_files.json`     | JSON       | List of input file names extracted from `data.keys()`.                        |

