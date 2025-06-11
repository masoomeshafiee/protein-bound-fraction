# Protein Bound Fraction Pipeline

This repository contains a modular Python pipeline to process microscopy tracking data and compute protein-bound fractions using Gaussian Mixture Model (GMM) classification.

## Features

- Quality control and Data validation
- Track-to-cell assignment from segmentation masks and Data filtering 
- Feature extraction (e.g., radius of gyration)
- GMM-based multi-class motion classification
- Single-cell level bound fraction analysis
- Plotting
- Modular configurable, and scalable
- Real-time logging to easily track errors and monitor pipeline progress
- Metadata saving for reproducibility

## Modules
- `load_data.py`: Load input tracking/spot/mask files
- `filter_data.py`: Clean and filter tracks
- `feature_extraction.py`: Compute motion features
- `motion_classifier.py`: Fit GMM and classify tracks
- `quality_control.py`, `data_validation.py`: Ensure data integrity
- `pipeline.py`: Main orchestrator script

### Inputs
1. Config File (.json)
Specifies paths, parameters, and module toggles ( check the `config.md` for details)

2. TrackMate spots files
One CSV per video. 
Typical columns include:
    | Column Name   | Description                              |
    |---------------|------------------------------------------|
    | TRACK_ID      | ID of the track     |
    | POSITION_T    | Time point (frame number).               |
    | POSITION_X    | X coordinate (in pixels).    |
    | POSITION_Y    | Y coordinate (in pixels).    |
    | INTENSITY     | Spot intensity             |

3. TrackMate trackfiles files
The mandatory coloumns include:
['track_id' , 'track_duration', 'x_loc', 'y_loc']

4. Segmentation Masks (.png)
Grayscale images with unique integer labels per cell (0 = background).
Recommended tool: [Cellpose](https://github.com/MouseLand/cellpose)
Use case: Enables per-cell analysis by mapping tracks to segmented regions.

##### Important Note:
The CSV files do not need to include column headers. The pipeline will assign column names automatically based on the columns_names specified in the config file.
Just ensure the column order in your CSV matches the expected structure.

### Outputs
All outputs are saved in the directory specified by `output_config["output_dir"]`. The pipeline generates the following files:

##### Intermediate Results (intermediate_results/)
- processed_data.csv: Filtered and validated data with extracted motion features.

- issues.csv: Summary of files or cells that encountered issues during processing, along with descriptive messages.

- spot_intensity_distribution.svg: Quality control plot showing the distribution of spot intensities across experiments with a fitted Gaussian curve.

- track_length_distribution.svg: Quality control plot of track length distribution across all samples with a fitted Gaussian curve.

##### Final Outputs
- classified_tracks.csv: Track-level classification results, including confidence scores and assigned motion class (e.g., bound vs. diffusive), per file and (if applicable) per cell.

- classification_summary.csv: Summary statistics per file, including total track count and estimated bound fraction.

- bound_fraction_per_cell.csv (if single-cell analysis is enabled): Bound fraction estimates for each individual cell, filtered by a configurable track count threshold.

- gmm_model.joblib: Serialized scikit-learn GMM model used for classification.

- Classification plots: Histograms of the classified feature (e.g., radius of gyration), with GMM components and decision thresholds visualized.

##### Metadata (metadata/)
- config_used.json: Exact configuration used during the run (copied from the input config for reproducibility).

- environment_used.txt: Snapshot of the package environment (e.g., conda list or pip freeze) to reproduce dependencies.

- input_files.json: Name of loaded input files.

##### Logging
pipeline_log_<timestamp>.log: Detailed log of the pipeline run, including progress updates, warnings, and errors (saved in the main output directory).


### Getting Started

**1. Download or clone the repository:**
``` bash
git clone https://github.com/yourusername/protein-bound-fraction-pipeline.git
cd protein-bound-fraction-pipeline
```
**2. Environment Set up:**

Create and activate a conda environment
``` bash

conda env create -f environment.yml
conda activate protein-bound-fraction
```
Alternatively with pip:
``` bash
pip install -r requirements.txt
```

**3. Run the pipeline:**
``` bash
python pipeline.py --config path/to/your_config.json
```
Example: 
``` bash
python pipeline.py --config ./config.json
```



### Usage

**1. Adjust the config file as desired**
Consult the `config_doc.md`. 

**2. Navigate into the protein-bound-fraction directory using your terminal.

**3. Activate your environment:**
conda activate protein-bound-fraction

**4. Run the file_name_handling.py (optional):**
To make sure that the spots and tracks files have the same base name as masks files.
``` bash
python file_name_handling.py --config path/to/your_config.json
``` 

**5. Run the pipeline:**
``` bash
python pipeline.py --config path/to/your_config.json
```
Example: 
``` bash
python pipeline.py --config ./config.json
```

## Project Structure
``` r
protein-bound-fraction-pipeline/
│
├── pipeline.py                   # Main entry point of the pipeline
├── file_name_handling.py        # Utility for matching track/spot/mask file names (optional)
├── README.md                    # Project overview and instructions
├── environment.yml              # Conda environment setup
├── requirements.txt
│
├── configs/                     # Configuration files
│   └── config.json       # Example JSON config file
│
├── examples/                    # Optional: example inputs and outputs
│
├── src/                         # Core source code modules
│   ├── __init__.py              # to behave as a python package recognized by the pipeline.
│   ├── load_data.py             # Load input tracking/spot/mask files
│   ├── filter_data.py           # Clean and filter tracks based on segmentation masks
│   ├── feature_extraction.py    # Compute motion features
│   ├── motion_classifier.py     # Fit GMM and classify tracks
│   ├── quality_control.py       # intensity/track length QC
│   ├── data_validation.py       # Validate data structure and completeness
│   ├── metadata_saver.py        # Save metadata about runs and settings
│
├── test/                        
│ 
├── docs/  
│   ├── config_doc.md   
│   ├── troubleshooting.md      
│   ├── modules.md       
│   ├── inputs/outputs.md 
│   ├── diagrams
│ 
```


## Example Usage

See the `examples/` directory for a working example with input files, config, and output results.
For common issues and solutions, check `docs/troubleshooting.md`.

## Configuration File
The pipeline is controlled via a JSON config file. See config.json for an example. It specifies input paths, filtering thresholds, features to extract, and output options.
See `docs/config_doc.md` for full documentation. 

## Contributors
- Masoumeh Shafiei

Contributions are welcome! Feel free to open issues or submit pull requests.

## Citation
If you use this tool for your research, please cite:
Reyes-Lamothe Lab, McGill University


