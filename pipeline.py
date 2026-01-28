import os
import json
import logging
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import joblib

from src.load_data import load_data_from_config
from src.filter_data import filter_tracks_and_spots
from src.feature_extraction import extract_features
from src.motion_classifier import classify_tracks
from src.save_metadata import save_metadata
from src.quality_control import plot_track_length_distribution, plot_intensity_distribution

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    logging.info(f"Logging initialized. Output -> {log_file}")

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)
    
def main(config_path):

    config = load_config(config_path)

    setup_logging(config['output_config']['output_dir'])
    logging.info("Pipeline started.")
    try:
        
        # Step 1: Load raw data
        logging.info("Loading data...")
        data = load_data_from_config(config['path'], config['columns_names'])

        if config['quality_control']:
            logging.info("Performing quality control...")
            intermediate_dir = os.path.join(config['output_config']['output_dir'], "intermediate_results")
            os.makedirs(intermediate_dir, exist_ok=True)
            plot_track_length_distribution(data, intermediate_dir)
            plot_intensity_distribution(data, intermediate_dir)

        # Step 2: Filter tracks and spots
        logging.info("Filtering data...")
        filtered_data, issue_dict = filter_tracks_and_spots(data, config['filtering']['min_track_duration'], config['columns_names'], pixel_size_um=config['filtering']['pixel_size_um'], analyze_single_cell=config['analyze_single_cell'])

        # Step 3: Extract features
        logging.info("Extracting features...")
        columns_names = config['columns_names'].copy()
        # adding the x and y in microns columns to columns_names dictionary
        columns_names['x_loc_um'] = 'x_loc_um'
        columns_names['y_loc_um'] = 'y_loc_um'
        processed_data, issue_dict = extract_features(filtered_data, issue_dict, columns_names, config['feature_extraction']['feature_list'], config['analyze_single_cell'], MSD_params=config['feature_extraction'].get('MSD_params', None))
        if processed_data.empty:
            logging.error("No valid tracks found after feature extraction. Exiting pipeline.")
            return

        # Step 4: Classification
        logging.info("Running classification...")
        results = classify_tracks(processed_data, config['classification'], config['plot'], config['output_config'], config['single_cell'], random_state=42, single_cell_analysis=config['analyze_single_cell'])


        # Unpack results
        if config['analyze_single_cell']:
            gmm, classified_df, summary_df, cell_records = results
        else:
            gmm, classified_df, summary_df = results
            cell_records = None

        # Step 5: Save results
        logging.info("Saving results...")
        out_dir = config['output_config']['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        intermediate_dir = os.path.join(out_dir, "intermediate_results")
        os.makedirs(intermediate_dir, exist_ok=True)

        # Saving intermediate results
        processed_data.to_csv(os.path.join(intermediate_dir, "processed_data.csv"), index=False)

        rows = []
        for file_name, issues in issue_dict.items():
            for cell_id, message in issues.items():
                rows.append({
                    "file_name": file_name,
                    "cell_id": None if cell_id == "file" else cell_id,
                    "message": message
                })
        issues_df = pd.DataFrame(rows)
        issues_df.to_csv(os.path.join(intermediate_dir, "issues.csv"), index=False)

        # Saving final results
        gmm_path = os.path.join(out_dir, "gmm_model.joblib")
        joblib.dump(gmm, gmm_path)
        classified_df.to_csv(os.path.join(out_dir, "classified_tracks.csv"), index=False)
        summary_df.to_csv(os.path.join(out_dir, "classification_summary.csv"), index=False)
        if cell_records is not None:
            cell_records.to_csv(os.path.join(out_dir, "bound_fraction_per_cell.csv"), index=False)


        logging.info("Pipeline finished successfully.")
        logging.info("Saving metadata...")
        save_metadata(config, data, out_dir)

    except Exception as e:
        logging.error("Pipeline failed due to an error.", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run single-molecule classification pipeline.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    args = parser.parse_args()

    main(args.config)
