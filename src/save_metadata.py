import json
import os
import subprocess
import logging

def save_metadata(config, data, out_dir):
    """
    Save metadata for reproducibility:
    - Config file
    - Python environment (pip freeze)
    - List of input files

    Parameters:
    - config: The pipeline configuration dictionary.
    - data: The raw or processed DataFrame (must include 'file_name' column).
    - out_dir: Path to the main output directory.
    """
    meta_data_dir = os.path.join(out_dir, "metadata")
    os.makedirs(meta_data_dir, exist_ok=True)

    # 1. Save config used
    config_path = os.path.join(meta_data_dir, "config_used.json")
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logging.info(f"Saved config to: {config_path}")
    except Exception as e:
        logging.warning(f"Failed to save config: {e}")

    # 2. Save environment (pip freeze)
    env_path = os.path.join(meta_data_dir, "environment_used.txt")
    try:
        with open(env_path, "w") as f:
            subprocess.run(["pip", "freeze"], stdout=f, check=True)
        logging.info(f"Saved environment info to: {env_path}")
    except Exception as e:
        logging.warning(f"Failed to save pip environment: {e}")

    # 3. Save input file names
    input_file_list = list(data.keys())
    input_file_path = os.path.join(meta_data_dir, "input_files.json")
    try:
        with open(input_file_path, 'w') as f:
            json.dump(input_file_list, f, indent=4)
        logging.info(f"Saved input file list to: {input_file_path}")
    except Exception as e:
        logging.warning(f"Failed to save input file list: {e}")
