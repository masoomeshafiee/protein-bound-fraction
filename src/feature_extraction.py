import pandas as pd
import numpy as np
import logging
from src.data_validation import validate_dataframe, log_issue

# ====================
# Helper Functions
# ====================

def radius_of_gyration(spots_within_track, x='x_loc', y='y_loc'):
    """
    Calculates the radius of gyration for a track using the spots within the track following the formula:
    Rg = sqrt(1/N * sum((x_i - x_c)^2 + (y_i - y_c)^2))
    where N is the number of spots in the track, (x_i, y_i) are the coordinates of each spot, and (x_c, y_c) are the coordinates
    of the center of mass (mean position of the track).
    Parameters:
    spots_within_track: pd.DataFrame containing the spots within the track must have x and y location.
    Note: it is assumed that x_loc and y_loc untis is in real-world units (e.g., micrometers).
    Returns: 
    radius_of_gyration: float, the radius of gyration of the track.
    
    """
    # Calculate the center of mass
    x_c = spots_within_track[x].mean()
    y_c = spots_within_track[y].mean()
    # Calculate the distance from the center of mass
    distances = ((spots_within_track[x] - x_c) ** 2 + (spots_within_track[y] - y_c) **2)
    radius_of_gyration = np.sqrt(distances.mean())
    return radius_of_gyration


def extract_rg(file_name, data_block, issue_dict, columns_names, cell_id = None):
    """
    Extract radius of gyration from a single data block (file or cell level).
    Parameters:
    - file_name: str, the name of the file or cell.
    - data_block: dict, the data block containing the tracks, spots, mask data.
    - issue_dict: dict, dictionary to log issues with the data extraction.
    - columns_names: dict, dictionary containing the expected column names for spots and tracks.
    - cell_id: str, optional, the ID of the cell if analyzing single cells.
    Returns:
    - rg_df: pd.DataFrame containing the radius of gyration and log transformed radius of gyration for each track_id within the data block.
    """
    if 'spots' not in data_block:
        log_issue(issue_dict, file_name, cell_id, message="No spots data found in the data block.")
        return None
    
    spots_data = data_block['spots']
    valid_spots, message = validate_dataframe(spots_data, columns_names['spots'])
    if not valid_spots:
        log_issue(issue_dict, file_name, cell_id, message=f"invalid spot structure {message}")
        return None
    try:
        rg_series = spots_data.groupby('track_id').apply(radius_of_gyration)
        rg_df = rg_series.reset_index(name="radius_of_gyration")
        if rg_df.empty:
            log_issue(issue_dict, file_name, cell_id, message="No valid tracks found for radius of gyration calculation.")
            return None
        # adding the log transformed radius of gyration for numerical stability and better fit
        rg_df['log_rg'] = np.log(rg_df['radius_of_gyration'])
        rg_df['file_name'] = file_name
        if cell_id:
            rg_df['cell_id'] = cell_id
        logging.info(f"Extracted radius of gyration for {file_name}" + (f", cell {cell_id}" if cell_id else ""))
        columns = ['file_name', 'track_id', 'radius_of_gyration', 'log_rg']
        if cell_id:
            columns.insert(1, 'cell_id')
        rg_df = rg_df[columns]
        return rg_df
    except Exception as e:
        log_issue(issue_dict, file_name, cell_id, message=f"Error calculating radius of gyration: {e}", level='error')
        return None
    
# ===============================
# Public API
# ===============================


def extract_features(filtered_data, issue_dict, columns_names, feature_list, analyze_single_cell=False):
    """
    Extracts features from the filtered data based on the specified features to extract. For now the radius of gyration is the only feature extracted.
    Parameters:
    - filtered_data: Dict of DataFrames containing the filtered data for each filename (and each cell if analyze_single_cell is True):
        - spots (pd.DataFrame): DataFrame containing the spots data.
        - tracks (pd.DataFrame): DataFrame containing the tracks data.
        - masks (np.ndarray): 2D numpy array containing the mask.
    - issue_dict: Dict containing the filename and the cell_id (if analyze_single_cell is True) of the cells that have issues and the issues.
    - columns_names: Dict containing the expected column names for spots and tracks.
    - feature_list: List of features to extract. Currently only 'radius_of_gyration' is supported.
    - analyze_single_cell: Bool, if Ture, the radius of gyration is calculated and stored for tracks in each cell separately for the single cell analysis.
    if False: the radius of gyration is calculated and stored for all tracks in the image.
    Returns:
    - processed_data(pd.DataFrame): DataFrame containing the file name, cell_id (if applicable), and the extracted features.
    """
    # Check if the filtered_data is empty
    if not filtered_data or not isinstance(filtered_data, dict):
        logging.error("The filtered_data is empty or not a dictionary.")
        raise ValueError("The filtered_data is empty or not a dictionary.")
    
    # handle unsupported features
    unsupported = [f for f in feature_list if f != 'radius_of_gyration']
    if unsupported:
        logging.warning(f"Unsupported features requested and ignored: {unsupported}")

    processed_data = []
    for file_name, file_data in filtered_data.items():
        if not file_data or not isinstance(file_data, dict):
            log_issue(issue_dict, file_name, message="The file data is empty or not a dictionary. Skipping this file.", level='warning')
            continue
        if not analyze_single_cell:
            # population level analysis
            if 'radius_of_gyration' in feature_list:
                rg_df = extract_rg(file_name, file_data, issue_dict, columns_names)
                if rg_df is None:
                    continue
                processed_data.append(rg_df)
        else:
            # Single cell analysis
            if 'radius_of_gyration' in feature_list:
                for cell_id, cell_data in file_data.items():
                    # Check if the cell_data is an unempty dictionary
                    if not cell_data or not isinstance(cell_data, dict):
                        log_issue(issue_dict, file_name, cell_id, message="The cell data is empty or not a dictionary. Skipping this cell.", level='warning')
                        continue
                    # extract radius of gyration
                    rg_df = extract_rg(file_name, cell_data, issue_dict, columns_names, cell_id)
                    if rg_df is None:
                        continue
                    processed_data.append(rg_df)
    if processed_data:
        return pd.concat(processed_data, ignore_index=True), issue_dict
    else:
        logging.error("No features extracted. The processed_data is empty.")
        return pd.DataFrame(), issue_dict
        


            
            
