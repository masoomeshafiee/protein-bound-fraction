import logging
import numpy as np
import pandas as pd
from src.data_validation import validate_dataframe, validate_mask, log_issue



# basic logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================
# Core Filtering Logic
# =====================

def filter_tracks_by_length(tracks, min_track_length):
    """
    Filters tracks based on the minimum track length.
    
    Parameters:
    - trcks: DataFrame containing the tracks data for a field of view.
    - min_track_length: Int Minimum length of the tracks to keep.

    Returns:
    - tracks_after_length_filter: DataFrame containing the filtered tracks data.
    """

    return tracks[tracks['track_duration'] >= min_track_length]


# write a function that filters the tracks based on the mask
def filter_tracks_by_mask(tracks, mask):
    """
    Filters tracks based on the mask.
    
    Parameters:
    - tracks: DataFrame containing the tracks data for a field of view.
    - mask: 2D numpy array containing the mask.
    Returns:
    - tracks_after_mask_filter: DataFrame containing the tracks that land in the cell mask.
    """
    # get the x and y coordinates of the tracks
    x_coords = tracks["x_loc"].round().astype(int)
    y_coords = tracks["y_loc"].round().astype(int)
    # filter out of bounds coordinates
    in_bounds = (x_coords >= 0) & (x_coords < mask.shape[1]) & (y_coords >= 0) & (y_coords < mask.shape[0])
    x_coords = x_coords[in_bounds]
    y_coords = y_coords[in_bounds]
    tracks_in_bounds = tracks[in_bounds].copy()
    # filter the tracks that land in the mask
    in_mask = mask[y_coords, x_coords] > 0
    filtered_tracks = tracks_in_bounds[in_mask]
    return filtered_tracks

def filter_spots(spots, filtered_tracks):
    """
    Filters spots based on the filtered tracks.
    
    Parameters:
    - spots: DataFrame containing the spots data for a field of view.
    - filtered_tracks: DataFrame containing the tracks filtered by length and mask.
    
    Returns:
    - filtered_spots: DataFrame containing the spots that belong to the filtered tracks.
    """

    return spots[spots['track_id'].isin(filtered_tracks['track_id'])].copy()


# ===========================
# Filtering Entry Points
# ===========================



def filter_population_level(file_name, file_data, min_track_length, issue_dict, columns_names):
    """
    Population level analysis
    Filters the tracks and spots for each file (only keeps the tracks that land in the cell mask for that file).
    Parameters:
    - file_name: Name of the file.
    - file_data: Dictionary containing the data for the file.
    - min_track_length: Minimum length of the tracks to keep.
    - issue_dict: Dictionary to store issues encountered during filtering.
    - columns_names: Dictionary containing the expected column names for spots and tracks.
    Raises:
    - ValueError: If the file data is missing required keys or if the filtered tracks or spots are empty.
    - KeyError: If the file data does not contain the required keys.
    - TypeError: If the file data is not a dictionary or if the tracks or spots are not DataFrames.
    Returns:
    - filtered_data: Dictionary containing the filtered data (spots, tracks and mask) for the file.
    """
    # Validate the mask and the spots and tracks DataFrames
    spots_columns = columns_names['spots']
    tracks_columns = columns_names['tracks']
    spots_valid, spots_msg = validate_dataframe(file_data.get('spots'), spots_columns)
    if not spots_valid:
        log_issue(issue_dict, file_name, message=spots_msg, level = 'error')
        return {}
    
    tracks_valid, tracks_msg = validate_dataframe(file_data.get('tracks'), tracks_columns)
    if not tracks_valid:
        log_issue(issue_dict, file_name, message=tracks_msg, level = 'error')
        return {}
    
    mask_valid, mask_msg = validate_mask(file_data.get('mask'))
    if not mask_valid:
        log_issue(issue_dict, file_name, message = mask_msg)
        return {}
    
    # Fileter tracks by length
    tracks = filter_tracks_by_length(file_data['tracks'], min_track_length)
    if tracks.empty:
        log_issue(issue_dict, file_name, message="No tracks found after filtering by length, skipping the file.", level='warning')
        return {}
    # Filter tracks by mask
    filtered_tracks = filter_tracks_by_mask(tracks, file_data['mask'])
    if filtered_tracks.empty:
        log_issue(issue_dict, file_name, message="No tracks found withing the cells in the mask, skipping the file.", level='warning')
        return {}
    # Filter the spots based on the filtered tracks
    filtered_spots = filter_spots(file_data['spots'], filtered_tracks)
    if filtered_spots.empty:
        log_issue(issue_dict, file_name, message="No corresponding spots found for the filtered tracks, skipping the file.", level='warning')
        return {}
    

    # add the filtered data to the filtered_data dictionary
    filtered_data = {
        'spots': filtered_spots,
        'tracks': filtered_tracks,
        'mask': file_data['mask']
    }
    logging.info(f"Filtered data for {file_name} with total number of tracks: {len(file_data['tracks'])} and {len(filtered_tracks)} number of tracks in the mask.")
    return filtered_data

def filter_single_cell_level(file_name, file_data, min_track_length, issue_dict, columns_names):
    """
    single cell level analysis
    Preprocesses the tracks and filters the tracks and spots for each cell in the file (only keeps the tracks that land in that cell).
    Parameters:
    - file_name: Name of the file.
    - file_data: Dictionary containing the data for the file.
    - min_track_length: Minimum length of the tracks to keep.
    - issue_dict: Dictionary to store issues encountered during filtering.
    - columns_names: Dictionary containing the expected column names for spots and tracks.
    Returns:
    - cell_data: Dictionary containing the filtered data for each cell ({'cell_ID' : {spots, tracks, mask}) for the file.
    """
    spots_columns = columns_names['spots']
    tracks_columns = columns_names['tracks']
    # Validate the mask and the spots and tracks DataFrames
    spots_valid, spots_msg = validate_dataframe(file_data.get('spots'), spots_columns)
    if not spots_valid:
        log_issue(issue_dict, file_name, message=spots_msg, level = 'error')
        return {}
    tracks_valid, tracks_msg = validate_dataframe(file_data.get('tracks'), tracks_columns)
    if not tracks_valid:
        log_issue(issue_dict, file_name, message=tracks_msg, level = 'error')
        return {}
    mask_valid, mask_msg = validate_mask(file_data.get('mask'))
    if not mask_valid:
        log_issue(issue_dict, file_name, message = mask_msg)
        return {}
    
    # Filter tracks by length
    tracks = filter_tracks_by_length(file_data['tracks'], min_track_length)
    if tracks.empty:
        log_issue(issue_dict, file_name, message="No tracks found after filtering by length, skipping the file.", level='warning')
        return {}
    

    # filtering the tracks for each cell
    cell_data = {}
    # get the unique cell IDs from the mask
    cell_ids = np.unique(file_data['mask'])
    # remove the background cell ID (0)
    cell_ids = cell_ids[cell_ids != 0]
    for cell_id in cell_ids:
        cell_mask = file_data['mask'] == cell_id
        # filter the tracks by mask to get the tracks that land in the cell mask
        filtered_tracks = filter_tracks_by_mask(tracks, cell_mask)
        # check if the filterd tracks are empty
        if filtered_tracks.empty:
            log_issue(issue_dict, file_name, cell_id=cell_id, message="No tracks found in cell ")
            continue
        # filtering the spots based on the filtered tracks
        filtered_spots = filter_spots(file_data['spots'], filtered_tracks)
        # check if the filtered spots are empty
        if filtered_spots.empty:
            log_issue(issue_dict, file_name, cell_id=cell_id, message="No corresponding spots found for the filtered tracks in cell.")
            continue
        # add the filtered data to the cell_data dictionary
        cell_data[cell_id] = {
            'spots': filtered_spots,
            'tracks': filtered_tracks,
            'mask': cell_mask
        }
        logging.info(f"{file_name} - cell {cell_id}: {len(filtered_tracks)} tracks retained.")
    # check if the cell_data dictionary is empty
    if not cell_data:
        log_issue(issue_dict, file_name, message="No cells found after filtering by mask and length, skipping the file.", level='warning')
    return cell_data


# ===============================
# Public API
# ===============================


def filter_tracks_and_spots(data, min_track_length, columns_names, analyze_single_cell=False):
    """
    Filters the tracks and spots data based on the mask and minimum track length.

    Parameters:
    - data: Dictionary containing the data for each file.
    - min_track_length: Minimum length of the tracks to keep.
    - columns_names: Dictionary containing the expected column names for spots and tracks.
    - analyze_single_cell: Boolean indicating if the analysis is for single cell level or population level. If True, the function will filter tracks and spots for each cell in the file.

    Returns:
    - filtered: Dictionary containing the filtered data. {'file_name': {'spots': DataFrame, 'tracks': DataFrame, 'mask': np.ndarray}} or {file_name: {cell_id: {'spots': DataFrame, 'tracks': DataFrame, 'mask': np.ndarray}}} if analyze_single_cell is True.
    - issue_dict: Dictionary containing the issues encountered during filtering. {'file_name': {'cell_id (file)': issue message}}
    """
    
    filtered = {}
    issue_dict = {}
    for file_name, file_data in data.items():
        # decide if the analysis is for a single cell or not
        if not analyze_single_cell:
            # population level analysis
            result = filter_population_level(file_name, file_data, min_track_length, issue_dict, columns_names)
        else:
            # single cell level analysis
            result = filter_single_cell_level(file_name, file_data, min_track_length, issue_dict, columns_names)
        
        if result:
            filtered[file_name] = result

    # check if the final data dictionary is empty
    if not filtered:
        logging.error("No data found after filtering. Please check the input data.")
        raise ValueError("No data found after filtering. Please check the input data.")
    logging.info('successfully filtered the data based on the masks provided.')
    return filtered, issue_dict

