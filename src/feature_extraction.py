import pandas as pd
import numpy as np
import logging
from scipy.optimize import curve_fit
from src.data_validation import validate_dataframe, log_issue

SUPPORTED_FEATURES = ['radius_of_gyration', 'msd_features']
KEY_COLUMNS = ['file_name', 'cell_id', 'track_id'] # common columns to merge feature dataframes on
# ====================
# Helper Functions
# ====================
# =============================== Radius of Gyration approach ===============================
# calculate the Radius of Gyration for each track
def radius_of_gyration(spots_within_track, x='x_loc_um', y='y_loc_um'):
    """
    Calculates the radius of gyration for a track using the spots within the track following the formula:
    Rg = sqrt(1/N * sum((x_i - x_c)^2 + (y_i - y_c)^2))
    where N is the number of spots in the track, (x_i, y_i) are the coordinates of each spot, and (x_c, y_c) are the coordinates
    of the center of mass (mean position of the track).
    Parameters:
    spots_within_track: pd.DataFrame containing the spots within the track must have x and y location in microns.
    x: str, column name for x location in micrometers.
    y: str, column name for y location in micrometers.
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

# extract radius of gyration from a single data block (file or cell level)
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
    
# =============================== MSD approach ===============================
# calculate the Mean Squared Displacement (MSD) for each track
def calculate_MSD(spots_within_track, x="x_loc_um", y="y_loc_um"):
    """
    Calculates the Mean Squared Displacement (MSD) for a track using the spots within the track.
    Parameters:
    - spots_within_track: pd.DataFrame containing the spots within the track must have x and y location in micrometers.
    - x: str, column name for x location in micrometers.
    - y: str, column name for y location in micrometers.
    Returns:
    - msd: list of the average of squared displacements over all possible starting points for the same time lag based on the formula:
    MSD = <(x(t+dt) - x(t))^2 + (y(t+dt) - y(t))^2> for different time lags (dt)
    where <> denotes the average over all possible starting points t.
    The output is a list of MSD values for increasing time lags:
    msd = [MSD(1Δt), MSD(2Δt), MSD(3Δt), ...]
    """
    # Calculate displacements
    # track length:
    N = len(spots_within_track)
    msd = []
    for n in range(1, N):

        displacements = [(spots_within_track[x].iloc[i+n] - spots_within_track[x].iloc[i])**2 + 
                         (spots_within_track[y].iloc[i+n] - spots_within_track[y].iloc[i])**2 for i in range(N-n)]
        msd.append(np.mean(displacements))
    
    return np.array(msd, dtype=float)
# MSD fitting functions with localization error correction for each track
# anomalous diffusion fitting
def fit_MSD(msd, frame_interval, b, T_int, T_exp, max_time_lag=None):
    """
    MSD fitting function that accounts for localization error correction.
    MSD(t) = 4D*t^alpha + (4b^2 - 8*(1/6)*(T_int/T_exp)*D*T_int)
    T_int: integration time: the amount of time the camera sensor is actively collecting photons for one frame. It’s how long the shutter is “open” for each image.
    T_exp: exposure time
    b: localization error
    max_time_lag: int, optional, maximum time lag to consider for fitting.
    """
    def msd_func(t, D, alpha):
        return 4 * D * (t ** alpha) + (4 * b ** 2 - 8 * (1/6) * (T_int/T_exp) * D * T_int)
    if max_time_lag is not None:
        msd = msd[:max_time_lag]
    times = np.arange(1, len(msd) + 1) * frame_interval
    popt, _ = curve_fit(msd_func, times, msd, bounds=(0, [np.inf, 2]))
    D, alpha = popt
    return D, alpha  # returns [D, alpha]
# Brownian diffusion fitting (alpha = 1)
def fit_MSD_brownian(msd, frame_interval, b, T_int, T_exp, max_time_lag=None):
    """
    MSD fitting function assuming purely Brownian motion (alpha = 1).
    MSD(t) = 4D*t + (4b^2 - 8*(1/6)*(T_int/T_exp)*D*T_int)
    max_time_lag: int, optional, maximum time lag to consider for fitting.
    """
    def msd_func(t, D):
        return 4 * D * t + (4 * b ** 2 - 8 * (1/6) * (T_int/T_exp) * D * T_int)
    if max_time_lag is not None:
        msd = msd[:max_time_lag]
    times = np.arange(1, len(msd) + 1) * frame_interval
    popt, _ = curve_fit(msd_func, times, msd, bounds=(0, np.inf))
    D = float(popt[0])
    alpha = 1.0
    return D, alpha  # D and alpha fixed at 1
def extract_msd(file_name, data_block, issue_dict, columns_names, cell_id=None, use_brownian_only=False, b=0.0, frame_interval=0.01, T_int=0.009, T_exp=0.01, max_time_lag=None):
    """
    Extract MSD features from a single data block (file or cell level).
    Parameters:
    - file_name: str, the name of the file or cell.
    - data_block: dict, the data block containing the tracks, spots, mask data.
    - issue_dict: dict, dictionary to log issues with the data extraction.
    - columns_names: dict, dictionary containing the expected column names for spots and tracks.
    - cell_id: str, optional, the ID of the cell if analyzing single cells.
    - use_brownian_only: bool, whether to use only Brownian motion fitting (alpha = 1).
    - b: float, localization error in micrometers.
    - frame_interval: float, time interval between frames in seconds.
    - T_int: float, integration time in seconds.
    - T_exp: float, exposure time in seconds.
    Returns:
    - msd_df: pd.DataFrame containing the diffusion coefficient and anomalous exponent (alpha) for each track_id within the data block.
    """
    if 'spots' not in data_block:
        log_issue(issue_dict, file_name, cell_id, message="No spots data found in the data block.")
        return None
    
    spots_data = data_block['spots']
    valid_spots, message = validate_dataframe(spots_data, columns_names['spots'])
    if not valid_spots:
        log_issue(issue_dict, file_name, cell_id, message=f"invalid spot structure {message}")
        return None
    
    msd_features = []
    for track_id, track_spots in spots_data.groupby('track_id'):
        try: 
            
            time_col = "frame_number"
            track_spots = track_spots.sort_values(time_col).reset_index(drop=True)
            msd = calculate_MSD(track_spots, x='x_loc_um', y='y_loc_um') # list of MSD values for increasing time lags
            if len(msd) < 3 or not np.all(np.isfinite(msd)):
                log_issue(issue_dict, file_name, cell_id, message=f"Track ID {track_id} has insufficient or invalid MSD data for fitting.")
                continue
            if not use_brownian_only:
                D, alpha = fit_MSD(msd, frame_interval, b=b, T_int=T_int, T_exp=T_exp, max_time_lag=max_time_lag)
            else:
                D, alpha = fit_MSD_brownian(msd, frame_interval, b=b, T_int=T_int, T_exp=T_exp, max_time_lag=max_time_lag)
            if not np.isfinite(D) or D <= 0:
                log_issue(issue_dict, file_name, cell_id, message=f"Track ID {track_id} has invalid diffusion coefficient: {D}")
                continue
            msd_features.append({'track_id': track_id, 'diffusion_coefficient': float(D), 'anomalous_exponent': float(alpha), 'log_diffusion_coefficient': float(np.log10(D))})
        except Exception as e:
            log_issue(issue_dict, file_name, cell_id, message=f"MSD failed for track {track_id}: {e}" + (f", cell {cell_id}" if cell_id else ""), level="warning")
            continue

        
    msd_df = pd.DataFrame(msd_features)
    if msd_df.empty:
        log_issue(issue_dict, file_name, cell_id, message="No valid tracks found for MSD calculation.")
        return None
    msd_df['file_name'] = file_name
    if cell_id:
        msd_df['cell_id'] = cell_id
    columns = ['file_name', 'track_id', 'diffusion_coefficient', 'anomalous_exponent', 'log_diffusion_coefficient']
    if cell_id:
        columns.insert(1, 'cell_id')
    msd_df = msd_df[columns]
    logging.info(f"Extracted MSD features for {file_name}" + (f", cell {cell_id}" if cell_id else ""))
    return msd_df

    
# =============================== Merging feature dataframes ===============================
def _merge_feature_dataframes(feature_dfs, key_columns):
    """
    Outer merges multiple feature dataframes on the specified key columns.
    Parameters:
    - feature_dfs: list of pd.DataFrame, list of feature dataframes to merge (example: df of radius of gyration, df of MSD features).
    - key_columns: list of str, list of column names to merge on.
    Returns:
    - merged_df: pd.DataFrame, merged dataframe containing all features for the given data block.
    """
    out = None
    for df in feature_dfs:
        if df is None or df.empty:
            continue
        missing_keys = [key for key in key_columns if key not in df.columns]
        if missing_keys:
            raise ValueError(f"DataFrame is missing key columns: {missing_keys}")
            logging.error(f"DataFrame is missing key columns: {missing_keys}")

        # ensure 1 row per key in each feature df
        if df.duplicated(key_columns).any():
            dup = df[df.duplicated(key_columns, keep=False)][key_columns].head()
            log_issue({}, df['file_name'].iloc[0], df['cell_id'].iloc[0] if 'cell_id' in df.columns else None, message=f"Feature df has duplicate keys (should be 1 row per track). Example:\n{dup}", level='error')
            raise ValueError(f"Feature df has duplicate keys (should be 1 row per track). Example:\n{dup}")
        out = df if out is None else pd.merge(out, df, on=key_columns, how='outer')
    return out if out is not None else pd.DataFrame()
        
        
# ===============================
# Public API
# ===============================


def extract_features(filtered_data, issue_dict, columns_names, feature_list, analyze_single_cell=False, MSD_params=None):
    """
    Extracts features from the filtered data based on the specified features to extract. For now the radius of gyration and MAD are supported.

    Parameters:
    - filtered_data: Dict of DataFrames containing the filtered data for each filename (and each cell if analyze_single_cell is True):
        - spots (pd.DataFrame): DataFrame containing the spots data.
        - tracks (pd.DataFrame): DataFrame containing the tracks data.
        - masks (np.ndarray): 2D numpy array containing the mask.
    - issue_dict: Dict containing the filename and the cell_id (if analyze_single_cell is True) of the cells that have issues and the issues.
    - columns_names: Dict containing the expected column names for spots and tracks.
    - feature_list: List of features to extract. Currently only 'radius_of_gyration' and 'msd_features' are supported.
    - analyze_single_cell: Bool, if True, the features are calculated and stored for tracks in each cell separately for the single cell analysis.
    if False: the features are calculated and stored for all tracks in the image.
    Returns:
    - processed_data(pd.DataFrame): DataFrame containing the file name, cell_id (if applicable), and the extracted features.
    """
    # Check if the filtered_data is empty
    if not filtered_data or not isinstance(filtered_data, dict):
        logging.error("The filtered_data is empty or not a dictionary.")
        raise ValueError("The filtered_data is empty or not a dictionary.")
    
    # handle unsupported features
    unsupported = [f for f in feature_list if f not in SUPPORTED_FEATURES]
    if unsupported:
        logging.warning(f"Unsupported features requested and ignored: {unsupported}")

    processed_data = []
    
    for file_name, file_data in filtered_data.items():
        if not file_data or not isinstance(file_data, dict):
            log_issue(issue_dict, file_name, message="The file data is empty or not a dictionary. Skipping this file.", level='warning')
            continue
        if not analyze_single_cell:
            # population level analysis
            feature_dfs = []
            if 'radius_of_gyration' in feature_list:
                rg_df = extract_rg(file_name, file_data, issue_dict, columns_names)
                feature_dfs.append(rg_df)
                
            if 'msd_features' in feature_list:
                MSD_params = MSD_params or {}
                msd_df = extract_msd(file_name, file_data, issue_dict, columns_names, cell_id=None, use_brownian_only=MSD_params.get('use_brownian_only', False), b=MSD_params.get('b', 0.0), frame_interval=MSD_params.get('frame_interval', 0.01), T_int=MSD_params.get('T_int', 0.01), T_exp=MSD_params.get('T_exp', 0.01), max_time_lag=MSD_params.get('max_time_lag', None))
                feature_dfs.append(msd_df)
            # merge feature dataframes on file_name and track_id
            key_columns = KEY_COLUMNS.copy()  # remove cell_id for population level analysis
            key_columns.remove('cell_id')
            merged_df = _merge_feature_dataframes(feature_dfs, key_columns)
            if not merged_df.empty:
                processed_data.append(merged_df)
            else:
                log_issue(issue_dict, file_name, message="No features extracted for this file at population level.", level='warning')
                continue

        else:
            # Single cell analysis
            for cell_id, cell_data in file_data.items():
                # Check if the cell_data is an unempty dictionary
                if not cell_data or not isinstance(cell_data, dict):
                    log_issue(issue_dict, file_name, cell_id, message="The cell data is empty or not a dictionary. Skipping this cell.", level='warning')
                    continue
                feature_dfs = []
                if 'radius_of_gyration' in feature_list:
                # extract radius of gyration
                    rg_df = extract_rg(file_name, cell_data, issue_dict, columns_names, cell_id)
                    feature_dfs.append(rg_df)
                if 'msd_features' in feature_list:
                    MSD_params = MSD_params or {}
                    msd_df = extract_msd(file_name, cell_data, issue_dict, columns_names, cell_id=cell_id, use_brownian_only=MSD_params.get('use_brownian_only', False), b=MSD_params.get('b', 0.0), frame_interval=MSD_params.get('frame_interval', 0.01), T_int=MSD_params.get('T_int', 0.009), T_exp=MSD_params.get('T_exp', 0.01), max_time_lag=MSD_params.get('max_time_lag', None))
                    feature_dfs.append(msd_df)
                    
                # merge feature dataframes on file_name, cell_id and track_id
                merged_df = _merge_feature_dataframes(feature_dfs, KEY_COLUMNS)
                if not merged_df.empty:
                    processed_data.append(merged_df)
                else:
                    log_issue(issue_dict, file_name, cell_id, message="No features extracted for this cell.", level='warning')
                    continue
    if processed_data:
        processed_data = pd.concat(processed_data, ignore_index=True)
        return processed_data, issue_dict
    else:
        logging.error("No features extracted. The processed_data is empty.")
        return pd.DataFrame(), issue_dict
        


            
            
