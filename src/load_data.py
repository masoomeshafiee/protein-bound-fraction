import os
import pandas as pd
import numpy as np
from skimage import io
import logging
from tqdm import tqdm



# configur logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def load_data_from_config(path, columns_names):
    """
    Load data from a configuration file.
    Args:
        - path (dict): contains the path settings (directory and suffix) for the spots and tracks files as well as masks.
        -  columns_names (list): list of column names "with order" to be used for the spots and tracks data.

        Returns:
        data (dict): Dictionary containing the loaded data for each file name:
        - spots (pd.DataFrame): DataFrame containing the spots data.
        - tracks (pd.DataFrame): DataFrame containing the tracks data.
        - masks (np.ndarray): 2D numpy array containing the mask.
    
    """

    # Handling common errors
    # Check if the path is a dictionary with the required keys
    try:
        input_dir = path['input_dir']
        masks_dir = path['masks_dir']
        spot_suffix = path.get('spot_suffix', '_spots.csv')
        track_suffix = path.get('track_suffix', '_tracks.csv')
        mask_suffix = path.get('mask_suffix', '_mask.png')


    except (KeyError, TypeError) as e:
        logging.error(f"Invalid path format: {e}")
        raise ValueError("Path must be a dictionary with 'input_dir', 'masks_dir', 'spot_suffix', 'track_suffix', 'mask_suffix' keys.")
    spots_columns = columns_names['spots']
    tracks_columns = columns_names['tracks']
    if not isinstance(spots_columns, list) or not isinstance(tracks_columns, list):
        logging.error("Invalid columns names format. Must be a list.")
        raise ValueError("Invalid columns names format. Must be a list.")
    # Check if the columns names are provided
    if not all(isinstance(col, str) for col in spots_columns) or not all(isinstance(col, str) for col in tracks_columns):
        logging.error("Invalid columns names format. Must be a list of strings.")
        raise ValueError("Invalid columns names format. Must be a list of strings.")

    data = {}

    # check if the input directory exists
    if not os.path.exists(input_dir):
        logging.error(f'Input directory does not exist: {input_dir}')
        raise FileNotFoundError(f'Input directory does not exist: {input_dir}')
    # check if the masks directory exists
    if not os.path.exists(masks_dir):
        logging.error(f'Masks directory does not exist: {masks_dir}')
        raise FileNotFoundError(f'Masks directory does not exist: {masks_dir}')
    
    # Check if the input directory is empty
    if not any(fname.endswith(spot_suffix) for fname in os.listdir(input_dir)) or not any(fname.endswith(track_suffix) for fname in os.listdir(input_dir)):
        logging.error(f'Input directory is empty or does not contain files with suffix {spot_suffix} or {track_suffix}')
        raise FileNotFoundError(f'Input directory is empty or does not contain files with suffix {spot_suffix} or {track_suffix}')
    
    # Check if the masks directory is empty
    if not any(fname.endswith(mask_suffix) for fname in os.listdir(masks_dir)):
        logging.error(f'Masks directory is empty or does not contain files with suffix {mask_suffix}')
        raise FileNotFoundError(f'Masks directory is empty or does not contain files with suffix {mask_suffix}')
    
    logging.info(f"Loading data from: {input_dir}")

    try:

        for file_name in tqdm(os.listdir(input_dir)):
            if file_name.endswith(spot_suffix):
                # get the base name of the file
                base_name = file_name[:-len(spot_suffix)]
                if not base_name in data:
                    data[base_name] = {}
                spot_file = os.path.join(input_dir, file_name)
                data[base_name]['spots'] = pd.read_csv(spot_file, header=None, names=spots_columns)
                logging.info(f"Loaded spots data from {spot_file}")
            elif file_name.endswith(track_suffix):
                # get the base name of the file
                base_name = file_name[:-len(track_suffix)]
                if not base_name in data:
                    data[base_name] = {}
                track_file = os.path.join(input_dir, file_name)
                data[base_name]['tracks'] = pd.read_csv(track_file, header=None, names=tracks_columns)
                logging.info(f"Loaded tracks data from {track_file}")
        
        # Check if each file name has both spots and tracks data
        for base_name in data.keys():
            if 'spots' not in data[base_name]:
                logging.error(f"Missing spots data for {base_name}")
                raise ValueError(f"Missing spots data for {base_name}")
            if 'tracks' not in data[base_name]:
                logging.error(f"Missing tracks data for {base_name}")
                raise ValueError(f"Missing tracks data for {base_name}")
        
        # loading the mask files
        for base_name in data.keys():
            mask_file = os.path.join(masks_dir, base_name + mask_suffix)
            if os.path.exists(mask_file):
                data[base_name]['mask'] = io.imread(mask_file)
                logging.info(f"Loaded mask data from {mask_file}")
            else:
                logging.error(f"Mask file does not exist: {mask_file}")
                raise FileNotFoundError(f"Mask file does not exist: {mask_file}")
            
        logging.info("All data loaded successfully.")

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
            
    return data

'''
path = {
    'input_dir': '/Users/masoomeshafiee/Desktop/test/input',
    'masks_dir': '/Users/masoomeshafiee/Desktop/test/masks',
    'spot_suffix': '_spots.csv',
    'track_suffix': '_tracks.csv',
    'mask_suffix': '_w2T2 GFP_cp_masks.png'
}
columns_names = {'spots':['track_id', 'frame_number', 'x_loc', 'y_loc', 'intensity'], 'tracks':['track_id', 'spt_tr', 'spt_widt', 'mean_sp', 'max_sp', 'min_sp',
'med_sp', 'std_sp', 'mean_q', 'max_q', 'min_q', 'med_q', 'std_q', 'track_duration', 'tr_start', 'tr_fin', 'x_loc', 'y_loc']}
data = load_data_from_config(path, columns_names)
#print(data['2-20240308_Rfa1_fast_549_60%_10ms_50nM_dye_CPT_40uM_3']['spots'].head())
# filter the data
filtered_data, issue_dict = filter_tracks_and_spots(data,4,columns_names, True)
feature_list = ['radius_of_gyration']
processed_data, issue_dict = extract_features(filtered_data, issue_dict, columns_names, feature_list, analyze_single_cell=True)
classification_config = {
    'n_components': 2,  # Number of GMM components
    'confidence_level': 0.6,  # Confidence level for classification
    'feature_to_classify': 'log_rg',  # Feature to classify
}
plot_config = {
    'plot_results': True,  # Whether to plot the results
    'plot_thresholds': True,  # whether to plot thresholds for classification
    'plot_settings':{
    'bins': 50,  # number of histogram bins
    'hist_color': 'gray',  # histogram bar color
    'hist_label': 'Data histogram',  # label for histogram in legend
    'title': 'GMM Classification of Radius of Gyration',
    'xlabel': 'log(Radius of Gyration)',
    'ylabel': 'Density',
    'title_fontsize': 14,
    'label_fontsize': 14
}

}
output_config = {
    'output_dir': '/Users/masoomeshafiee/Desktop/test/output',  # Directory to save the plot
    'single_cell_plot_name': 'single_cell_classification.png',  # Name of the single cell classification plot
    'classification_plot_name': 'classification_plot.png',  # Name of the classification plot
}

single_cell_config = {'box_plot': True,  # Whether to plot box plots for single cell data}
                      'min_tracks_per_cell': 100,  # Minimum number of tracks per cell to conside
                      
}

gmm, classified_df, summary_df, cell_records = classify_tracks(processed_data, classification_config, plot_config, output_config, single_cell_config, random_state=42, single_cell_analysis=True)
'''