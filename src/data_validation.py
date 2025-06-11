import logging
import numpy as np
import pandas as pd


# basic logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ====================
# Validation Utilities
# ====================

def validate_dataframe(df, expected_columns):
    """
    Validate if the DataFrame contains the expected columns.
    
    Args:
        df (pd.DataFrame): The DataFrame to validate.
        expected_columns (list): List of expected column names.
    
    Returns:
        bool: True if the DataFrame contains all expected columns, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a DataFrame."
    if df is None or df.empty:
        return False, "DataFrame is empty."
    if expected_columns and not set(expected_columns).issubset(df.columns):
        missing_cols = set(expected_columns) - set(df.columns)
        return False, f"Missing columns: {missing_cols}"
    return True, "DataFrame is valid."

def validate_mask(mask): 
    """
    Validate if the mask is a 2D numpy array.
    
    Args:
        mask (np.ndarray): The mask to validate.
    
    Returns:
        bool: True if the mask is a 2D numpy array, False otherwise.
    """
    if not isinstance(mask, np.ndarray):
        return False, "Mask is not a numpy array."
    if mask is None or mask.size == 0:
        return False, "Mask is empty."
    if mask.ndim != 2:
        return False, "Mask is not a 2D array."
    if np.sum(mask) == 0:
        return False, "Mask has non cells."
    return True, "Mask is valid."


# =====================
# Logging + Issue Track
# =====================

def log_issue(issue_dict, file_name, cell_id=None, message = "", level ='warning'):
    """
    Log an issue with a specific file and cell ID. The log both will be stored in the issue_dict for each file (and cell if applicable)
    and also logged to the console. 
    
    Args:
        issue_dict (dict): Dictionary to store issues.
        file_name (str): Name of the file where the issue occurred.
        cell_id (str, optional): Cell ID if applicable. Defaults to None.
        message (str, optional): Custom message for the issue. Defaults to "".
        level (str, optional): Logging level ( 'warning', 'error'). Defaults to 'warning'.
    
    Returns:
        None
    """
    context = f"{file_name}" if cell_id is None else f"{file_name} - Cell ID: {cell_id}"
    full_message = f"{context} : {message}"
    getattr(logging, level)(full_message)

    issue_dict.setdefault(file_name, {})
    issue_dict[file_name][cell_id or "file"] = message

