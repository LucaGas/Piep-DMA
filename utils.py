# utils.py
import logging
from logging.handlers import RotatingFileHandler
import sys
import re
import pandas as pd

def setup_logging(log_level=logging.WARNING, log_file='app.log', max_bytes=5*1024*1024, backup_count=5):
    """
    Set up logging configuration for the entire application with log rotation and UTF-8 encoding.

    Parameters:
        log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file (str): File path for the log file.
        max_bytes (int): Maximum size of the log file in bytes before rotation.
        backup_count (int): Number of backup log files to keep.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Check if handlers are already added to avoid duplicate logs
    if not logger.handlers:
        # Create console handler with UTF-8 encoding
        try:
            c_handler = logging.StreamHandler(sys.stdout)
            c_handler.setLevel(log_level)
            c_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            c_handler.setStream(sys.stdout)
            c_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
            logger.addHandler(c_handler)
        except Exception as e:
            # Fallback if the above method fails
            c_handler = logging.StreamHandler(sys.stdout)
            c_handler.setLevel(log_level)
            c_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(c_handler)
            logger.error(f"Failed to set UTF-8 encoding for console handler: {e}")

        # Create rotating file handler with UTF-8 encoding
        try:
            f_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8')
            f_handler.setLevel(log_level)
            f_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(f_handler)
        except Exception as e:
            logger.error(f"Failed to set UTF-8 encoding for file handler: {e}")


def save_results(data, metadata_property_column, analysis_name, analysis_value):
    """
    Save the results of the analysis in the first empty cell of the 'Metadata Property' column.
    
    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - metadata_property_column (str): The name of the 'Metadata Property' column.
    - analysis_name (str): The name of the analysis (e.g., "Onset of E'").
    - analysis_value (float or str): The value of the analysis (e.g., the onset X value or a string description).
    
    Returns:
    - DataFrame: Updated DataFrame with the results saved.
    """
    logger = logging.getLogger(__name__)
    try:
        # Ensure the 'Metadata Property' column exists
        if metadata_property_column not in data.columns:
            logger.error(f"save_results: Column '{metadata_property_column}' not found in DataFrame.")
            return data

        # Find the first empty cell in the 'Metadata Property' column
        empty_row_idx = data[metadata_property_column].isnull().idxmax()
        if pd.notna(data.at[empty_row_idx, metadata_property_column]):
            # No empty rows found; append a new row at the end
            empty_row_idx = len(data)

        # Add the analysis name and value to the DataFrame
        data.at[empty_row_idx, metadata_property_column] = analysis_name
        # Assuming the 'Value' column is next to 'Metadata Property'
        value_column_idx = data.columns.get_loc(metadata_property_column) + 1
        if value_column_idx < len(data.columns):
            value_column = data.columns[value_column_idx]
            data.at[empty_row_idx, value_column] = analysis_value
        else:
            # If 'Value' column doesn't exist, add it
            data['Value'] = None
            data.at[empty_row_idx, 'Value'] = analysis_value

        logger.debug(f"save_results: Saved '{analysis_name}' with value '{analysis_value}' at row {empty_row_idx}.")
        return data

    except Exception as e:
        logger.error(f"save_results: Error occurred while saving results: {e}")
        return data


def get_aligned_data(data, x_column, y_column):
    """
    Align X and Y data by dropping rows where Y is NaN and resetting indices.

    Parameters:
    - data (DataFrame): The DataFrame containing X and Y columns.
    - x_column (str): The name of the X-axis column.
    - y_column (str): The name of the Y-axis column.

    Returns:
    - (Series, Series): Aligned and indexed X and Y data.
    """
    logger = logging.getLogger(__name__)
    aligned_data = data[[x_column, y_column]].dropna().reset_index(drop=True)
    aligned_x = pd.to_numeric(aligned_data[x_column], errors='coerce')
    aligned_y = pd.to_numeric(aligned_data[y_column], errors='coerce')

    logger.debug(f"Aligned X-axis data for '{y_column}':\n{aligned_x}")
    logger.debug(f"Aligned Y-axis data for '{y_column}':\n{aligned_y}")

    return aligned_x, aligned_y


def map_column_name(y_column):
    """
    Replace specific substrings in column names with desired labels based on a predefined mapping.

    Parameters:
    - y_column (str): Original column name.

    Returns:
    - str: Modified column name with replacements.
    """
    logger = logging.getLogger(__name__)
    # Define your mappings here
    replacements = {
        "E''": "Edoubleprime",  # Matches E'' (two single quotes)
        'E"': "Edoubleprime",   # Matches E" (single double quote)
        "E'": "Eprime",          # Matches E' (single single quote)
        "tan d": "tan_delta",    # Matches tan d
        "tanδ": "tan_delta",     # Matches tanδ
        # Add more mappings as needed
    }
        
    for original, replacement in replacements.items():
        # Use case-insensitive replacement
        y_column = re.sub(re.escape(original), replacement, y_column, flags=re.IGNORECASE)
    
    logger.debug(f"Mapped column name '{y_column}'.")
    return y_column
# utils.py

def sanitize_column_name(col_name):
    """
    Sanitize column names by replacing specific unwanted substrings with desired ones.
    
    Parameters:
        col_name (str): The original column name.
    
    Returns:
        str: The sanitized column name.
    """
    replacements = {
        'Temp./�C': 'Temp ºC',
        '"E""(1.000 Hz)/MPa"': "E'' (1.000 Hz) MPa",
        "E'(1.000 Hz)/MPa": "E' (1.000 Hz) MPa",
        # Add more replacements as needed
    }
    sanitized_name = replacements.get(col_name, col_name)
    if sanitized_name != col_name:
        logger = logging.getLogger(__name__)
        logger.debug(f"Sanitized column name from '{col_name}' to '{sanitized_name}'.")
    return sanitized_name
