# utils.py

import pandas as pd
import re
import logging

# Configure logging
logging.basicConfig(level=logging.WARN, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def save_results(data, metadata_property_column, analysis_name, analysis_value, debug=False):
    """
    Save the results of the analysis in the first empty cell of the 'Metadata Property' column.
    
    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - metadata_property_column (str): The name of the 'Metadata Property' column.
    - analysis_name (str): The name of the analysis (e.g., "Onset of E'").
    - analysis_value (float or str): The value of the analysis (e.g., the onset X value or a string description).
    - debug (bool): Enable debug logging.
    
    Returns:
    - DataFrame: Updated DataFrame with the results saved.
    """
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

        if debug:
            logger.debug(f"save_results: Saved '{analysis_name}' with value '{analysis_value}' at row {empty_row_idx}.")
        return data

    except Exception as e:
        logger.error(f"save_results: Error occurred while saving results: {e}")
        return data


def get_aligned_data(data, x_column, y_column, debug=False):
    """
    Align X and Y data by dropping rows where Y is NaN and resetting indices.

    Parameters:
    - data (DataFrame): The DataFrame containing X and Y columns.
    - x_column (str): The name of the X-axis column.
    - y_column (str): The name of the Y-axis column.
    - debug (bool): Enable debug logging.

    Returns:
    - (Series, Series): Aligned and indexed X and Y data.
    """
    aligned_data = data[[x_column, y_column]].dropna().reset_index(drop=True)
    aligned_x = pd.to_numeric(aligned_data[x_column], errors='coerce')
    aligned_y = pd.to_numeric(aligned_data[y_column], errors='coerce')

    if debug:
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
    # Define your mappings here
    replacements = {
        "E''": "Edoubleprime",  # Matches E'' (two single quotes)
        "E\"": "Edoubleprime",  # Matches E" (single double quote)
        "E'": "Eprime",         # Matches E' (single single quote)
        "tan d": "tan_delta",   # Matches tan d
        "tanδ": "tan_delta",    # Matches tanδ
        # Add more mappings as needed
    }
        
    for original, replacement in replacements.items():
        # Use case-insensitive replacement
        y_column = re.sub(re.escape(original), replacement, y_column, flags=re.IGNORECASE)
    
    return y_column
