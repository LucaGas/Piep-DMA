# assays/analyze_temperature_sweep.py

import sys
import os

# Add parent directory to sys.path to access utils.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
import re
import logging
from utils import save_results, get_aligned_data, map_column_name
from scipy.signal import find_peaks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def find_onset(aligned_x, aligned_y, debug=False, threshold_multiplier=2, window_size=5):
    """
    Detect the onset point where y_data starts to decrease significantly.

    Parameters:
    - aligned_x (Series or array-like): The aligned X-axis data (e.g., temperature).
    - aligned_y (Series or array-like): The aligned Y-axis data (e.g., Eprime values).
    - debug (bool): Enable debug logging.
    - threshold_multiplier (float): Multiplier for standard deviation in threshold calculation.
    - window_size (int): Window size for moving average smoothing.

    Returns:
    - float: The X-value corresponding to the onset, or None if not found.
    """
    try:
        # Apply moving average to smooth the data
        y_smooth = aligned_y.rolling(window=window_size, center=True, min_periods=1).mean()

        if debug:
            logger.debug(f"Find Onset: Smoothed Y-data:\n{y_smooth}")

        # Calculate the first derivative
        dy_dx = y_smooth.diff() / aligned_x.diff()

        if debug:
            logger.debug(f"Find Onset: First Derivative:\n{dy_dx}")

        # Define a negative threshold to detect significant decrease
        threshold_negative = dy_dx.mean() - threshold_multiplier * dy_dx.std()

        if debug:
            logger.debug(f"Find Onset: Threshold Negative: {threshold_negative}")

        # Detect where the derivative falls below the threshold
        onset_indices = dy_dx[dy_dx < threshold_negative].index

        if not onset_indices.empty:
            onset_index = onset_indices[0]
            onset_x = aligned_x.iloc[onset_index]
            if debug:
                logger.debug(f"Find Onset: Detected onset at index {onset_index}, X = {onset_x}")
            return onset_x
        else:
            if debug:
                logger.debug("Find Onset: No significant decrease detected.")
            return None
    except Exception as e:
        if debug:
            logger.error(f"Find Onset: Error during onset detection: {e}")
        return None


def find_tip_with_scipy(aligned_x, aligned_y, debug=False, height=None, distance=5, prominence=10):
    """
    Detect the first tip of the data using SciPy's find_peaks function.

    Parameters:
    - aligned_x (Series or array-like): The aligned X-axis data.
    - aligned_y (Series or array-like): The aligned Y-axis data.
    - debug (bool): Enable debug logging.
    - height (float or None): Minimum height of peaks.
    - distance (int): Minimum number of samples between peaks.
    - prominence (float): Minimum prominence of peaks.

    Returns:
    - (float, float): The X and Y value at the tip, or (None, None) if no tip is found.
    """
    try:
        # Find peaks using SciPy
        peaks, properties = find_peaks(aligned_y, height=height, distance=distance, prominence=prominence)

        if debug:
            logger.debug(f"Find Tip with SciPy: Detected peaks at indices {peaks}")
            logger.debug(f"Find Tip with SciPy: Peak properties {properties}")

        if len(peaks) == 0:
            if debug:
                logger.debug("Find Tip with SciPy: No peaks detected.")
            return None, None

        # Select the first peak
        first_peak = peaks[0]
        tip_x = aligned_x.iloc[first_peak]
        tip_y = aligned_y.iloc[first_peak]

        if debug:
            logger.debug(f"Find Tip with SciPy: Tip detected at index {first_peak}, X = {tip_x}, Y = {tip_y}")

        return tip_x, tip_y

    except Exception as e:
        if debug:
            logger.error(f"Find Tip with SciPy: Error during tip detection: {e}")
        return None, None


def analyze_temperature_sweep(data, debug=False):
    """
    Analyze data specific to 'Temperature Sweep' ASSAY.
    Perform onset analysis for E' and peak detection for E'' and tan δ.

    Parameters:
    - data (DataFrame): The loaded CSV data.
    - debug (bool): Enable debug logging.

    Returns:
    - DataFrame: Updated DataFrame with analysis results.
    """
    if debug:
        logger.debug("Analyze Temperature Sweep: Received the following data:")
        logger.debug(data.head())

    # Clean column names (trim whitespace and standardize encoding)
    cleaned_columns = [col.strip() for col in data.columns]
    data.columns = cleaned_columns  # Update DataFrame with cleaned column names

    # Identify the X-axis column (first column)
    x_column = data.columns[0]

    # Track completed analyses
    if "Metadata Property" in data.columns:
        completed_analyses = data[data["Metadata Property"] == "Completed Analyses"]
        if not completed_analyses.empty:
            completed_analyses_list = completed_analyses.iloc[0, 1]
            if isinstance(completed_analyses_list, str):
                completed_analyses_list = [item.strip() for item in completed_analyses_list.split(",")]
            else:
                completed_analyses_list = []
        else:
            completed_analyses_list = []
    else:
        completed_analyses_list = []

    # Analyze onset for E' columns
    if debug:
        logger.debug(f"Cleaned columns {cleaned_columns}")
    eprime_columns = [col for col in cleaned_columns if re.search(r"\bE'(?!')", col)]

    if debug:
        logger.debug(f"Found Eprime columns {eprime_columns}")
    for eprime_column in eprime_columns:
        analysis_name = "Onset of Eprime"
        if analysis_name in completed_analyses_list:
            if debug:
                logger.debug(f"Analyze Temperature Sweep: Skipping already completed analysis '{analysis_name}'.")
            continue

        # Align X and Y data
        aligned_x, aligned_y = get_aligned_data(data, x_column, eprime_column, debug=debug)

        if aligned_y.empty:
            if debug:
                logger.debug(f"Analyze Temperature Sweep: Aligned Y-data for '{eprime_column}' is empty. Skipping onset analysis.")
            continue

        # Perform onset analysis
        onset_x = find_onset(aligned_x, aligned_y, debug=debug)

        if onset_x is not None:
            print(f"Analyze Temperature Sweep: Onset detected for Eprime at X = {onset_x}")
            logger.info(f"Analyze Temperature Sweep: Onset detected for Eprime at X = {onset_x}")
            data = save_results(
                data=data,
                metadata_property_column="Metadata Property",
                analysis_name=analysis_name,
                analysis_value=onset_x,
                debug=debug
            )
            completed_analyses_list.append(analysis_name)
        else:
            if debug:
                logger.debug(f"Analyze Temperature Sweep: No onset detected for '{eprime_column}'.")

    # Analyze peaks for E'' (Edoubleprime)
    matching_columns_edoubleprime = [
        col for col in cleaned_columns if re.search(r"E''|E\"", col, re.IGNORECASE)
    ]

    if debug:
        logger.debug(f"Found Edoubleprime columns {matching_columns_edoubleprime}")
    for column in matching_columns_edoubleprime:
        analysis_name = "Peak of Edoubleprime"
        if analysis_name in completed_analyses_list:
            if debug:
                logger.debug(f"Analyze Temperature Sweep: Skipping already completed analysis '{analysis_name}'.")
            continue

        # Align X and Y data
        aligned_x, aligned_y = get_aligned_data(data, x_column, column, debug=debug)

        if aligned_y.empty:
            if debug:
                logger.debug(f"Analyze Temperature Sweep: Aligned Y-data for '{column}' is empty. Skipping peak analysis.")
            continue

        # Perform peak detection using SciPy
        peak_x, peak_y = find_tip_with_scipy(
            aligned_x,
            aligned_y,
            debug=debug,
            height=0.01,      # Adjust based on your data
            distance=5,       # Minimum number of samples between peaks
            prominence=10     # Minimum prominence of peaks
        )

        if peak_x is not None and peak_y is not None:
            print(f"Analyze Temperature Sweep: Peak detected for Edoubleprime at X = {peak_x}, Y = {peak_y}")
            logger.info(f"Analyze Temperature Sweep: Peak detected for Edoubleprime at X = {peak_x}, Y = {peak_y}")
            data = save_results(
                data=data,
                metadata_property_column="Metadata Property",
                analysis_name=analysis_name,
                analysis_value=f"X: {peak_x}, Y: {peak_y}",
                debug=debug
            )
            completed_analyses_list.append(analysis_name)
        else:
            if debug:
                logger.debug(f"Analyze Temperature Sweep: No peak detected for '{column}'.")

    # Analyze peaks for tan δ (tand)
    matching_columns_tand = [col for col in cleaned_columns if re.search(r"tan ?δ|tan d", col, re.IGNORECASE)]
    for column in matching_columns_tand:
        analysis_name = "Peak of tand"
        if analysis_name in completed_analyses_list:
            if debug:
                logger.debug(f"Analyze Temperature Sweep: Skipping already completed analysis '{analysis_name}'.")
            continue

        # Align X and Y data
        aligned_x, aligned_y = get_aligned_data(data, x_column, column, debug=debug)

        if aligned_y.empty:
            if debug:
                logger.debug(f"Analyze Temperature Sweep: Aligned Y-data for '{column}' is empty. Skipping peak analysis.")
            continue

        # Perform peak detection using SciPy
        peak_x, peak_y = find_tip_with_scipy(
            aligned_x,
            aligned_y,
            debug=debug,
            height=0.01,      # Adjust based on your data
            distance=5,       # Minimum number of samples between peaks
            prominence=0.5    # Minimum prominence of peaks
        )

        if peak_x is not None and peak_y is not None:
            print(f"Analyze Temperature Sweep: Peak detected for tand at X = {peak_x}, Y = {peak_y}")
            logger.info(f"Analyze Temperature Sweep: Peak detected for tand at X = {peak_x}, Y = {peak_y}")
            data = save_results(
                data=data,
                metadata_property_column="Metadata Property",
                analysis_name=analysis_name,
                analysis_value=f"X: {peak_x}, Y: {peak_y}",
                debug=debug
            )
            completed_analyses_list.append(analysis_name)
        else:
            if debug:
                logger.debug(f"Analyze Temperature Sweep: No peak detected for '{column}'.")

    # Update the "Completed Analyses" metadata
    if "Completed Analyses" in data["Metadata Property"].values:
        data.loc[data["Metadata Property"] == "Completed Analyses", "Value"] = ", ".join(completed_analyses_list)
    else:
        data = save_results(
            data=data,
            metadata_property_column="Metadata Property",
            analysis_name="Completed Analyses",
            analysis_value=", ".join(completed_analyses_list),
            debug=debug
        )

    print("Analyze Temperature Sweep: Analysis complete.")
    logger.info("Analyze Temperature Sweep: Analysis complete.")
    return data


def main():
    """
    Entry point for the Temperature Sweep analysis script.
    """
    if len(sys.argv) < 2:
        print("Analyze Temperature Sweep: No CSV file provided.")
        print("Usage: python analyze_temperature_sweep.py <path_to_csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    print(f"Analyze Temperature Sweep: Received '{csv_file}' for analysis.")
    debug = True  # Set to False to reduce verbosity

    try:
        # Load the CSV file
        data = pd.read_csv(csv_file, encoding='utf-8')
        if debug:
            logger.debug(f"Analyze Temperature Sweep: Loaded data from '{csv_file}' with {len(data)} records.")
            logger.debug("Analyze Temperature Sweep: Displaying first few rows of the data:")
            logger.debug(data.head())

        # Perform Temperature Sweep analysis
        analyzed_data = analyze_temperature_sweep(data, debug=debug)

        # Save the analyzed DataFrame to a new CSV file
        output_file = csv_file.replace(".csv", "_analyzed.csv")
        analyzed_data.to_csv(output_file, index=False)
        print(f"Analyze Temperature Sweep: Results saved to '{output_file}'.")
        logger.info(f"Analyze Temperature Sweep: Results saved to '{output_file}'.")

    except FileNotFoundError:
        print(f"Analyze Temperature Sweep Error: The file '{csv_file}' does not exist.")
        logger.error(f"Analyze Temperature Sweep Error: The file '{csv_file}' does not exist.")
    except pd.errors.EmptyDataError:
        print(f"Analyze Temperature Sweep Error: The file '{csv_file}' is empty.")
        logger.error(f"Analyze Temperature Sweep Error: The file '{csv_file}' is empty.")
    except Exception as e:
        print(f"Analyze Temperature Sweep Error: An unexpected error occurred - {e}")
        logger.error(f"Analyze Temperature Sweep Error: An unexpected error occurred - {e}")


if __name__ == "__main__":
    main()
