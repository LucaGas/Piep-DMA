# analyzer.py

import sys
import pandas as pd
import subprocess
from pathlib import Path
import re
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

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
            print(f"save_results: Column '{metadata_property_column}' not found in DataFrame.")
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
            print(f"save_results: Saved '{analysis_name}' with value '{analysis_value}' at row {empty_row_idx}.")
        return data

    except Exception as e:
        print(f"save_results: Error occurred while saving results: {e}")
        return data


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
            print(f"Find Onset: Smoothed Y-data:\n{y_smooth}")

        # Calculate the first derivative
        dy_dx = y_smooth.diff() / aligned_x.diff()

        if debug:
            print(f"Find Onset: First Derivative:\n{dy_dx}")

        # Define a negative threshold to detect significant decrease
        threshold_negative = dy_dx.mean() - threshold_multiplier * dy_dx.std()

        if debug:
            print(f"Find Onset: Threshold Negative: {threshold_negative}")

        # Detect where the derivative falls below the threshold
        onset_indices = dy_dx[dy_dx < threshold_negative].index

        if not onset_indices.empty:
            onset_index = onset_indices[0]
            onset_x = aligned_x.iloc[onset_index]
            if debug:
                print(f"Find Onset: Detected onset at index {onset_index}, X = {onset_x}")
            return onset_x
        else:
            if debug:
                print("Find Onset: No significant decrease detected.")
            return None
    except Exception as e:
        if debug:
            print(f"Find Onset: Error during onset detection: {e}")
        return None


def find_tip_with_scipy(aligned_x, aligned_y, debug=False, height=None, distance=50, prominence=0.001):
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
            print(f"Find Tip with SciPy: Detected peaks at indices {peaks}")
            print(f"Find Tip with SciPy: Peak properties {properties}")

        if len(peaks) == 0:
            if debug:
                print("Find Tip with SciPy: No peaks detected.")
            return None, None

        # Select the first peak
        first_peak = peaks[0]
        tip_x = aligned_x.iloc[first_peak]
        tip_y = aligned_y.iloc[first_peak]

        if debug:
            print(f"Find Tip with SciPy: Tip detected at index {first_peak}, X = {tip_x}, Y = {tip_y}")

        return tip_x, tip_y

    except Exception as e:
        if debug:
            print(f"Find Tip with SciPy: Error during tip detection: {e}")
        return None, None


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
        print(f"Aligned X-axis data for '{y_column}':\n{aligned_x}")
        print(f"Aligned Y-axis data for '{y_column}':\n{aligned_y}")

    return aligned_x, aligned_y


def analyze_temperature_sweep(data, debug=False):
    """
    Analyze data specific to 'Temperature Sweep' ASSAY.
    Perform onset analysis for E' and peak detection for E'' and tan δ.

    Parameters:
    - data (DataFrame): The loaded CSV data.
    - debug (bool): Enable debug logging.
    """
    if debug:
        print("Analyze Temperature Sweep: Received the following data:")
        print(data.head())

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
        print(f"Cleaned columns {cleaned_columns}")
    eprime_columns = [col for col in cleaned_columns if re.search(r"\bE'(?!')", col)]

    if debug:
        print(f"Found Eprime columns {eprime_columns}")
    for eprime_column in eprime_columns:
        analysis_name = "Onset of Eprime"
        if analysis_name in completed_analyses_list:
            if debug:
                print(f"Analyze Temperature Sweep: Skipping already completed analysis '{analysis_name}'.")
            continue

        # Align X and Y data
        aligned_x, aligned_y = get_aligned_data(data, x_column, eprime_column, debug=debug)

        if aligned_y.empty:
            if debug:
                print(f"Analyze Temperature Sweep: Aligned Y-data for '{eprime_column}' is empty. Skipping onset analysis.")
            continue

        # Perform onset analysis
        onset_x = find_onset(aligned_x, aligned_y, debug=debug)

        if onset_x is not None:
            print(f"Analyze Temperature Sweep: Onset detected for Eprime at X = {onset_x}")
            data = save_results(
                data=data,
                metadata_property_column="Metadata Property",
                analysis_name=analysis_name,
                analysis_value=onset_x,
                debug=debug
            )
            completed_analyses_list.append(analysis_name)

    # Analyze peaks for E'' (Edoubleprime)
    matching_columns_edoubleprime = [
        col for col in cleaned_columns if re.search(r"E''|E\"", col, re.IGNORECASE)
    ]

    if debug:
        print(f"Found Edoubleprime columns {matching_columns_edoubleprime}")
    for column in matching_columns_edoubleprime:
        analysis_name = "Peak of Edoubleprime"
        if analysis_name in completed_analyses_list:
            if debug:
                print(f"Analyze Temperature Sweep: Skipping already completed analysis '{analysis_name}'.")
            continue

        # Align X and Y data
        aligned_x, aligned_y = get_aligned_data(data, x_column, column, debug=debug)

        if aligned_y.empty:
            if debug:
                print(f"Analyze Temperature Sweep: Aligned Y-data for '{column}' is empty. Skipping peak analysis.")
            continue

        # Perform peak detection using SciPy
        peak_x, peak_y = find_tip_with_scipy(
            aligned_x,
            aligned_y,
            debug=debug,
            height=0.01,      # Adjust based on your data
            distance=5,      # Minimum number of samples between peaks
            prominence=10  # Minimum prominence of peaks
        )

        if peak_x is not None and peak_y is not None:
            print(f"Analyze Temperature Sweep: Peak detected for Edoubleprime at X = {peak_x}, Y = {peak_y}")
            data = save_results(
                data=data,
                metadata_property_column="Metadata Property",
                analysis_name=analysis_name,
                analysis_value=f"X: {peak_x}, Y: {peak_y}",
                debug=debug
            )
            completed_analyses_list.append(analysis_name)

    # Analyze peaks for tan δ (tand)
    matching_columns_tand = [col for col in cleaned_columns if re.search(r"tan ?δ|tan d", col, re.IGNORECASE)]
    for column in matching_columns_tand:
        analysis_name = "Peak of tand"
        if analysis_name in completed_analyses_list:
            if debug:
                print(f"Analyze Temperature Sweep: Skipping already completed analysis '{analysis_name}'.")
            continue

        # Align X and Y data
        aligned_x, aligned_y = get_aligned_data(data, x_column, column, debug=debug)

        if aligned_y.empty:
            if debug:
                print(f"Analyze Temperature Sweep: Aligned Y-data for '{column}' is empty. Skipping peak analysis.")
            continue

        # Perform peak detection using SciPy
        peak_x, peak_y = find_tip_with_scipy(
            aligned_x,
            aligned_y,
            debug=debug,
            height=0.01,      # Adjust based on your data
            distance=5,      # Minimum number of samples between peaks
            prominence=0.5  # Minimum prominence of peaks
        )

        if peak_x is not None and peak_y is not None:
            print(f"Analyze Temperature Sweep: Peak detected for tand at X = {peak_x}, Y = {peak_y}")
            data = save_results(
                data=data,
                metadata_property_column="Metadata Property",
                analysis_name=analysis_name,
                analysis_value=f"X: {peak_x}, Y: {peak_y}",
                debug=debug
            )
            completed_analyses_list.append(analysis_name)

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
    return data


def analyze(csv_path, debug=False):
    """
    Analysis function that extracts and outputs the value of 'ASSAY' from metadata.
    If the ASSAY is 'Temperature Sweep', passes the data to analyze_temperature_sweep.
    """
    try:
        # Load the CSV file with detected encoding
        data = pd.read_csv(csv_path, encoding='utf-8')
        if debug:
            print(f"Analyzer: Received data from '{csv_path}' with {len(data)} records.")
            print("Analyzer: Displaying first few rows of the data:")
            print(data.head())

        # Extract 'ASSAY' value
        assay_value = None
        if "Metadata Property" in data.columns:
            # Find the row where "Metadata Property" equals "ASSAY"
            assay_row = data[data["Metadata Property"] == "ASSAY"]
            if not assay_row.empty:
                # Identify the column immediately to the right of "Metadata Property"
                metadata_property_index = data.columns.get_loc("Metadata Property")
                value_column_index = metadata_property_index + 1
                if value_column_index < len(data.columns):
                    assay_value = assay_row.iloc[0, value_column_index]

        if assay_value:
            print(f"Analyzer: Global metadata ASSAY = '{assay_value}'")
        else:
            print("Analyzer: Metadata 'ASSAY' not found in the CSV.")

        # If ASSAY is 'Temperature Sweep', call the analyze_temperature_sweep function
        if assay_value == "Temperature Sweep":
            print("Analyzer: Detected 'Temperature Sweep'. Passing data to analyze_temperature_sweep.")
            data = analyze_temperature_sweep(data, debug=debug)

            # Save the updated DataFrame to a new CSV file
            output_file = csv_path.replace(".csv", "_analyzed.csv")
            data.to_csv(output_file, index=False)
            print(f"Analyzer: Results saved to '{output_file}'.")

    except FileNotFoundError:
        print(f"Analyzer Error: The file '{csv_path}' does not exist.")
    except pd.errors.EmptyDataError:
        print(f"Analyzer Error: The file '{csv_path}' is empty.")
    except Exception as e:
        print(f"Analyzer Error: An unexpected error occurred - {e}")


def main():
    """
    Analyzer that passes the analyzed CSV file to plotter.py for plotting.
    """
    if len(sys.argv) < 2:
        print("Analyzer: No CSV file provided.")
        sys.exit(1)

    csv_file = sys.argv[1]
    print(f"Analyzer: Received '{csv_file}' for analysis.")
    debug=True
    # Perform analysis and get the path to the analyzed CSV
    analyzed_csv = csv_file.replace(".csv", "_analyzed.csv")
    analyze(csv_file, debug=debug)  # Pass debug flag

    try:
        # Call plotter.py as a standalone script with the analyzed CSV file
        print(f"Analyzer: Calling plotter.py for {analyzed_csv}")
        result = subprocess.run(
            [sys.executable, "plotter.py", analyzed_csv],
            check=True,
            capture_output=True,  # Capture stdout and stderr for logging
            text=True  # Decode stdout and stderr as text
        )
        print(f"Analyzer: Plotter completed for {analyzed_csv}")
        print(f"Plotter Output:\n{result.stdout}")
        if result.stderr:
            print(f"Plotter Warnings/Errors:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Analyzer: Plotter failed for {analyzed_csv} with error: {e}")
        print(f"Plotter Error Output:\n{e.stderr}")
    except Exception as ex:
        print(f"Analyzer: Unexpected error while calling plotter.py: {ex}")


if __name__ == "__main__":
    main()
