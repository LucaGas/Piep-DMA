# analyzer.py

import sys
import pandas as pd
import subprocess
from pathlib import Path
from pprint import pprint
import re

def save_results(data, metadata_property_column, analysis_name, analysis_value, debug=False):
    """
    Save the results of the analysis in the first empty cell of the 'Metadata Property' column.
    
    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - metadata_property_column (str): The name of the 'Metadata Property' column.
    - analysis_name (str): The name of the analysis (e.g., "Onset of E'").
    - analysis_value (float): The value of the analysis (e.g., the onset X value).
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
        value_column_idx = data.columns.get_loc(metadata_property_column) + 1
        value_column = data.columns[value_column_idx]
        data.at[empty_row_idx, value_column] = analysis_value

        if debug:
            print(f"save_results: Saved '{analysis_name}' with value '{analysis_value}' at row {empty_row_idx}.")
        return data

    except Exception as e:
        print(f"save_results: Error occurred while saving results: {e}")
        return data


def find_onset(x_data, y_data, debug=False, threshold_multiplier=2):
    """
    Detect the onset point where y_data starts to decrease significantly.

    Parameters:
    - x_data (Series or array-like): The X-axis data (e.g., temperature).
    - y_data (Series or array-like): The Y-axis data (e.g., Eprime values).
    - debug (bool): Enable debug logging.
    - threshold_multiplier (float): Multiplier for standard deviation in threshold calculation.

    Returns:
    - float: The X-value corresponding to the onset, or None if not found.
    """
    try:
        # Combine x and y into a DataFrame and drop NaN values
        data = pd.DataFrame({'x': x_data, 'y': y_data}).dropna().reset_index(drop=True)
        x = data['x']
        y = data['y']

        # Calculate the first derivative
        dy_dx = y.diff() / x.diff()

        if debug:
            print(f"First Derivative:\n{dy_dx}")

        # Define a negative threshold to detect significant decrease
        threshold_negative = dy_dx.mean() - threshold_multiplier * dy_dx.std()

        if debug:
            print(f"Onset Threshold (Negative): {threshold_negative}")

        # Detect where the derivative falls below the threshold
        onset_indices = dy_dx[dy_dx < threshold_negative].index

        if not onset_indices.empty:
            onset_index = onset_indices[0]
            onset_x = x.iloc[onset_index]
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

def find_peak(x_data, y_data, debug=False):
    """
    Detect the peak in the data (maximum Y value) and return the corresponding X value.

    Parameters:
    - x_data (Series): The X-axis data (e.g., temperature).
    - y_data (Series): The Y-axis data (e.g., E'' or tan δ).
    - debug (bool): Enable debug logging.

    Returns:
    - (float, float): The X value at the peak and the peak Y value, or (None, None) if no peak is found.
    """
    try:
        # Ensure valid data
        if y_data.empty or x_data.empty:
            if debug:
                print("Find Peak: X or Y data is empty.")
            return None, None

        # Find the peak (maximum Y value)
        peak_idx = y_data.idxmax()
        peak_x = x_data.iloc[peak_idx]
        peak_y = y_data.iloc[peak_idx]

        if debug:
            print(f"Find Peak: Peak detected at X = {peak_x}, Y = {peak_y}")
        return peak_x, peak_y
    except Exception as e:
        if debug:
            print(f"Find Peak: Error during peak detection: {e}")
        return None, None

def analyze_temperature_sweep(data, debug=False):
    """
    Analyze data specific to 'Temperature Sweep' ASSAY.
    Perform onset analysis for E' and peak analysis for E'' and tan δ.

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
    x_data = pd.to_numeric(data[x_column], errors='coerce').dropna()
    if debug:
        print(f"Analyze Temperature Sweep: Identified X-axis column: '{x_column}'")
        print(f"Analyze Temperature Sweep: X-axis data:\n{x_data.head()}")

    # Track completed analyses
    if "Metadata Property" in data.columns:
        completed_analyses = data[data["Metadata Property"] == "Completed Analyses"]
        if not completed_analyses.empty:
            completed_analyses_list = completed_analyses.iloc[0, 1]
            if isinstance(completed_analyses_list, str):
                completed_analyses_list = completed_analyses_list.split(", ")
            else:
                completed_analyses_list = []
        else:
            completed_analyses_list = []
    else:
        completed_analyses_list = []

    # Analyze onset for E' columns
    if debug:
        print (f"Cleaned columns {cleaned_columns}")
    eprime_columns = [col for col in cleaned_columns if re.search(r"\bE'(?!')", col)]

    if debug:
        print (f"found Eprime columns {eprime_columns}")
    for eprime_column in eprime_columns:
        analysis_name = f"Onset of Eprime"
        if analysis_name in completed_analyses_list:
            print(f"Analyze Temperature Sweep: Skipping already completed analysis Eprime.")
            continue

        y_data = pd.to_numeric(data[eprime_column], errors='coerce').dropna()
        if debug:
            print(f"Analyze Temperature Sweep: Analyzing onset for Eprime")
            print(f"X-axis data:\n{x_data}")
            print(f"Y-axis data:\n{y_data}")

        # Perform onset analysis
        onset_x = find_onset(x_data, y_data, debug=debug)

        if onset_x is not None:
            print(f"Analyze Temperature Sweep: Onset detected for Eprime at X = {onset_x}")
            data = save_results(
                data=data,
                metadata_property_column="Metadata Property",
                analysis_name="Onset of Eprime",
                analysis_value=onset_x,
                debug=debug
            )
            completed_analyses_list.append(analysis_name)

    # Analyze peaks for E'' (Edoubleprime)
    matching_columns_edoubleprime = [col for col in cleaned_columns if re.search(r"E''", col, re.IGNORECASE)]
    for column in matching_columns_edoubleprime:
        analysis_name = f"Peak of Edoubleprime"
        if analysis_name in completed_analyses_list:
            print(f"Analyze Temperature Sweep: Skipping already completed analysis '{analysis_name}'.")
            continue

        # Perform peak analysis for E''
        y_data = pd.to_numeric(data[column], errors='coerce').dropna()
        peak_x, peak_y = find_peak(x_data, y_data, debug=debug)

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

    # Analyze peaks for tan δ (Tand)
    matching_columns_tand = [col for col in cleaned_columns if re.search(r"tan ?δ|tan d", col, re.IGNORECASE)]
    for column in matching_columns_tand:
        analysis_name = f"Peak of tand"
        if analysis_name in completed_analyses_list:
            print(f"Analyze Temperature Sweep: Skipping already completed analysis '{analysis_name}'.")
            continue

        # Perform peak analysis for tan δ
        y_data = pd.to_numeric(data[column], errors='coerce').dropna()
        peak_x, peak_y = find_peak(x_data, y_data, debug=debug)

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


def analyze(csv_path, debug=True):
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
    analyze(csv_file)  # Call the analyze function, which saves the analyzed CSV

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