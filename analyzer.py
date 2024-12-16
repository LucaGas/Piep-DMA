# analyzer.py

import sys
import pandas as pd
import subprocess
from pathlib import Path
import re

def analyze_temperature_sweep(data, debug=False):
    """
    Analyze data specific to 'Temperature Sweep' ASSAY.

    Parameters:
    - data (DataFrame): The loaded CSV data.
    - debug (bool): Enable debug logging.
    """
    if debug:
        print("Analyze Temperature Sweep: Received the following data:")
        print(data.head())

    # Debug column names
    if debug:
        print("Analyze Temperature Sweep: Column names:")
        for col in data.columns:
            print(f"'{col}'")

    # Clean column names (trim whitespace and standardize encoding)
    cleaned_columns = [col.strip() for col in data.columns]

    # Check for columns containing "E'"
    eprime_columns = [
        col for col in cleaned_columns if re.search(r"E'", col)
    ]
    if eprime_columns:
        print("Eprime found")
        if debug:
            print(f"Analyze Temperature Sweep: Identified 'Eprime' columns: {eprime_columns}")
    else:
        if debug:
            print("Analyze Temperature Sweep: No 'Eprime' columns found.")

    # Placeholder for further analysis
    print("Analyze Temperature Sweep: Analysis complete.")


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
            analyze_temperature_sweep(data, debug=debug)

        # Placeholder for future analysis logic
        print(f"Analyzer: Successfully analyzed '{csv_path}'.")

        # Call plotter.py with the path to the saved CSV
        try:
            if debug:
                print(f"Analyzer: Calling plotter.py for '{csv_path}'")
            subprocess.run([sys.executable, "plotter.py", str(csv_path)], check=True)
            if debug:
                print(f"Analyzer: Plotter completed for '{csv_path}'")
        except subprocess.CalledProcessError as e:
            print(f"Analyzer Error: Plotter failed for '{csv_path}' with error: {e}")

    except FileNotFoundError:
        print(f"Analyzer Error: The file '{csv_path}' does not exist.")
    except pd.errors.EmptyDataError:
        print(f"Analyzer Error: The file '{csv_path}' is empty.")
    except Exception as e:
        print(f"Analyzer Error: An unexpected error occurred - {e}")

def main():
    """
    Analyzer that passes the CSV file to plotter.py for plotting.
    """
    if len(sys.argv) < 2:
        print("Analyzer: No CSV file provided.")
        sys.exit(1)

    csv_file = sys.argv[1]
    print(f"Analyzer: Received '{csv_file}' for analysis.")
    analyze(csv_file)  # Call the analyze function
    try:
        # Call plotter.py as a standalone script with the CSV file
        print(f"Analyzer: Calling plotter.py for {csv_file}")
        result = subprocess.run(
            [sys.executable, "plotter.py", csv_file],
            check=True,
            capture_output=True,  # Capture stdout and stderr for logging
            text=True  # Decode stdout and stderr as text
        )
        print(f"Analyzer: Plotter completed for {csv_file}")
        print(f"Plotter Output:\n{result.stdout}")
        if result.stderr:
            print(f"Plotter Warnings/Errors:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Analyzer: Plotter failed for {csv_file} with error: {e}")
        print(f"Plotter Error Output:\n{e.stderr}")
    except Exception as ex:
        print(f"Analyzer: Unexpected error while calling plotter.py: {ex}")

if __name__ == "__main__":
    main()
