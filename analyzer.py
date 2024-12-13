# analyzer.py

import sys
import pandas as pd
import subprocess
from pathlib import Path

def analyze(csv_path, debug=False):
    """
    Dummy analysis function.
    In the future, this function will perform data analysis and pass results to plotter.py.
    """
    try:
        # Load the CSV file with detected encoding
        data = pd.read_csv(csv_path, encoding='utf-8')
        if debug:
            print(f"Analyzer: Received data from '{csv_path}' with {len(data)} records.")
            print("Analyzer: Displaying first few rows of the data:")
            print(data.head())

        # Placeholder for future analysis logic
        # For now, just print a confirmation message
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
    print(f"Analyzer: Received {csv_file} for analysis.")

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
