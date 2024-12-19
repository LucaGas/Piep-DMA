# analyzer.py

import sys
import pandas as pd
import subprocess
from pathlib import Path
import re
import logging
from utils import save_results, get_aligned_data, map_column_name

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def analyze(csv_path, debug=False):
    """
    Main analysis function that extracts and delegates analysis based on the 'ASSAY' metadata.

    Parameters:
    - csv_path (str): Path to the input CSV file.
    - debug (bool): Enable debug logging.
    """
    try:
        # Load the CSV file
        data = pd.read_csv(csv_path, encoding='utf-8')
        if debug:
            logger.debug(f"Analyzer: Received data from '{csv_path}' with {len(data)} records.")
            logger.debug("Analyzer: Displaying first few rows of the data:")
            logger.debug(data.head())

        # Extract 'ASSAY' value
        assay_value = None
        if "Metadata Property" in data.columns:
            # Find the row where "Metadata Property" equals "ASSAY"
            assay_row = data[data["Metadata Property"].str.lower() == "assay"]
            if not assay_row.empty:
                # Identify the column immediately to the right of "Metadata Property"
                metadata_property_index = data.columns.get_loc("Metadata Property")
                value_column_index = metadata_property_index + 1
                if value_column_index < len(data.columns):
                    assay_value = assay_row.iloc[0, value_column_index]

        if assay_value:
            print(f"Analyzer: Global metadata ASSAY = '{assay_value}'")
            logger.info(f"Analyzer: Detected ASSAY type: '{assay_value}'")
        else:
            print("Analyzer: Metadata 'ASSAY' not found in the CSV.")
            logger.warning("Analyzer: Metadata 'ASSAY' not found in the CSV.")

        # Define assay script mapping
        assay_scripts = {
            "Temperature Sweep": "analyze_temperature_sweep.py",
            "Frequency Sweep": "frequency_sweep.py",
            # Add mappings for additional assays here
            # "Other Assay": "analyze_other_assay.py",
        }

        # Check if the assay is supported
        if assay_value in assay_scripts:
            assay_script = assay_scripts[assay_value]
            assay_script_path = Path(__file__).parent / "assays" / assay_script

            if not assay_script_path.exists():
                logger.error(f"Analyzer: Assay script '{assay_script}' not found in 'assays/' directory.")
                return

            print(f"Analyzer: Detected '{assay_value}'. Invoking '{assay_script}'.")
            logger.info(f"Analyzer: Invoking '{assay_script}' for analysis.")

            # Call the assay-specific analysis script
            result = subprocess.run(
                [sys.executable, str(assay_script_path), csv_path],
                check=True,
                capture_output=True,
                text=True
            )

            # Output the results from the assay script
            print(f"Analyzer: {assay_script} Output:\n{result.stdout}")
            if result.stderr:
                print(f"Analyzer: {assay_script} Errors/Warnings:\n{result.stderr}")

            print(f"Analyzer: Analysis complete. Results saved to '{csv_path.replace('.csv', '_analyzed.csv')}'.")
            logger.info("Analyzer: Analysis complete.")

            # Define the path to plotter.py
            plotter_script_path = Path(__file__).parent / "plotter.py"

            if not plotter_script_path.exists():
                logger.error(f"Analyzer: Plotter script 'plotter.py' not found in the root directory.")
            else:
                try:
                    # Call plotter.py with the analyzed CSV
                    plot_result = subprocess.run(
                        [sys.executable, str(plotter_script_path), csv_path.replace(".csv", "_analyzed.csv")],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    print(f"Analyzer: Plotter completed for '{csv_path.replace('.csv', '_analyzed.csv')}'.")
                    print(f"Plotter Output:\n{plot_result.stdout}")
                    if plot_result.stderr:
                        print(f"Plotter Errors/Warnings:\n{plot_result.stderr}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Analyzer: Plotter failed with error: {e}")
                    logger.error(f"Plotter stderr: {e.stderr}")

    except FileNotFoundError:
        logger.error(f"Analyzer Error: The file '{csv_path}' does not exist.")
    except pd.errors.EmptyDataError:
        logger.error(f"Analyzer Error: The file '{csv_path}' is empty.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Analyzer: Assay script failed with error: {e}")
        logger.error(f"Analyzer: Assay Script stderr: {e.stderr}")
    except Exception as e:
        logger.error(f"Analyzer Error: An unexpected error occurred - {e}")


def main():
    """
    Entry point for the analyzer script.
    """
    if len(sys.argv) < 2:
        print("Analyzer: No CSV file provided.")
        print("Usage: python analyzer.py <path_to_csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    print(f"Analyzer: Received '{csv_file}' for analysis.")
    debug = True  # Set to False to reduce verbosity

    # Perform analysis
    analyze(csv_file, debug=debug)


if __name__ == "__main__":
    main()
