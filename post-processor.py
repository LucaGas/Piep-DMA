# post-processor.py

import sys
import pandas as pd
from pathlib import Path
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("post_processor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def find_analyzed_csv_files(output_dir):
    """
    Recursively find all *_analyzed.csv files within the output directory.

    Parameters:
        output_dir (Path): The base output directory to search within.

    Returns:
        List[Path]: A list of Paths to the found analyzed CSV files.
    """
    logger.info(f"Scanning for analyzed CSV files in '{output_dir}'...")
    analyzed_csv_files = list(output_dir.rglob("*_analyzed.csv"))
    logger.info(f"Found {len(analyzed_csv_files)} analyzed CSV file(s).")
    return analyzed_csv_files

def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
        file_path (Path): Path to the CSV file.

    Returns:
        DataFrame: The loaded DataFrame, or None if loading fails.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded '{file_path}'.")
        return df
    except Exception as e:
        logger.error(f"Failed to load '{file_path}': {e}")
        return None

def get_metadata_value(df, property_name):
    """
    Extract the metadata value for a given property from the DataFrame.

    Parameters:
        df (DataFrame): The loaded DataFrame.
        property_name (str): The metadata property to extract.

    Returns:
        str: The metadata value, or None if not found.
    """
    try:
        # Ensure case-insensitive matching and strip any whitespace
        mask = df['Metadata Property'].str.lower().str.strip() == property_name.lower().strip()
        value = df.loc[mask, 'Metadata Value'].values[0]
        logger.debug(f"Extracted '{property_name}': '{value}'")
        return value
    except IndexError:
        logger.warning(f"Metadata Property '{property_name}' not found.")
        return None
    except Exception as e:
        logger.error(f"Error extracting metadata '{property_name}': {e}")
        return None
import re  # Ensure this import is present at the top of your script

def process_temperature_sweep_data(csv_file, df, post_process_dir):
    """
    Process Temperature Sweep data by extracting specific metadata.

    Parameters:
        csv_file (Path): Path to the analyzed CSV file.
        df (DataFrame): The loaded DataFrame.
        post_process_dir (Path): Directory to save the processing results.

    Returns:
        dict: Extracted metadata values.
    """
    logger.info(f"Processing Temperature Sweep data for '{csv_file.name}'.")

    extracted_values = {}

    # Extract 'Onset of Eprime'
    try:
        onset = get_metadata_value(df, 'Onset of Eprime')
        if onset is not None:
            extracted_values['Onset of Eprime'] = float(onset)
        else:
            extracted_values['Onset of Eprime'] = None
    except Exception as e:
        logger.error(f"Error extracting 'Onset of Eprime' from '{csv_file.name}': {e}")
        extracted_values['Onset of Eprime'] = None

    # Extract 'Peak of Edoubleprime'
    try:
        peak_edoubleprime = get_metadata_value(df, 'Peak of Edoubleprime')
        if peak_edoubleprime is not None:
            match = re.match(r'X:\s*([\d\.]+),\s*Y:\s*([\d\.]+)', peak_edoubleprime)
            if match:
                extracted_values['Peak of Edoubleprime X'] = float(match.group(1))
                extracted_values['Peak of Edoubleprime Y'] = float(match.group(2))
            else:
                logger.warning(f"Unexpected format for 'Peak of Edoubleprime' in '{csv_file.name}': '{peak_edoubleprime}'")
                extracted_values['Peak of Edoubleprime X'] = None
                extracted_values['Peak of Edoubleprime Y'] = None
        else:
            extracted_values['Peak of Edoubleprime X'] = None
            extracted_values['Peak of Edoubleprime Y'] = None
    except Exception as e:
        logger.error(f"Error extracting 'Peak of Edoubleprime' from '{csv_file.name}': {e}")
        extracted_values['Peak of Edoubleprime X'] = None
        extracted_values['Peak of Edoubleprime Y'] = None

    # Extract 'Peak of tand'
    try:
        peak_tand = get_metadata_value(df, 'Peak of tand')
        if peak_tand is not None:
            match = re.match(r'X:\s*([\d\.]+),\s*Y:\s*([\d\.]+)', peak_tand)
            if match:
                extracted_values['Peak of tand X'] = float(match.group(1))
                extracted_values['Peak of tand Y'] = float(match.group(2))
            else:
                logger.warning(f"Unexpected format for 'Peak of tand' in '{csv_file.name}': '{peak_tand}'")
                extracted_values['Peak of tand X'] = None
                extracted_values['Peak of tand Y'] = None
        else:
            extracted_values['Peak of tand X'] = None
            extracted_values['Peak of tand Y'] = None
    except Exception as e:
        logger.error(f"Error extracting 'Peak of tand' from '{csv_file.name}': {e}")
        extracted_values['Peak of tand X'] = None
        extracted_values['Peak of tand Y'] = None

    return extracted_values

def perform_post_processing(all_data, output_dir):
    """
    Perform post-processing on all analyzed CSV files.

    Parameters:
        all_data (List[Tuple[Path, DataFrame]]): A list of tuples containing file paths and their corresponding DataFrames.
        output_dir (Path): The base output directory.

    Returns:
        None
    """
    logger.info("Starting post-processing of analyzed CSV files...")

    # Categorize files by ASSAY and SAMPLE
    categorized_data = {}

    for csv_file, df in all_data:
        # Extract ASSAY and SAMPLE metadata
        assay = get_metadata_value(df, 'ASSAY')
        sample = get_metadata_value(df, 'SAMPLE:')  # Note the change to 'SAMPLE:'

        if not assay:
            logger.warning(f"Skipping '{csv_file.name}' due to missing ASSAY metadata.")
            continue
        if not sample:
            logger.warning(f"Skipping '{csv_file.name}' due to missing SAMPLE metadata.")
            continue

        # Initialize dictionaries if not already present
        if assay not in categorized_data:
            categorized_data[assay] = {}
        if sample not in categorized_data[assay]:
            categorized_data[assay][sample] = []

        # Append the csv file and its DataFrame to the appropriate category
        categorized_data[assay][sample].append((csv_file, df))
        logger.debug(f"Categorized '{csv_file.name}' under ASSAY '{assay}' and SAMPLE '{sample}'.")

    # Initialize a dictionary to accumulate extracted data per ASSAY/SAMPLE
    data_accumulator = {}
    # Structure: {assay: {sample: {'Onset of Eprime': [...], 'Peak of Edoubleprime X': [...], ...}}}

    # Process each category
    for assay, samples in categorized_data.items():
        for sample, files in samples.items():
            logger.info(f"Processing ASSAY '{assay}', SAMPLE '{sample}' with {len(files)} file(s).")

            # Initialize accumulator for the current ASSAY/SAMPLE
            if assay not in data_accumulator:
                data_accumulator[assay] = {}
            if sample not in data_accumulator[assay]:
                data_accumulator[assay][sample] = {
                    'Onset of Eprime': [],
                    'Peak of Edoubleprime X': [],
                    'Peak of Edoubleprime Y': [],
                    'Peak of tand X': [],
                    'Peak of tand Y': []
                }

            for csv_file, df in files:
                # Define the post-process directory within the same directory as the CSV file
                post_process_dir = csv_file.parent / "post-process"
                post_process_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured existence of post-process directory '{post_process_dir}'.")

                # Run the specific processing function based on ASSAY
                if assay.lower() == "temperature sweep":
                    extracted_values = process_temperature_sweep_data(csv_file, df, post_process_dir)

                    # Accumulate extracted values
                    for key, value in extracted_values.items():
                        if value is not None:
                            data_accumulator[assay][sample][key].append(value)
                else:
                    logger.warning(f"No processing function defined for ASSAY '{assay}' in '{csv_file.name}'. Skipping.")

    # Compute averages and save summaries
    for assay, samples in data_accumulator.items():
        for sample, metrics in samples.items():
            logger.info(f"Computing averages for ASSAY '{assay}', SAMPLE '{sample}'.")

            averages = {}
            for key, values in metrics.items():
                if values:
                    averages[key] = sum(values) / len(values)
                else:
                    averages[key] = None

            # Create a DataFrame for the averages
            summary_df = pd.DataFrame([averages])

            # Define the summary file path
            summary_file = post_process_dir / "Summary_Averages.csv"

            # Save the summary to CSV
            try:
                summary_df.to_csv(summary_file, index=False)
                logger.info(f"Saved summary averages to '{summary_file}'.")
            except Exception as e:
                logger.error(f"Error saving summary averages to '{summary_file}': {e}")

    logger.info("Post-processing of analyzed CSV files completed.")

def main():
    """
    Main function to execute the post-processor script.
    """
    if len(sys.argv) < 2:
        print("Usage: python post-processor.py <output_directory>")
        sys.exit(1)

    output_directory = Path(sys.argv[1])

    if not output_directory.exists():
        logger.error(f"The specified output directory '{output_directory}' does not exist.")
        sys.exit(1)

    # Find all analyzed CSV files
    analyzed_csv_files = find_analyzed_csv_files(output_directory)

    if not analyzed_csv_files:
        logger.warning("No analyzed CSV files found. Exiting.")
        sys.exit(0)

    # Load all analyzed CSV files
    all_loaded_data = []
    for csv_file in analyzed_csv_files:
        df = load_csv(csv_file)
        if df is not None:
            all_loaded_data.append((csv_file, df))

    if not all_loaded_data:
        logger.warning("No valid analyzed CSV files loaded. Exiting.")
        sys.exit(0)

    # Perform the desired post-processing
    perform_post_processing(all_loaded_data, output_directory)

    logger.info("Post-processing complete.")

if __name__ == "__main__":
    main()
