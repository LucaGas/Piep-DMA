# post-processor.py

import sys
import pandas as pd
from pathlib import Path
import logging
import re
from plotter import generate_bar_graph  # Import the updated plotting function

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

    # Define the central post-process directory within the output directory
    central_post_process_dir = output_dir / "post-process"
    central_post_process_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured existence of central post-process directory '{central_post_process_dir}'.")

    # Categorize files by ASSAY and SAMPLE
    categorized_data = {}

    for csv_file, df in all_data:
        # Extract ASSAY and SAMPLE metadata
        assay = get_metadata_value(df, 'ASSAY')
        sample = get_metadata_value(df, 'SAMPLE:')  # Ensure 'SAMPLE:' matches your metadata property name

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

    # Initialize accumulators for summaries
    summary_data = {}
    file_metadata = {}

    # Process each category
    for assay, samples in categorized_data.items():
        for sample, files in samples.items():
            logger.info(f"Processing ASSAY '{assay}', SAMPLE '{sample}' with {len(files)} file(s).")

            for csv_file, df in files:
                # Run the specific processing function based on ASSAY
                if assay.lower() == "temperature sweep":
                    extracted_values = process_temperature_sweep_data(csv_file, df, central_post_process_dir)

                    # Initialize summary structures if not already present
                    if assay not in summary_data:
                        summary_data[assay] = {}
                    if sample not in summary_data[assay]:
                        summary_data[assay][sample] = {
                            'Onset of Eprime': [],
                            'Peak of Edoubleprime': [],
                            'Peak of tand': []
                        }

                    # Collect 'Onset of Eprime' values
                    onset = extracted_values.get('Onset of Eprime')
                    if onset is not None:
                        summary_data[assay][sample]['Onset of Eprime'].append(onset)

                    # Collect 'Peak of Edoubleprime' values
                    peak_edoubleprime = extracted_values.get('Peak of Edoubleprime X'), extracted_values.get('Peak of Edoubleprime Y')
                    if all(peak_edoubleprime):
                        # Average X and Y for a single value
                        average_peak_edoubleprime = (peak_edoubleprime[0] + peak_edoubleprime[1]) / 2
                        summary_data[assay][sample]['Peak of Edoubleprime'].append(average_peak_edoubleprime)

                    # Collect 'Peak of tand' values
                    peak_tand = extracted_values.get('Peak of tand X'), extracted_values.get('Peak of tand Y')
                    if all(peak_tand):
                        average_peak_tand = (peak_tand[0] + peak_tand[1]) / 2
                        summary_data[assay][sample]['Peak of tand'].append(average_peak_tand)

                    # Collect 'FILE:' metadata
                    file_meta = get_metadata_value(df, 'FILE:')
                    if file_meta:
                        if assay not in file_metadata:
                            file_metadata[assay] = {}
                        if sample not in file_metadata[assay]:
                            file_metadata[assay][sample] = []
                        file_metadata[assay][sample].append(file_meta)
                else:
                    logger.warning(f"No processing function defined for ASSAY '{assay}' in '{csv_file.name}'. Skipping.")

    # Create an Excel writer
    excel_path = output_dir / "Summary_Averages.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 1. Create the Summary Sheet
        summary_sheet_data = []
        for assay, samples in file_metadata.items():
            for sample, files in samples.items():
                summary_sheet_data.append({
                    'ASSAY': assay,
                    'SAMPLE': sample,
                    'FILES': "; ".join(files)
                })
        summary_df = pd.DataFrame(summary_sheet_data)

        # Reformat Summary Sheet: First row sample names, below rows FILE: metadata
        # Pivot the DataFrame to have samples as columns and FILEs as rows
        pivot_data = {}
        for entry in summary_sheet_data:
            sample = entry['SAMPLE']
            files = entry['FILES'].split("; ")
            if sample not in pivot_data:
                pivot_data[sample] = []
            pivot_data[sample].extend(files)

        # Determine the maximum number of FILEs among samples
        max_files = max(len(files) for files in pivot_data.values()) if pivot_data else 0

        # Create a DataFrame with samples as columns and FILEs as rows
        summary_pivot_df = pd.DataFrame({
            sample: files + [None]*(max_files - len(files))
            for sample, files in pivot_data.items()
        })

        # Write the Summary sheet
        summary_pivot_df.to_excel(writer, sheet_name='Summary', index=False)

        # 2. Aggregate Metrics Across All Samples
        # Initialize a dictionary to hold metric data across all samples
        metrics_summary = {}

        for assay, samples in summary_data.items():
            for sample, metrics in samples.items():
                for metric, values in metrics.items():
                    if values:
                        average = sum(values) / len(values)
                        std_dev = pd.Series(values).std()
                        if metric not in metrics_summary:
                            metrics_summary[metric] = {}
                        metrics_summary[metric][sample] = {
                            'average': average,
                            'std': std_dev if pd.notna(std_dev) else 0  # Replace NaN std with 0
                        }

        # 3. Generate and Save Bar Graphs for Each Metric
        for metric, samples_data in metrics_summary.items():
            # Ensure all samples are included, replacing NaN std with 0
            cleaned_samples_data = {}
            for sample, data in samples_data.items():
                average = data['average']
                std = data['std']
                if pd.isna(std):
                    std = 0
                cleaned_samples_data[sample] = {
                    'average': average,
                    'std': std
                }

            if cleaned_samples_data:
                generate_bar_graph(metric, cleaned_samples_data, central_post_process_dir)
            else:
                logger.warning(f"No valid data to plot for metric '{metric}'. Skipping graph generation.")

        # 4. **Uncommented Section: Create Individual SAMPLE Sheets**
        for assay, samples in summary_data.items():
            for sample, metrics in samples.items():
                sheet_name = f"{sample}"
                sheet_data = {
                    'Metric': [],
                    'Average': [],
                    'STD': []
                }

                # Process each metadata field
                for key, values in metrics.items():
                    if values:
                        average = sum(values) / len(values)
                        std_dev = pd.Series(values).std()
                    else:
                        average = None
                        std_dev = None
                    sheet_data['Metric'].append(key)
                    sheet_data['Average'].append(average)
                    sheet_data['STD'].append(std_dev)

                sheet_df = pd.DataFrame(sheet_data)
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
                logger.info(f"Created Excel sheet for sample '{sample}'.")

        logger.info(f"Saved summary and averages to Excel file '{excel_path}'.")
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
