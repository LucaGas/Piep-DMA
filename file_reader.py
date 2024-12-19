# file_reader.py

import pandas as pd  # Import pandas
import numpy as np
from pathlib import Path
import logging
from pprint import pprint
import re
from utils import setup_logging, sanitize_column_name  # Import necessary utilities


def read_file(file_path, assay):
    """
    Read a .txt file, determine its format (single or multi), and extract metadata and data.

    Parameters:
        file_path (str or Path): Path to the .txt file.
        assay (str): Assay type.

    Returns:
        tuple: (metadata dictionary, structured_output dictionary)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reading file '{file_path}'.")

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        logger.debug(f"Successfully read {len(lines)} lines from '{file_path}'.")

        # Determine if the file is multi-format by checking the #FILE: metadata line
        file_metadata_lines = [line for line in lines if line.startswith('#FILE:')]

        if file_metadata_lines:
            # Split the first #FILE: line to count the number of entries
            file_values = file_metadata_lines[0].split('\t')[1:]  # Skip the '#FILE:' key
            file_values = [value.strip() for value in file_values if value.strip()]  # Clean empty values

            logger.debug(f"Found FILE metadata entries: {file_values}")

            if len(file_values) > 1:
                logger.info(f"File '{file_path}' detected as multi-format with {len(file_values)} FILE entries.")
                return extract_multi(lines, file_path, assay)
            else:
                logger.info(f"File '{file_path}' detected as single-format with one FILE entry.")
                return extract_single(lines, file_path, assay)
        else:
            logger.warning(f"No '#FILE:' metadata found in '{file_path}'. Defaulting to single-format.")
            return extract_single(lines, file_path, assay)

    except FileNotFoundError:
        logger.error(f"File Reader: The file '{file_path}' does not exist.")
        return None, None
    except pd.errors.EmptyDataError:
        logger.error(f"File Reader: The file '{file_path}' is empty.")
        return None, None
    except Exception as e:
        logger.error(f"File Reader: An error occurred while processing '{file_path}': {e}")
        return None, None


def extract_common_header(headers):
    """
    Identify the common header as the one containing 'Temp' or 'Time'.
    Priority: 'Temp' > 'Time' > first column.

    Parameters:
        headers (list): List of header strings.

    Returns:
        str or None: The identified common header.
    """
    logger = logging.getLogger(__name__)
    temp_headers = [h for h in headers if 'temp' in h.lower()]
    if temp_headers:
        logger.debug(f"Identified common header based on 'Temp': {temp_headers[0]}")
        return temp_headers[0]

    time_headers = [h for h in headers if 'time' in h.lower()]
    if time_headers:
        logger.debug(f"Identified common header based on 'Time': {time_headers[0]}")
        return time_headers[0]

    if headers:
        logger.debug(f"No 'Temp' or 'Time' found. Defaulting to first header: {headers[0]}")
        return headers[0]
    else:
        logger.warning("No headers found to identify common header.")
        return None


def extract_single(lines, file_path, assay):
    """
    Extract data from single format files and structure it like multi-format files,
    handling multiple header lines within the same file.

    Parameters:
        lines (list): List of lines from the file.
        file_path (str or Path): Path to the file.
        assay (str): Assay type.

    Returns:
        tuple: (metadata dictionary, structured_output dictionary)
    """
    logger = logging.getLogger(__name__)
    metadata = {}
    structured_output = {}
    
    current_header = None
    current_data_lines = []
    experiment_count = 0

    logger.info(f"Extracting single-format data from '{file_path}'.")

    for line_number, line in enumerate(lines, 1):
        line = line.strip()

        # Metadata lines
        if line.startswith('#') and not line.startswith('##'):
            parts = line[1:].split('\t', 1)
            if len(parts) == 2:
                key, value = parts
                metadata[key.strip()] = value.strip()
                logger.debug(f"Extracted metadata: {key.strip()} = {value.strip()}")
            continue

        # Header lines
        if line.startswith('##'):
            # If there's an existing header and data, process it
            if current_header and current_data_lines:
                experiment_count += 1
                experiment_name = f"{metadata.get('ASSAY', 'Unknown_Assay')} {metadata.get('FILE', Path(file_path).stem)}_{experiment_count}"
                data_df = pd.DataFrame(current_data_lines, columns=current_header)
                
                logger.debug(f"Created DataFrame for experiment '{experiment_name}' with {len(data_df)} rows.")

                # Sanitize column names
                original_columns = data_df.columns.tolist()
                data_df.columns = [sanitize_column_name(col) for col in data_df.columns]
                logger.debug(f"Sanitized column names: {data_df.columns.tolist()}")

                # Add a placeholder for the common header if needed
                common_header = extract_common_header(current_header)
                if common_header and common_header not in data_df.columns:
                    data_df[common_header] = pd.NA
                    logger.debug(f"Added placeholder for common header '{common_header}'.")

                # Structure the output
                structured_output[experiment_name] = {
                    'metadata': metadata.copy(),  # Use a copy to prevent overwriting
                    'data': data_df
                }

                logger.info(f"Extracted single experiment structure for '{experiment_name}'.")
                logger.debug("Structured Output:")
                pprint(structured_output[experiment_name])

                # Reset for the next experiment
                current_data_lines = []
            
            # Update the current header
            current_header = line[2:].split('\t')
            logger.debug(f"Extracted column headers: {current_header}")
            continue

        # Data lines
        if line and not line.startswith('#'):
            data = line.split('\t')
            current_data_lines.append(data)
            logger.debug(f"Added data line (Line {line_number}): {data}")

    # After processing all lines, handle the last block if exists
    if current_header and current_data_lines:
        experiment_count += 1
        experiment_name = f"{metadata.get('ASSAY', 'Unknown_Assay')} {metadata.get('FILE', Path(file_path).stem)}_{experiment_count}"
        data_df = pd.DataFrame(current_data_lines, columns=current_header)
        
        logger.debug(f"Created DataFrame for experiment '{experiment_name}' with {len(data_df)} rows.")

        # Sanitize column names
        original_columns = data_df.columns.tolist()
        data_df.columns = [sanitize_column_name(col) for col in data_df.columns]
        logger.debug(f"Sanitized column names: {data_df.columns.tolist()}")

        # Add a placeholder for the common header if needed
        common_header = extract_common_header(current_header)
        if common_header and common_header not in data_df.columns:
            data_df[common_header] = pd.NA
            logger.debug(f"Added placeholder for common header '{common_header}'.")

        # Structure the output
        structured_output[experiment_name] = {
            'metadata': metadata.copy(),  # Use a copy to prevent overwriting
            'data': data_df
        }

        logger.info(f"Extracted single experiment structure for '{experiment_name}'.")
        logger.debug("Structured Output:")
        pprint(structured_output[experiment_name])

    # Add ASSAY to metadata
    metadata['ASSAY'] = assay
    logger.debug(f"Added 'ASSAY' to metadata: {assay}")

    # Extract FILE property from metadata or default to file name
    file_property = metadata.get('FILE', Path(file_path).stem)
    metadata['FILE'] = file_property
    logger.debug(f"Set 'FILE' metadata: {file_property}")

    if structured_output:
        logger.info(f"Successfully extracted {experiment_count} experiment(s) from '{file_path}'.")
    else:
        logger.warning("No valid data or headers found in the single-format file.")

    return metadata, structured_output


def extract_multi(lines, file_path, assay):
    """
    Extract data from multi-format files representing multiple experiments.

    Parameters:
        lines (list): List of lines from the file.
        file_path (str or Path): Path to the file.
        assay (str): Assay type.

    Returns:
        tuple: (metadata dictionary, consolidated_experiments dictionary)
    """
    logger = logging.getLogger(__name__)
    metadata = {}
    consolidated_experiments = {}
    common_header = None
    experiment_headers = []
    num_experiments = 0
    expected_columns = 0

    logger.info(f"Extracting multi-format data from '{file_path}'.")

    for line_number, line in enumerate(lines, 1):
        line = line.strip()

        # Handle metadata lines
        if line.startswith('#') and not line.startswith('##'):
            parts = line[1:].split('\t')
            if not parts:
                continue

            key = parts[0].strip()
            values = parts[1:]

            # Initialize experiments from FILE:
            if key.upper() == 'FILE:' and not consolidated_experiments:
                num_experiments = len(values)
                for i, file_name in enumerate(values):
                    exp_name = file_name.strip() if file_name.strip() else f"Experiment_{i+1}"
                    consolidated_experiments[exp_name] = {
                        'metadata': {'FILE:': exp_name},
                        'data': pd.DataFrame()  # Start with empty DataFrame
                    }
                logger.debug(f"Initialized {num_experiments} experiments based on FILE metadata: {list(consolidated_experiments.keys())}")
                continue

            # Assign metadata to experiments
            if num_experiments and len(values) < num_experiments:
                values.extend([np.nan] * (num_experiments - len(values)))

            for i, exp_name in enumerate(consolidated_experiments.keys()):
                if i < len(values):
                    value = values[i].strip() if isinstance(values[i], str) else np.nan
                    consolidated_experiments[exp_name]['metadata'][key] = value
                    logger.debug(f"Assigned metadata '{key}' = '{value}' to experiment '{exp_name}'.")

            continue

        # Handle column headers
        if line.startswith('##') and not experiment_headers:
            headers = line[2:].split('\t')
            expected_columns = len(headers)
            
            common_header = extract_common_header(headers)
            if not common_header:
                logger.warning("Could not identify common header. Skipping data extraction.")
                return metadata, consolidated_experiments

            try:
                common_header_idx = headers.index(common_header)
                logger.debug(f"Common header '{common_header}' found at index {common_header_idx}.")
            except ValueError:
                logger.error(f"Common header '{common_header}' not found in headers.")
                return metadata, consolidated_experiments

            experiment_headers = headers[:common_header_idx] + headers[common_header_idx + 1:]
            logger.debug(f"Experiment-specific headers: {experiment_headers}")
            continue

        # Handle data lines
        if experiment_headers and not line.startswith('#'):
            data_parts = line.split('\t')

            # Pad missing values if row is incomplete
            if len(data_parts) < expected_columns:
                data_parts.extend([np.nan] * (expected_columns - len(data_parts)))
                logger.debug(f"Padded data parts to match expected columns: {data_parts}")

            # Extract common header value
            try:
                common_value = data_parts[common_header_idx].strip() if isinstance(data_parts[common_header_idx], str) else np.nan
            except IndexError:
                logger.error(f"Line {line_number}: Common header index {common_header_idx} out of range.")
                continue

            # Assign data to each experiment
            # Use the experiment_headers for column keys
            for i, exp_name in enumerate(consolidated_experiments.keys()):
                data_index = i + 1  # Assuming common header is at index 0
                if data_index >= len(data_parts):
                    logger.warning(f"Line {line_number}: Data index {data_index} out of range for experiment '{exp_name}'.")
                    continue

                raw_value = data_parts[data_index]
                exp_value = raw_value.strip() if isinstance(raw_value, str) else np.nan

                # Build row dict
                row = {common_header: common_value}
                # Add experiment column only if exp_value is not empty
                if exp_value and exp_value != '':
                    row[experiment_headers[i]] = exp_value
                    logger.debug(f"Line {line_number}: Added data for experiment '{exp_name}': {row}")

                # Append row to DataFrame
                if not consolidated_experiments[exp_name]['data'].empty:
                    consolidated_experiments[exp_name]['data'] = pd.concat(
                        [consolidated_experiments[exp_name]['data'], pd.DataFrame([row])],
                        ignore_index=True
                    )
                else:
                    consolidated_experiments[exp_name]['data'] = pd.DataFrame([row])
                logger.debug(f"Appended row to experiment '{exp_name}': {row}")

    # Add ASSAY to metadata
    metadata['ASSAY'] = assay
    logger.debug(f"Added 'ASSAY' to metadata: {assay}")

    # Sanitize column names for each experiment's DataFrame
    for exp_name, exp_data in consolidated_experiments.items():
        if exp_data['data'] is not None and not exp_data['data'].empty:
            original_columns = exp_data['data'].columns.tolist()
            exp_data['data'].columns = [sanitize_column_name(col) for col in exp_data['data'].columns]
            logger.debug(f"Sanitized column names for experiment '{exp_name}': {exp_data['data'].columns.tolist()}")

    logger.info("Final experiments structure:")
    pprint(consolidated_experiments)

    return metadata, consolidated_experiments


def main():
    """
    Main function to handle standalone execution of the file_reader script.
    """
    import argparse

    parser = argparse.ArgumentParser(description="File Reader Script for Piep-DMA Project")
    parser.add_argument("file_path", help="Path to the input .txt file.")
    parser.add_argument("assay", help="Assay type.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode with detailed logs.")

    args = parser.parse_args()

    # Set up logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level=log_level, log_file='file_reader.log')  # Customize log_file as needed

    # Acquire a logger for file_reader.py
    logger = logging.getLogger(__name__)

    file_path = args.file_path
    assay = args.assay

    logger.info(f"File Reader: Processing file '{file_path}' with assay '{assay}'.")
    metadata, structured_output = read_file(file_path, assay)

    if metadata and structured_output:
        logger.info("File Reader: Successfully extracted metadata and structured output.")
    else:
        logger.error("File Reader: Failed to extract data.")

    # Optionally, save the structured output to a file or perform further processing
    # For demonstration, we'll print the metadata and structured output
    logger.info("Metadata:")
    pprint(metadata)
    logger.info("Structured Output:")
    pprint(structured_output)


if __name__ == "__main__":
    main()
