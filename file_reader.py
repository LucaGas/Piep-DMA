# file_reader.py

from chardet import detect
import pandas as pd
from pathlib import Path

def read_file(file_path, assay, debug=False):
    """
    Read a .txt file, determine its format, and call the appropriate extraction function.
    """
    try:
        # Detect file encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            detected = detect(raw_data)
            detected_encoding = detected['encoding']
            confidence = detected['confidence']
            if debug:
                print(f"Detected encoding: {detected_encoding} with confidence {confidence}")

        # If detection confidence is low or encoding is None, use a fallback
        if not detected_encoding or confidence < 0.5:
            if debug:
                print("Low confidence in detected encoding. Using 'latin1' as fallback.")
            detected_encoding = 'latin1'  # Fallback encoding

        if debug:
            print(f"Using encoding: {detected_encoding} for file '{file_path}'")

        # Open the file and read lines
        with open(file_path, 'r', encoding=detected_encoding, errors='replace') as f:
            for line in f:
                # Process only lines starting with a single '#'
                if line.startswith('#') and not line.startswith('##'):
                    # Determine if the format is multi or single
                    is_multi = len(line.strip().split('\t')) > 2  # Check for multiple tab-separated columns
                    if debug:
                        print(f"File '{file_path}' detected as {'multi' if is_multi else 'single'} format.")

                    # Call the appropriate extraction function
                    if is_multi:
                        return extract_multi(file_path, assay, debug=debug)
                    else:
                        return extract_single(file_path, assay, debug=debug)

        if debug:
            print(f"No valid metadata line found to determine file format in '{file_path}'.")
        return None, None

    except Exception as e:
        if debug:
            print(f"An error occurred while processing the file '{file_path}': {e}")
        return None, None

def extract_common_header(headers, debug=False):
    """
    Identify the common header as the one containing 'Temp' or 'Time'.
    Priority: 'Temp' > 'Time' > first column.
    """
    temp_headers = [h for h in headers if 'temp' in h.lower()]
    if temp_headers:
        if debug:
            print(f"Identified common header based on 'Temp': {temp_headers[0]}")
        return temp_headers[0]
    
    time_headers = [h for h in headers if 'time' in h.lower()]
    if time_headers:
        if debug:
            print(f"Identified common header based on 'Time': {time_headers[0]}")
        return time_headers[0]
    
    # Default to first column if neither 'Temp' nor 'Time' is found
    if headers:
        if debug:
            print(f"No 'Temp' or 'Time' found. Defaulting to first header: {headers[0]}")
        return headers[0]
    else:
        if debug:
            print("No headers found to identify common header.")
        return None

def extract_single(file_path, assay, debug=False):
    """
    Extract data from single format files.
    """
    metadata = {}
    column_headers = None
    data_lines = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()

                # Metadata lines start with one #
                if line.startswith('#') and not line.startswith('##'):
                    # Split metadata into property and value
                    parts = line[1:].split('\t', 1)  # Remove # and split by first tab
                    if len(parts) == 2:
                        key, value = parts
                        metadata[key.strip()] = value.strip()
                        if debug:
                            print(f"Metadata - {key.strip()}: {value.strip()}")
                    continue

                # Skip empty lines
                if not line:
                    continue

                # Column headers start with ##
                if line.startswith('##'):
                    column_headers = line[2:].split('\t')  # Remove ## and split into columns
                    if debug:
                        print(f"Column Headers: {column_headers}")
                    continue

                # Remaining lines are data
                data = line.split('\t')
                data_lines.append(data)

        # Add ASSAY to metadata
        metadata['ASSAY'] = assay

        # Extract FILE property from metadata
        file_property = metadata.get('FILE', Path(file_path).stem)
        metadata['FILE'] = file_property

        # Determine the common header
        if column_headers:
            common_header = extract_common_header(column_headers, debug=debug)
        else:
            common_header = None

        # Create DataFrame
        if column_headers and data_lines:
            data = pd.DataFrame(data_lines, columns=column_headers)

            # Naming the experiment by combining ASSAY and FILE
            experiment_name = f"{metadata.get('ASSAY', 'Unknown_Assay')} {metadata.get('FILE', 'Unknown_File')}"
            if debug:
                print(f"Experiment Name: {experiment_name}")
                print("Metadata extracted:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")

                print("\nColumn Headers:")
                print(column_headers)

                print("\nData Preview (without headers):")
                print(data.head())

            return metadata, {experiment_name: data}
        else:
            if debug:
                print("No valid data or column headers found.")
            return metadata, None

    except Exception as e:
        if debug:
            print(f"An error occurred while processing the file: {e}")
        return None, None

def extract_multi(file_path, assay, debug=False):
    """
    Extract data from multi-format files representing multiple experiments.
    """
    metadata = {}
    experiments = {}
    common_header = None
    experiment_headers = []
    experiment_names = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()

                # Metadata lines start with a single '#'
                if line.startswith('#') and not line.startswith('##'):
                    # Split metadata into property-value pairs
                    parts = line.split('#')[1:]  # Split by '#' and ignore the first empty split
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        if '\t' in part:
                            key, value = part.split('\t', 1)
                            metadata[key.strip()] = value.strip()
                            if debug:
                                print(f"Metadata - {key.strip()}: {value.strip()}")
                        else:
                            if debug:
                                print(f"Line {line_number}: Ignored malformed metadata part '{part}'.")
                    continue

                # Skip empty lines
                if not line:
                    continue

                # Column headers line
                if line.startswith('##') and not experiment_headers:
                    headers = line[2:].split('\t')
                    if debug:
                        print(f"Column Headers: {headers}")
                    # Determine the common header dynamically
                    common_header = extract_common_header(headers, debug=debug)
                    if not common_header:
                        if debug:
                            print(f"Line {line_number}: Unable to identify common header. Skipping file.")
                        return metadata, None
                    # Identify the index of the common header
                    try:
                        common_header_idx = headers.index(common_header)
                    except ValueError:
                        if debug:
                            print(f"Line {line_number}: Common header '{common_header}' not found in headers. Skipping file.")
                        return metadata, None
                    # The rest are experiment headers (excluding the common header)
                    experiment_headers = headers[:common_header_idx] + headers[common_header_idx+1:]
                    if debug:
                        print(f"Common Header: {common_header}")
                        print(f"Experiment Headers: {experiment_headers}")
                    # Initialize experiments dictionary
                    for idx, exp_header in enumerate(experiment_headers):
                        # Naming each experiment by combining ASSAY and FILE
                        file_property = metadata.get('FILE', Path(file_path).stem)
                        experiment_name = f"{assay} {file_property}"
                        # Handle duplicate experiment names by maintaining a count
                        if experiment_name in experiments:
                            # Append unique identifier to experiment name
                            unique_exp_name = f"{experiment_name} {idx+1}"
                            experiments[unique_exp_name] = {
                                'header': exp_header,
                                'data': []
                            }
                            if debug:
                                print(f"Duplicate Experiment Name detected: {experiment_name}. Adding new experiment '{unique_exp_name}' with header '{exp_header}'.")
                        else:
                            experiments[experiment_name] = {
                                'header': exp_header,
                                'data': []
                            }
                            if debug:
                                print(f"Initialized {experiment_name} with header '{exp_header}'")
                    continue

                # Data lines
                if experiment_headers:
                    data_parts = line.split('\t')
                    if len(data_parts) < 1:
                        if debug:
                            print(f"Line {line_number}: No data found. Skipping line.")
                        continue
                    try:
                        common_value = data_parts[common_header_idx].strip()
                    except IndexError:
                        if debug:
                            print(f"Line {line_number}: Missing common header value. Skipping line.")
                        continue

                    for idx, exp_name in enumerate(experiments.keys()):
                        # Calculate the index for this experiment's data
                        # Since we excluded the common header, data_index = original index - 1
                        data_index = idx + 1
                        if data_index >= len(data_parts):
                            if debug:
                                print(f"Line {line_number}: Missing value for {exp_name}. Skipping this data point.")
                            continue
                        exp_value = data_parts[data_index].strip()
                        if exp_value:
                            experiments[exp_name]['data'].append({
                                common_header: common_value,
                                experiments[exp_name]['header']: exp_value
                            })
                            if debug:
                                print(f"Line {line_number}: Added data to {exp_name}")
                        else:
                            if debug:
                                print(f"Line {line_number}: Missing value for {exp_name}. Skipping this data point.")

        # Add ASSAY to metadata
        metadata['ASSAY'] = assay

        # Extract FILE property from metadata
        file_property = metadata.get('FILE', Path(file_path).stem)
        metadata['FILE'] = file_property

        # Consolidate experiments with the same name
        consolidated_experiments = {}
        for exp_name, exp_info in experiments.items():
            df = pd.DataFrame(exp_info['data'])
            if not df.empty:
                # Check if this experiment name already exists in consolidated_experiments
                if exp_name in consolidated_experiments:
                    existing_df = consolidated_experiments[exp_name]
                    # Check if columns are the same
                    if set(existing_df.columns) != set(df.columns):
                        # Merge the DataFrames by adding new columns based on common_header
                        merged_df = pd.merge(existing_df, df, on=common_header, how='outer', suffixes=('', '_dup'))
                        # Drop duplicate columns
                        cols_to_drop = [col for col in merged_df.columns if col.endswith('_dup')]
                        merged_df.drop(columns=cols_to_drop, inplace=True)
                        consolidated_experiments[exp_name] = merged_df
                        if debug:
                            print(f"Merged DataFrames for experiment: {exp_name} due to differing columns.")
                    else:
                        # If columns are same, append rows
                        consolidated_experiments[exp_name] = pd.concat([existing_df, df], ignore_index=True)
                        if debug:
                            print(f"Appended rows to existing experiment: {exp_name}")
                else:
                    consolidated_experiments[exp_name] = df
                    if debug:
                        print(f"Added experiment to consolidated_experiments: {exp_name}")
            else:
                if debug:
                    print(f"Experiment {exp_name} has no complete data and will be excluded.")

        return metadata, consolidated_experiments

    except Exception as e:
        if debug:
            print(f"An error occurred while processing the file '{file_path}': {e}")
        return metadata, experiments  # Return what has been processed so far
