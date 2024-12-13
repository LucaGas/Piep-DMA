import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint

def read_file(file_path, assay, debug=False):
    """
    Read a .txt file, determine its format (single or multi), and extract metadata and data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        if debug:
            print(f"Reading file '{file_path}' with UTF-8 encoding.")
        
        # Determine if the file is multi-format by checking the number of metadata entries
        metadata_lines = [line for line in lines if line.startswith('#') and not line.startswith('##')]
        if len(metadata_lines) > 1:
            if debug:
                print(f"File '{file_path}' detected as multi format.")
            return extract_multi(lines, file_path, assay, debug=debug)
        else:
            if debug:
                print(f"File '{file_path}' detected as single format.")
            return extract_single(lines, file_path, assay, debug=debug)
    
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
    
    if headers:
        if debug:
            print(f"No 'Temp' or 'Time' found. Defaulting to first header: {headers[0]}")
        return headers[0]
    else:
        if debug:
            print("No headers found to identify common header.")
        return None

def extract_single(lines, file_path, assay, debug=False):
    """
    Extract data from single format files.
    """
    metadata = {}
    column_headers = None
    data_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Metadata lines start with one #
        if line.startswith('#') and not line.startswith('##'):
            parts = line[1:].split('\t', 1)  # Remove # and split by first tab
            if len(parts) == 2:
                key, value = parts
                metadata[key.strip()] = value.strip()
                if debug:
                    print(f"Metadata - {key.strip()}: {value.strip()}")
            continue
        
        # Column headers start with ##
        if line.startswith('##'):
            column_headers = line[2:].split('\t')  # Remove ## and split into columns
            if debug:
                print(f"Column Headers: {column_headers}")
            continue
        
        # Data lines
        if line and not line.startswith('#'):
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
        data_df = pd.DataFrame(data_lines, columns=column_headers)
        experiment_name = f"{metadata.get('ASSAY', 'Unknown_Assay')} {metadata.get('FILE', 'Unknown_File')}"
        if debug:
            print(f"Experiment Name: {experiment_name}")
            print("Metadata extracted:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            print("\nData Preview:")
            print(data_df.head())
        return metadata, {experiment_name: data_df}
    else:
        if debug:
            print("No valid data or column headers found.")
        return metadata, None

def extract_multi(lines, file_path, assay, debug=False):
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from pprint import pprint

    metadata = {}
    experiments = {}
    common_header = None
    experiment_headers = []
    num_experiments = 0
    expected_columns = 0

    for line_number, line in enumerate(lines, 1):
        line = line.strip()

        # Handle metadata lines (start with '#' and not '##')
        if line.startswith('#') and not line.startswith('##'):
            parts = line[1:].split('\t')
            if not parts:
                continue

            key = parts[0].strip()
            values = parts[1:]

            if debug:
                print(f"Line {line_number}: Metadata Key: '{key}', Values: {values}")

            # Initialize experiments from FILE: line
            if key.upper() == 'FILE:' and not experiments:
                num_experiments = len(values)
                for i, file_name in enumerate(values):
                    exp_name = file_name.strip() if file_name.strip() else f"Experiment_{i+1}"
                    experiments[exp_name] = {
                        'metadata': {'FILE:': exp_name},  # Initialize FILE: metadata
                        'data': []
                    }
                continue

            # Assign metadata to experiments
            if num_experiments and len(values) < num_experiments:
                values.extend([np.nan] * (num_experiments - len(values)))

            for i, exp_name in enumerate(experiments.keys()):
                value = values[i].strip() if i < len(values) and isinstance(values[i], str) else np.nan
                experiments[exp_name]['metadata'][key] = value
                if debug:
                    print(f"Line {line_number}: Metadata - {key}: {value} (Experiment: {exp_name})")

            continue

        # Handle column header line (starts with '##')
        if line.startswith('##') and not experiment_headers:
            headers = line[2:].split('\t')
            expected_columns = len(headers)
            if debug:
                print(f"Line {line_number}: Column Headers: {headers}")

            common_header = extract_common_header(headers, debug=debug)
            if not common_header:
                if debug:
                    print(f"Line {line_number}: Unable to identify common header. Skipping file.")
                return metadata, None

            try:
                common_header_idx = headers.index(common_header)
            except ValueError:
                if debug:
                    print(f"Line {line_number}: Common header '{common_header}' not found. Skipping file.")
                return metadata, None

            experiment_headers = headers[:common_header_idx] + headers[common_header_idx + 1:]
            if debug:
                print(f"Line {line_number}: Common Header: {common_header}")
                print(f"Line {line_number}: Experiment Headers: {experiment_headers}")

            continue

        # Handle data lines
        if experiment_headers and not line.startswith('#'):
            data_parts = line.split('\t')

            # Pad missing values if row is incomplete
            if len(data_parts) < expected_columns:
                data_parts.extend([np.nan] * (expected_columns - len(data_parts)))
                if debug:
                    print(f"Line {line_number}: Incomplete data. Expected {expected_columns} columns, got {len(data_parts)}. Padded: {data_parts}")

            # Extract common header value
            common_value = data_parts[common_header_idx].strip() if isinstance(data_parts[common_header_idx], str) else np.nan

            # Assign data to each experiment
            # Use experiment_headers[i] for the experiment-specific column name
            for i, exp_name in enumerate(experiments.keys()):
                data_index = i + 1  # +1 for common_header
                if data_index >= len(data_parts):
                    if debug:
                        print(f"Line {line_number}: Missing value for '{exp_name}'. Skipping.")
                    continue

                raw_value = data_parts[data_index]
                exp_value = raw_value.strip() if isinstance(raw_value, str) else np.nan

                # Build data dict
                data_dict = {common_header: common_value}
                # Add column only if exp_value is not empty
                if exp_value and exp_value != '':
                    data_dict[experiment_headers[i]] = exp_value

                experiments[exp_name]['data'].append(data_dict)

    # Add ASSAY to metadata
    metadata['ASSAY'] = assay

    # Consolidate experiments into DataFrames
    consolidated_experiments = {}
    for exp_name, exp_info in experiments.items():
        df = pd.DataFrame(exp_info['data'])
        if not df.empty:
            consolidated_experiments[exp_name] = df

    # Output the final experiments dictionary structure
    print("Final experiments structure:")
    pprint(consolidated_experiments)

    return metadata, consolidated_experiments
