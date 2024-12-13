# main.py

import argparse
from pathlib import Path
from file_reader import read_file
import subprocess
import sys
import pandas as pd
import numpy as np
from pprint import pprint

def prepare(debug=False):
    """
    Prepare the data directory structure.
    """
    if debug:
        print("Running prepare logic...")
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for known assay
    assay = "Temperature Sweep"
    assay_dir = data_dir / assay
    assay_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        print(f"Directory '{assay_dir}' is ready.")

def process(debug=False):
    """
    Process multiple data files and unify experiments if needed.

    If two files have the same experiment (#FILE:) but different sets of columns,
    they will be merged into a single DataFrame. The merge is done on the common header (e.g., Temp./°C).
    """
    if debug:
        print("Running process logic...")

    data_dir = Path("data")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        print(f"Output directory '{output_dir}' is ready.")

    # Global structure to hold all experiments from all files:
    # all_consolidated_experiments = {
    #    exp_name: {
    #       'metadata': {...},         # The most recent or merged metadata if needed
    #       'data': DataFrame(...)     # Merged DataFrame across files
    #    },
    #    ...
    # }
    all_consolidated_experiments = {}

    for file_path in data_dir.rglob("*.txt"):
        if debug:
            print(f"\nProcessing file: {file_path}")

        # Extract ASSAY from the parent directory name
        assay = file_path.parent.name
        if debug:
            print(f"ASSAY extracted: {assay}")

        metadata, data = read_file(file_path, assay=assay, debug=debug)

        # metadata is global to the file
        if metadata and debug:
            print("\nGlobal Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")

        if data is not None and isinstance(data, dict):
            # data = {
            #   'ExperimentName': {
            #       'metadata': {...},
            #       'data': DataFrame
            #   }
            # }
            for exp_name, exp_info in data.items():
                new_df = exp_info['data']
                new_metadata = exp_info['metadata']

                if exp_name in all_consolidated_experiments:
                    # Merge with existing DataFrame
                    existing_info = all_consolidated_experiments[exp_name]
                    existing_df = existing_info['data']
                    # Identify common header - assume the first column is common or known
                    # If we have a known common header like 'Temp./°C', identify it in both DataFrames
                    # Here we assume both have a common header. Let's guess it's one of the DataFrame's first column
                    # Ideally, we know the common_header from the data
                    common_headers = set(existing_df.columns).intersection(set(new_df.columns))
                    # Find a common header like 'Temp' or 'Time'
                    # If known, we can guess:
                    possible_common_headers = [c for c in common_headers if 'temp' in c.lower() or 'time' in c.lower()]
                    if not possible_common_headers:
                        # fallback to first column of existing or new_df
                        common_col = existing_df.columns[0]
                    else:
                        common_col = possible_common_headers[0]

                    # Perform outer merge on the common column
                    merged_df = pd.merge(existing_df, new_df, on=common_col, how='outer')
                    # Update metadata by merging or just keeping the old and update keys from new_metadata
                    merged_metadata = {**existing_info['metadata'], **new_metadata}
                    all_consolidated_experiments[exp_name]['data'] = merged_df
                    all_consolidated_experiments[exp_name]['metadata'] = merged_metadata
                    if debug:
                        print(f"Merged experiment '{exp_name}' from file '{file_path.name}'.")
                else:
                    # Add new experiment entry
                    all_consolidated_experiments[exp_name] = {
                        'metadata': {**metadata, **new_metadata},  # Combine global file-level and experiment-level metadata
                        'data': new_df
                    }
                    if debug:
                        print(f"Added new experiment '{exp_name}' from file '{file_path.name}'.")
        else:
            if debug:
                print("No data extracted or data not in dictionary format.")

    # After processing all files, save all experiments
    for exp_name, exp_info in all_consolidated_experiments.items():
        assay = exp_info['metadata'].get('ASSAY', 'Unknown_Assay')
        assay_output_dir = output_dir / assay
        assay_output_dir.mkdir(parents=True, exist_ok=True)

        df = exp_info['data']
        # Integrate all metadata into the DataFrame as columns if you wish
        # For demonstration, just integrate metadata keys
        for k, v in exp_info['metadata'].items():
            df[k] = v

        safe_exp_name = exp_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        output_file = assay_output_dir / f"{safe_exp_name}.csv"
        df.to_csv(output_file, index=False)

        if debug:
            print(f"Saved consolidated DataFrame for '{exp_name}' to {output_file}")

    if debug:
        print("\nFinal consolidated experiments structure:")
        pprint(all_consolidated_experiments)

def main():
    """
    Parse command-line arguments and execute the corresponding functions.
    """
    parser = argparse.ArgumentParser(description="Piep-DMA Project")
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--process", action="store_true", help="Process the data")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.prepare:
        prepare(debug=args.debug)
    elif args.process:
        process(debug=args.debug)
    else:
        print("Please specify either --prepare or --process.")

if __name__ == "__main__":
    main()
