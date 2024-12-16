# main.py

import argparse
from pathlib import Path
from file_reader import read_file
import subprocess
import sys
import pandas as pd
import numpy as np

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

    Generate CSVs with the following structure:
    - First columns: experiment data (as they are).
    - One empty column.
    - One column for metadata property names.
    - One column for corresponding metadata values.
    """
    if debug:
        print("Running process logic...")

    data_dir = Path("data")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        print(f"Output directory '{output_dir}' is ready.")

    all_consolidated_experiments = {}

    for file_path in data_dir.rglob("*.txt"):
        if debug:
            print(f"\nProcessing file: {file_path}")

        # Extract ASSAY from the parent directory name
        assay = file_path.parent.name
        if debug:
            print(f"ASSAY extracted: {assay}")

        metadata, data = read_file(file_path, assay=assay, debug=debug)

        if metadata and debug:
            print("\nGlobal Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")

        if data is not None and isinstance(data, dict):
            for exp_name, exp_info in data.items():
                new_df = exp_info['data']
                new_metadata = exp_info['metadata']

                if exp_name in all_consolidated_experiments:
                    # Concatenate horizontally if the experiment already exists
                    existing_info = all_consolidated_experiments[exp_name]
                    existing_df = existing_info['data']
                    concatenated_df = pd.concat([existing_df, new_df], axis=1)

                    # Merge metadata
                    merged_metadata = {**existing_info['metadata'], **new_metadata}
                    all_consolidated_experiments[exp_name]['data'] = concatenated_df
                    all_consolidated_experiments[exp_name]['metadata'] = merged_metadata
                else:
                    # Add new experiment entry
                    combined_metadata = {**metadata, **new_metadata}
                    all_consolidated_experiments[exp_name] = {
                        'metadata': combined_metadata,
                        'data': new_df
                    }

    # After processing all files, save all experiments
    for exp_name, exp_info in all_consolidated_experiments.items():
        df = exp_info['data']
        experiment_metadata = exp_info['metadata']

        # Prepare metadata DataFrame
        metadata_rows = {
            "Metadata Property": list(experiment_metadata.keys()),
            "Metadata Value": list(experiment_metadata.values())
        }
        metadata_df = pd.DataFrame(metadata_rows)

        # Add an empty column to the main DataFrame
        df[""] = ""  # Empty column as a placeholder

        # Concatenate the metadata to the right side of the DataFrame
        final_df = pd.concat([df, metadata_df], axis=1)

        # Save the final DataFrame
        assay = experiment_metadata.get("ASSAY", "Unknown_Assay")
        assay_output_dir = output_dir / assay
        assay_output_dir.mkdir(parents=True, exist_ok=True)

        safe_exp_name = exp_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        output_file = assay_output_dir / f"{safe_exp_name}.csv"
        final_df.to_csv(output_file, index=False)

        if debug:
            print(f"Saved final DataFrame for '{exp_name}' to {output_file}")

        # Call analyzer.py with the path to the saved CSV
        try:
            if debug:
                print(f"Calling analyzer.py for {output_file}")
            subprocess.run([sys.executable, "analyzer.py", str(output_file)], check=True)
            if debug:
                print(f"Analyzer completed for {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Analyzer failed for {output_file} with error: {e}")
            
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