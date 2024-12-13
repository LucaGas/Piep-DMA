# main.py

import argparse
from pathlib import Path
from file_reader import read_file
import pandas as pd
import subprocess
import sys

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
    Process the data files: extract metadata and data, integrate metadata into DataFrames,
    save them as CSV files within ASSAY directories, and pass them to analyzer.py.
    """
    if debug:
        print("Running process logic...")
    data_dir = Path("data")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        print(f"Output directory '{output_dir}' is ready.")

    for file_path in data_dir.rglob("*.txt"):
        if debug:
            print(f"\nProcessing file: {file_path}")

        # Extract ASSAY from the parent directory name
        assay = file_path.parent.name
        if debug:
            print(f"ASSAY extracted: {assay}")

        metadata, data = read_file(file_path, assay=assay, debug=debug)

        if metadata:
            if debug:
                print("\nMetadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
        else:
            if debug:
                print("No metadata extracted.")

        if data is not None:
            if isinstance(data, dict):
                # Create ASSAY-specific output directory
                assay_output_dir = output_dir / assay
                assay_output_dir.mkdir(parents=True, exist_ok=True)
                if debug:
                    print(f"Assay output directory '{assay_output_dir}' is ready.")

                for exp_name, df in data.items():
                    if not df.empty:
                        # Integrate metadata into the DataFrame
                        for key, value in metadata.items():
                            df[key] = value
                        if debug:
                            print(f"\nSaving DataFrame for experiment: {exp_name} with metadata.")

                        # Define the output file path
                        safe_exp_name = exp_name.replace(' ', '_').replace('/', '_')  # Replace spaces and slashes for file naming
                        output_file = assay_output_dir / f"{safe_exp_name}.csv"
                        df.to_csv(output_file, index=False)
                        if debug:
                            print(f"Saved DataFrame to {output_file}")

                        # Call analyzer.py with the path to the saved CSV
                        try:
                            if debug:
                                print(f"Calling analyzer.py for {output_file}")
                            subprocess.run([sys.executable, "analyzer.py", str(output_file)], check=True)
                            if debug:
                                print(f"Analyzer completed for {output_file}")
                        except subprocess.CalledProcessError as e:
                            print(f"Analyzer failed for {output_file} with error: {e}")
            else:
                if debug:
                    print("Data is not in expected format.")
        else:
            if debug:
                print("No data extracted.")

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
