# main.py

import argparse
from pathlib import Path
from file_reader import read_file
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
    save them as CSV files within ASSAY directories, and (optionally) pass them to analyzer.py.
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

        # metadata is global file-level metadata (if any)
        if metadata and debug:
            print("\nGlobal Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")

        if data is not None:
            # data should be a dictionary:
            # data = {
            #   'ExperimentName': {
            #       'metadata': {...},  # Experiment-level metadata
            #       'data': DataFrame  # Experiment data
            #   },
            #   ...
            # }

            if isinstance(data, dict):
                # Create ASSAY-specific output directory
                assay_output_dir = output_dir / assay
                assay_output_dir.mkdir(parents=True, exist_ok=True)
                if debug:
                    print(f"Assay output directory '{assay_output_dir}' is ready.")

                # Iterate over each experiment
                for exp_name, exp_info in data.items():
                    df = exp_info.get('data', None)
                    exp_metadata = exp_info.get('metadata', {})

                    if df is not None and not df.empty:
                        # Integrate per-experiment metadata into the DataFrame
                        for key, value in exp_metadata.items():
                            df[key] = value

                        # Also, if you want to integrate global metadata into DataFrame:
                        for k, v in metadata.items():
                            df[k] = v

                        if debug:
                            print(f"\nSaving DataFrame for experiment: {exp_name} with metadata.")

                        # Define the output file path
                        safe_exp_name = exp_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                        output_file = assay_output_dir / f"{safe_exp_name}.csv"
                        df.to_csv(output_file, index=False)
                        if debug:
                            print(f"Saved DataFrame to {output_file}")

                        # (Optional) Call analyzer.py with the path to the saved CSV
                        try:
                            if debug:
                                print(f"Calling analyzer.py for {output_file}")
                            # Uncomment to run analyzer if needed
                            # subprocess.run([sys.executable, "analyzer.py", str(output_file)], check=True)
                            if debug:
                                print(f"Analyzer completed for {output_file}")
                        except subprocess.CalledProcessError as e:
                            print(f"Analyzer failed for {output_file} with error: {e}")
                    else:
                        if debug:
                            print(f"No data for experiment '{exp_name}' or DataFrame is empty.")
            else:
                if debug:
                    print("Data is not in expected dictionary format.")
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
