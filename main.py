# main.py

import argparse
from pathlib import Path
from file_reader import read_file
from utils import setup_logging, save_results, get_aligned_data, map_column_name
import subprocess
import sys
import pandas as pd
import numpy as np
import logging

def prepare(logger):
    """
    Prepare the data directory structure.

    Parameters:
        logger (Logger): The logger object.
    """
    logger.info("Running prepare logic...")
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created/verified data directory: {data_dir}")

    # Create subdirectory for known assay
    assay = "Temperature Sweep"
    assay_dir = data_dir / assay
    assay_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory '{assay_dir}' is ready.")
    logger.debug(f"Created/verified assay directory: {assay_dir}")


def process(logger):
    """
    Process multiple data files and unify experiments if needed.

    Generate CSVs with the following structure:
    - First columns: experiment data (as they are).
    - One empty column.
    - One column for metadata property names.
    - One column for corresponding metadata values.

    Parameters:
        logger (Logger): The logger object.
    """
    logger.info("Running process logic...")

    data_dir = Path("data")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created/verified output directory: {output_dir}")

    all_consolidated_experiments = {}

    for file_path in data_dir.rglob("*.txt"):
        logger.info(f"\nProcessing file: {file_path}")

        # Extract ASSAY from the parent directory name
        assay = file_path.parent.name
        logger.debug(f"ASSAY extracted: {assay}")

        metadata, data = read_file(file_path, assay=assay)

        if metadata:
            logger.debug(f"Global Metadata for '{file_path}': {metadata}")
        else:
            logger.warning(f"No metadata extracted for '{file_path}'.")

        if data is not None and isinstance(data, dict):
            for exp_name, exp_info in data.items():
                new_df = exp_info['data']
                new_metadata = exp_info['metadata']

                if exp_name in all_consolidated_experiments:
                    # Concatenate horizontally if the experiment already exists
                    existing_info = all_consolidated_experiments[exp_name]
                    existing_df = existing_info['data']
                    concatenated_df = pd.concat([existing_df, new_df], axis=1)
                    logger.debug(f"Concatenated data for existing experiment '{exp_name}'.")

                    # Merge metadata
                    merged_metadata = {**existing_info['metadata'], **new_metadata}
                    all_consolidated_experiments[exp_name]['data'] = concatenated_df
                    all_consolidated_experiments[exp_name]['metadata'] = merged_metadata
                    logger.debug(f"Merged metadata for experiment '{exp_name}'.")
                else:
                    # Add new experiment entry
                    combined_metadata = {**metadata, **new_metadata}
                    all_consolidated_experiments[exp_name] = {
                        'metadata': combined_metadata,
                        'data': new_df
                    }
                    logger.info(f"Added new experiment '{exp_name}'.")
        else:
            logger.warning(f"No data extracted from '{file_path}'.")

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
        logger.debug(f"Prepared metadata DataFrame for experiment '{exp_name}'.")

        # Add an empty column to the main DataFrame
        df[""] = ""  # Empty column as a placeholder
        logger.debug(f"Added empty column to DataFrame for experiment '{exp_name}'.")

        # Concatenate the metadata to the right side of the DataFrame
        final_df = pd.concat([df, metadata_df], axis=1)
        logger.debug(f"Concatenated metadata to DataFrame for experiment '{exp_name}'.")

        # Save the final DataFrame
        assay = experiment_metadata.get("ASSAY", "Unknown_Assay")
        assay_output_dir = output_dir / assay
        assay_output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured assay-specific output directory: {assay_output_dir}")

        safe_exp_name = exp_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        output_file = assay_output_dir / f"{safe_exp_name}.csv"
        final_df.to_csv(output_file, index=False)
        logger.info(f"Saved final DataFrame for '{exp_name}' to '{output_file}'.")

        # Call analyzer.py with the path to the saved CSV
        try:
            logger.info(f"Calling analyzer.py for '{output_file}'.")
            subprocess.run([sys.executable, "analyzer.py", str(output_file)], check=True)
            logger.info(f"Analyzer completed successfully for '{output_file}'.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Analyzer failed for '{output_file}' with error: {e}")
        except FileNotFoundError:
            logger.error("analyzer.py not found. Ensure it exists in the same directory.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while calling analyzer.py: {e}")


def main():
    """
    Parse command-line arguments and execute the corresponding functions.
    """
    parser = argparse.ArgumentParser(description="Piep-DMA Project")
    parser.add_argument("--prepare", action="store_true", help="Prepare the data directory structure.")
    parser.add_argument("--process", action="store_true", help="Process the data files.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode with detailed logs.")

    args = parser.parse_args()

    # Set up logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level=log_level, log_file='app.log')  # Customize log_file as needed

    # Acquire a logger for main.py
    logger = logging.getLogger(__name__)

    if args.prepare:
        prepare(logger)
    elif args.process:
        process(logger)
    else:
        logger.error("No action specified. Please use --prepare or --process.")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
