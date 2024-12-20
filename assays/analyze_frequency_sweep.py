# assays/analyze_frequency_sweep.py

import sys
import os
import re
import logging
from pathlib import Path
import shutil
import time
import traceback
# Add parent directory to sys.path to access utils.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import setup_logging  # Ensure utils.py is accessible


def extract_frequency_from_header(header_line):
    """
    Extract the frequency value from the header line.

    Parameters:
        header_line (str): The header line from the CSV file.

    Returns:
        str or None: The extracted frequency formatted as 'N_Hz', or None if not found.
    """
    # Regex pattern to find (N Hz), capturing N
    pattern = r'\((\d+\.\d+)\s*Hz\)'
    match = re.search(pattern, header_line)
    if match:
        frequency = match.group(1)  # Extracted numeric value
        frequency_str = f"{frequency}_Hz"
        return frequency_str
    return None

def is_file_locked(file_path):
    """
    Check if a file is locked by attempting to open it in exclusive mode.

    Parameters:
        file_path (Path): The path to the file.

    Returns:
        bool: True if the file is locked, False otherwise.
    """
    try:
        with file_path.open('a'):
            pass
        return False
    except IOError:
        return True

def copy_csv_file(file_path, frequency_str, logger):
    """
    Copy the CSV file by appending the frequency string to the filename.

    Parameters:
        file_path (Path): The path to the CSV file.
        frequency_str (str): The frequency string to insert into the new filename.
        logger (Logger): The logger instance.

    Returns:
        bool: True if copying was successful, False otherwise.
    """
    try:
        # Original filename parts
        original_name = file_path.stem  # e.g., ExpDat_2_-_Varrimento_frequencia_T2_PVC_TD_8
        suffix = file_path.suffix      # e.g., .csv

        # Define the new filename by appending the frequency string
        new_name = f"{original_name}_{frequency_str}"
        new_file_path = file_path.with_name(new_name + suffix)

        # Check if the new file already exists to prevent overwriting
        if new_file_path.exists():
            logger.warning(f"The file '{new_file_path.name}' already exists. Skipping copy.")
            return False

        # Perform the copy operation
        shutil.copy2(file_path, new_file_path)
        logger.info(f"Copied '{file_path.name}' to '{new_file_path.name}'.")
        return True
    except Exception as e:
        logger.error(f"Failed to copy '{file_path.name}': {e}")
        logger.debug(traceback.format_exc())  # Log the full stack trace
        return False

def copy_csv_file_with_retry(file_path, frequency_str, logger, retries=3, delay=2):
    """
    Attempt to copy the CSV file with retries if it's locked.

    Parameters:
        file_path (Path): The path to the CSV file.
        frequency_str (str): The frequency string to insert into the new filename.
        logger (Logger): The logger instance.
        retries (int): Number of retry attempts.
        delay (int): Delay in seconds between retries.

    Returns:
        bool: True if copying was successful, False otherwise.
    """
    for attempt in range(1, retries + 1):
        try:
            return copy_csv_file(file_path, frequency_str, logger)
        except Exception as e:
            logger.warning(f"Attempt {attempt} to copy '{file_path.name}' failed: {e}")
            if attempt < retries:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All {retries} attempts to copy '{file_path.name}' have failed.")
                return False

def analyze_frequency_sweep(output_dir):
    """
    Analyze frequency sweep CSV files by copying them with new names based on frequency information in headers.

    Parameters:
        output_dir (Path): The directory containing the CSV files to process.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting frequency sweep analysis in directory: '{output_dir}'")

    # Ensure the output directory exists
    if not output_dir.exists() or not output_dir.is_dir():
        logger.error(f"The specified output directory does not exist or is not a directory: '{output_dir}'")
        return

    # Iterate over all CSV files in the output directory
    for csv_file in output_dir.glob('*.csv'):
        logger.debug(f"Processing file: '{csv_file.name}'")
        try:
            with csv_file.open('r', encoding='utf-8') as f:
                header_line = f.readline().strip()
                logger.debug(f"Header Line: '{header_line}'")
                frequency_str = extract_frequency_from_header(header_line)

                if frequency_str:
                    logger.debug(f"Extracted frequency: '{frequency_str}'")
                    # Check if the file is locked
                    if is_file_locked(csv_file):
                        logger.warning(f"File '{csv_file.name}' is currently locked by another process. Skipping copying.")
                        continue
                    # Attempt to copy with retry logic
                    copy_success = copy_csv_file_with_retry(csv_file, frequency_str, logger)
                    if not copy_success:
                        logger.error(f"Failed to copy file '{csv_file.name}'.")
                else:
                    logger.warning(f"No frequency pattern '(N Hz)' found in header of '{csv_file.name}'. Skipping copying.")

        except Exception as e:
            logger.error(f"Error processing file '{csv_file.name}': {e}")
            logger.debug(traceback.format_exc())  # Log the full stack trace

    logger.info("Frequency sweep analysis completed.")

def main():
    """
    Entry point for the frequency sweep analysis script.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Frequency Sweep ASSAY by Copying CSV Files Based on Header Frequencies.")
    parser.add_argument("output_directory", help="Path to the output directory containing CSV files.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode with detailed logs.")

    args = parser.parse_args()

    # Set up logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level=log_level, log_file='analyze_frequency_sweep.log')  # Customize log_file as needed

    # Acquire a logger for this script
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_directory)

    analyze_frequency_sweep(output_dir)

if __name__ == "__main__":
    main()
