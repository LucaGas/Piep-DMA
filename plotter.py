# plotter.py

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import re
import logging
from pathlib import Path
from utils import setup_logging, map_column_name, sanitize_column_name  # Import necessary utilities
import argparse

def generate_bar_graph(metric, samples_data, output_dir):
    """
    Generate and save a bar graph for a specific metric across all samples.

    Parameters:
        metric (str): The name of the metric (e.g., 'Onset of Eprime').
        samples_data (dict): A dictionary with SAMPLE names as keys and dictionaries containing 'average' and 'std' as values.
        output_dir (Path): Directory to save the graph.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    if not samples_data:
        logger.warning(f"No data provided for metric '{metric}'. Skipping graph generation.")
        return

    samples = list(samples_data.keys())
    averages = [samples_data[sample]['average'] for sample in samples]
    std_devs = [samples_data[sample]['std'] for sample in samples]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(samples, averages, yerr=std_devs, capsize=5, color='skyblue', edgecolor='black')
    plt.xlabel('Sample', fontsize=14, fontweight='bold')
    plt.ylabel(metric, fontsize=14, fontweight='bold')
    plt.title(f'{metric} by Sample', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Annotate bars with their average values
    for bar, avg in zip(bars, averages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{avg:.2f}', ha='center', va='bottom', fontsize=12)

    # Save the plot
    graph_filename = f"{metric.replace(' ', '_')}_Bar_Graph.png"
    graph_path = output_dir / graph_filename
    plt.savefig(graph_path)
    plt.close()

    logger.info(f"Saved bar graph for '{metric}' to '{graph_path}'.")


def sanitize_dataframe(df):
    """
    Clean the DataFrame by handling NaN values and ensuring columns are usable.

    Parameters:
    - df (DataFrame): Input DataFrame to sanitize.

    Returns:
    - Sanitized DataFrame.
    """
    logger = logging.getLogger(__name__)
    # Drop entirely empty columns
    df = df.dropna(axis=1, how='all')
    logger.debug("Dropped empty columns.")

    # Fill remaining NaN values with placeholders
    df = df.fillna("")
    logger.debug("Filled NaN values with empty strings.")

    # Convert numeric columns where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
            logger.debug(f"Converted column '{col}' to numeric.")
        except ValueError:
            logger.debug(f"Could not convert column '{col}' to numeric. Keeping it as-is.")

    logger.debug("Sanitized DataFrame Head:")
    logger.debug(df.head())

    return df


def identify_x_axis(df):
    """
    Identify the primary X-axis column, preferring 'Temp' over 'Time'.

    Parameters:
    - df (DataFrame): The DataFrame to search.

    Returns:
    - str: The name of the X-axis column.
    """
    logger = logging.getLogger(__name__)
    # Search for 'Temp' first
    for col in df.columns:
        if re.search(r'\btemp\b', col, re.IGNORECASE):
            logger.info(f"Identified '{col}' as the X-axis column based on 'Temp'.")
            return col

    # If no 'Temp' column, search for 'Time'
    for col in df.columns:
        if re.search(r'\btime\b', col, re.IGNORECASE):
            logger.info(f"Identified '{col}' as the X-axis column based on 'Time'.")
            return col

    # If neither is found, return None
    logger.warning("No X-axis column found containing 'Temp' or 'Time'.")
    return None


def plot_temperature_sweep(csv_file):
    """
    Plot data from a CSV file where Y-axis values are plotted against the primary X-axis column ('Temp' or 'Time').
    Highlights relevant analysis results (Onset of Eprime and peaks) as markers.

    Parameters:
    - csv_file (str): Path to the input CSV file.

    Returns:
    """
    logger = logging.getLogger(__name__)
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded CSV file: {csv_file}")

        # Sanitize column names
        df.columns = [sanitize_column_name(col) for col in df.columns]
        logger.debug(f"Plotter: Sanitized data columns: {list(df.columns)}")

        # Sanitize the DataFrame
        df = sanitize_dataframe(df)

        # Identify the primary X-axis column ('Temp' or 'Time')
        x_column = identify_x_axis(df)
        if not x_column:
            logger.error("Plotter: No suitable X-axis column ('Temp' or 'Time') found. Exiting.")
            return

        # Extract analysis results from the metadata
        analysis_results = {}
        if "Metadata Property" in df.columns and "Metadata Value" in df.columns:
            metadata_property_index = df.columns.get_loc("Metadata Property")  # Find the column index
            value_column_index = df.columns.get_loc("Metadata Value")          # Find the corresponding value column

            metadata_rows = df[df["Metadata Property"].notna()]
            for _, row in metadata_rows.iterrows():
                key = str(row["Metadata Property"]).strip()
                value = row.iloc[value_column_index]
                analysis_results[key] = value

        # Output the extracted values of Onset and Peaks
        logger.info("\n--- Extracted Analysis Results ---")
        for key, value in analysis_results.items():
            if "Onset" in key or "Peak" in key:
                logger.info(f"{key}: {value}")
        logger.info("----------------------------------")
        logger.debug(f"Plotter: Extracted analysis results: {analysis_results}")

        # Specifically check for 'Peak of tand' and 'Onset of Eprime'
        if 'Peak of tand' in analysis_results:
            logger.info(f"'Peak of tand' found with value: {analysis_results['Peak of tand']}")
        else:
            logger.warning("'Peak of tand' not found in analysis results.")

        if 'Onset of Eprime' in analysis_results:
            logger.info(f"'Onset of Eprime' found with value: {analysis_results['Onset of Eprime']}")
        else:
            logger.warning("'Onset of Eprime' not found in analysis results.")

        # Identify Y-axis columns (all numeric columns excluding the X-axis)
        y_columns = df.select_dtypes(include=[float, int]).columns.tolist()
        if x_column in y_columns:
            y_columns.remove(x_column)

        logger.info(f"Plotter: Identified X-axis column: '{x_column}'")
        logger.info(f"Plotter: Identified Y-axis columns: {y_columns}")

        if not y_columns:
            logger.warning("Plotter: No Y-axis columns found to plot against the X-axis.")
            return

        # Define mappings between Y-axis columns and their corresponding analysis keys
        column_to_analysis_keys = {
            "Eprime": ["Onset of Eprime"],
            "Edoubleprime": ["Peak of Edoubleprime"],
            "tan_delta": ["Peak of tand"],
            # Add any additional mappings if necessary
        }

        logger.info("Starting to process Y-axis columns...")
        # Iterate over Y-axis columns and plot against the X-axis column
        for y_column in y_columns:
            logger.debug(f"Plotter: Preparing to plot '{y_column}' against '{x_column}'")

            # Ensure valid numeric data
            x_data = pd.to_numeric(df[x_column], errors='coerce')
            y_data = pd.to_numeric(df[y_column], errors='coerce')

            # Drop rows with NaN values in either X or Y
            valid_data = pd.DataFrame({x_column: x_data, y_column: y_data}).dropna()

            # Debug valid data
            if valid_data.empty:
                logger.debug(f"Plotter: No valid data for '{x_column}' and '{y_column}'. Skipping.")
                continue

            # Plot the data
            plt.figure(figsize=(8, 6))
            plt.plot(valid_data[x_column], valid_data[y_column], marker='o', linestyle='-', label=y_column)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f"{y_column} vs {x_column}")
            plt.grid(True)

            # Map the y_column name to desired label
            mapped_y_column = map_column_name(y_column)

            # Sanitize the mapped y_column for logging or other purposes (if needed)
            sanitized_y_column_log = re.sub(r'[^\w\-_. ]', '_', mapped_y_column)
            logger.debug(f"sanitized_y_column_log: {sanitized_y_column_log}")

            # Get relevant analysis keys for the current y_column
            # Extract base name (e.g., "Edoubleprime" from "Edoubleprime (1.000 Hz) MPa")
            base_y_column = re.split(r'\s*\(', mapped_y_column)[0].strip()
            relevant_analysis_keys = column_to_analysis_keys.get(base_y_column, [])

            logger.debug(f"Relevant analysis keys for '{base_y_column}': {relevant_analysis_keys}")

            # Highlight relevant analysis points
            for analysis_key in relevant_analysis_keys:
                if analysis_key in analysis_results:
                    analysis_value = str(analysis_results[analysis_key]).strip()

                    if not analysis_value or analysis_value.lower() == "nan":
                        logger.warning(f"Plotter: Skipping invalid metadata value for '{analysis_key}': {analysis_value}")
                        continue

                    logger.info(f"Plotter: Highlighting '{analysis_key}' in column '{y_column}'")

                    # Process based on the type of analysis key
                    if "Onset of Eprime" in analysis_key:
                        try:
                            # Extract X from the analysis value, expected format: a float
                            tonset_x = float(analysis_value)

                            # Retrieve corresponding Y from the data
                            # Assuming Eprime column corresponds to Y-axis data
                            tonset_y_series = df[df[x_column] == tonset_x][y_column]
                            if not tonset_y_series.empty:
                                tonset_y = tonset_y_series.iloc[0]
                            else:
                                tonset_y = None

                            if tonset_y is not None:
                                # Plot Onset of Eprime point
                                plt.scatter(tonset_x, tonset_y, color='red', label='Onset of Eprime')
                                plt.axvline(x=tonset_x, color='red', linestyle='--', label=f"Onset Line: X={tonset_x:.2f}")
                                plt.annotate(
                                    f"Onset\n({tonset_x:.2f}, {tonset_y:.2f})",
                                    xy=(tonset_x, tonset_y),
                                    xytext=(tonset_x, tonset_y + 0.05 * (plt.ylim()[1] - plt.ylim()[0])),
                                    arrowprops=dict(facecolor='red', arrowstyle="->"),
                                    fontsize=8,
                                    ha='center',
                                )
                                # Adjust y-axis limits to accommodate annotations
                                y_min, y_max = plt.ylim()  # Get current y-axis limits
                                new_y_max = y_max + 0.1 * (y_max - y_min)  # Add 10% extra space to the top
                                plt.ylim(y_min, new_y_max)  # Set the adjusted y-axis limits
                            else:
                                logger.error(f"Plotter: Could not find Y-value for Onset of Eprime at X={tonset_x}")
                        except ValueError as e:
                            logger.error(f"Plotter: Failed to highlight Onset of Eprime '{analysis_value}' - {e}")

                    elif "Peak of tand" in analysis_key or "Peak of Edoubleprime" in analysis_key:
                        try:
                            # Extract X and Y from the analysis value, expected format: "X: 149.59108, Y: 156.37594"
                            peak_matches = re.findall(r"X:\s*([\d\.]+)", analysis_value)
                            peak_y_matches = re.findall(r"Y:\s*([\d\.]+)", analysis_value)
                            if peak_matches and peak_y_matches:
                                peak_x = float(peak_matches[0])
                                peak_y = float(peak_y_matches[0])

                                # Plot Peak point
                                plt.scatter(peak_x, peak_y, color='green', label='Peak')
                                plt.axvline(x=peak_x, color='green', linestyle='--', label=f"Peak Line: X={peak_x:.2f}")
                                plt.annotate(
                                    f"Peak\n({peak_x:.2f}, {peak_y:.5f})",
                                    xy=(peak_x, peak_y),
                                    xytext=(peak_x, peak_y + 0.05 * (plt.ylim()[1] - plt.ylim()[0])),
                                    arrowprops=dict(facecolor='green', arrowstyle="->"),
                                    fontsize=8,
                                    ha='center',
                                )
                                # Adjust y-axis limits to accommodate annotations
                                y_min, y_max = plt.ylim()  # Get current y-axis limits
                                new_y_max = y_max + 0.1 * (y_max - y_min)  # Add 10% extra space to the top
                                plt.ylim(y_min, new_y_max)  # Set the adjusted y-axis limits
                            else:
                                logger.error(f"Plotter: Invalid format for {analysis_key} value: '{analysis_value}'")
                        except ValueError as e:
                            logger.error(f"Plotter: Failed to highlight {analysis_key} '{analysis_value}' - {e}")

            # Add legend for highlighted points
            plt.legend()

            # Sanitize the mapped y_column for filenames
            sanitized_y_column = re.sub(r'[^\w\-_. ]', '_', mapped_y_column)  # Remove problematic characters

            # Save the plot
            output_file = f"{os.path.splitext(csv_file)[0]}_{sanitized_y_column}.png"
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()

            logger.info(f"Plotter: Plot saved to '{output_file}'")

    except FileNotFoundError:
        logger.error(f"Plotter: The file '{csv_file}' does not exist.")
    except pd.errors.EmptyDataError:
        logger.error(f"Plotter: The file '{csv_file}' is empty.")
    except Exception as e:
        logger.error(f"Plotter: An error occurred while processing '{csv_file}': {e}")


def main():
    """
    Main function to handle standalone execution of the plotting script.
    """
    parser = argparse.ArgumentParser(description="Plotter Script for Piep-DMA Project")
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode with detailed logs.")

    args = parser.parse_args()

    # Set up logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level=log_level, log_file='app.log')  # Customize log_file as needed

    # Acquire a logger for plotter.py
    logger = logging.getLogger(__name__)

    csv_file = args.csv_file
    logger.info(f"Plotter: Processing '{csv_file}'")
    # Call the plotting function
    plot_temperature_sweep(csv_file)


if __name__ == "__main__":
    main()
