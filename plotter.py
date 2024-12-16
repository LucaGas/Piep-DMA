import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import re
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def sanitize_column_name(column_name):
    """
    Sanitize column names by replacing problematic characters.
    
    Parameters:
    - column_name (str): Original column name.
    
    Returns:
    - str: Sanitized column name.
    """
    if not isinstance(column_name, str):
        return column_name

    sanitized_name = (
        column_name
        .replace("�", "°")  # Handle encoding errors
        .strip()
    )
    return sanitized_name

def sanitize_dataframe(df, debug=False):
    """
    Clean the DataFrame by handling NaN values and ensuring columns are usable.

    Parameters:
    - df (DataFrame): Input DataFrame to sanitize.
    - debug (bool): If True, print debug information.

    Returns:
    - Sanitized DataFrame.
    """
    # Drop entirely empty columns
    df = df.dropna(axis=1, how='all')
    if debug:
        logger.debug("Dropped empty columns.")

    # Fill remaining NaN values with placeholders
    df = df.fillna("")
    if debug:
        logger.debug("Filled NaN values with empty strings.")

    # Convert numeric columns where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
            if debug:
                logger.debug(f"Converted column '{col}' to numeric.")
        except ValueError:
            if debug:
                logger.debug(f"Could not convert column '{col}' to numeric. Keeping it as-is.")

    if debug:
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

def map_column_name(y_column):
    """
    Replace specific substrings in column names with desired labels based on a predefined mapping.

    Parameters:
    - y_column (str): Original column name.

    Returns:
    - str: Modified column name with replacements.
    """
    # Define your mappings here
    replacements = {
        "E''": "Edoubleprime",
        "E'": "Eprime",
        "tan d": "tan_delta",
        "tanδ": "tan_delta",
        # Add more mappings as needed
    }
    
    for original, replacement in replacements.items():
        # Use case-insensitive replacement
        y_column = re.sub(re.escape(original), replacement, y_column, flags=re.IGNORECASE)
    
    return y_column

def plot_temperature_sweep(csv_file, debug=False):
    """
    Plot data from a CSV file where Y-axis values are plotted against the primary X-axis column ('Temp' or 'Time').
    Highlights relevant analysis results (onsets and peaks) as markers.

    Parameters:
    - csv_file (str): Path to the input CSV file.
    - debug (bool): If True, print debug information.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded CSV file: {csv_file}")

        # Sanitize column names
        df.columns = [sanitize_column_name(col) for col in df.columns]
        if debug:
            logger.info(f"Plotter: Sanitized data columns: {list(df.columns)}")

        # Sanitize the DataFrame
        df = sanitize_dataframe(df, debug=debug)

        # Identify the primary X-axis column ('Temp' or 'Time')
        x_column = identify_x_axis(df)
        if not x_column:
            logger.error("Plotter: No suitable X-axis column ('Temp' or 'Time') found. Exiting.")
            return

        # Extract analysis results from the metadata
        analysis_results = {}
        if "Metadata Property" in df.columns and "Metadata Value" in df.columns:
            metadata_property_index = df.columns.get_loc("Metadata Property")  # Find the column index
            value_column_index = df.columns.get_loc("Metadata Value")  # Find the corresponding value column

            metadata_rows = df[df["Metadata Property"].notna()]
            for _, row in metadata_rows.iterrows():
                key = str(row["Metadata Property"]).strip()
                value = row.iloc[value_column_index]
                analysis_results[key] = value

        # Output the extracted values of Onsets and Peaks
        print("\n--- Extracted Analysis Results ---")
        for key, value in analysis_results.items():
            if "Onset" in key or "Peak" in key:
                print(f"{key}: {value}")
        print("----------------------------------\n")
        if debug:
            logger.debug(f"Plotter: Extracted analysis results: {analysis_results}")

        # Specifically check for 'Peak of tand'
        if 'Peak of tand' in analysis_results:
            logger.info(f"'Peak of tand' found with value: {analysis_results['Peak of tand']}")
        else:
            logger.warning("'Peak of tand' not found in analysis results.")

        # Identify Y-axis columns (all numeric columns excluding the X-axis)
        y_columns = df.select_dtypes(include=[float, int]).columns.tolist()
        if x_column in y_columns:
            y_columns.remove(x_column)

        if debug:
            logger.info(f"Plotter: Identified X-axis column: '{x_column}'")
            logger.info(f"Plotter: Identified Y-axis columns: {y_columns}")

        if not y_columns:
            logger.warning("Plotter: No Y-axis columns found to plot against the X-axis.")
            return

        # Define mappings between Y-axis columns and their corresponding analysis keys
        column_to_analysis_keys = {
            "Eprime (1.000 Hz) MPa": ["Onset of Eprime"],
            "Edoubleprime (1.000 Hz) MPa": ["Peak of Edoubleprime"],
            "tan_delta(1.000 Hz)": ["Peak of tand"],
            # Add any additional mappings if necessary
        }

        print("Starting to process Y-axis columns...")
        # Iterate over Y-axis columns and plot against the X-axis column
        for y_column in y_columns:
            if debug:
                logger.info(f"Plotter: Preparing to plot '{y_column}' against '{x_column}'")

            # Ensure valid numeric data
            x_data = pd.to_numeric(df[x_column], errors='coerce')
            y_data = pd.to_numeric(df[y_column], errors='coerce')

            # Drop rows with NaN values in either X or Y
            valid_data = pd.DataFrame({x_column: x_data, y_column: y_data}).dropna()

            # Debug valid data
            if valid_data.empty:
                if debug:
                    logger.debug(f"Plotter: No valid data for '{x_column}' and '{y_column}'. Skipping.")
                continue

            # Plot the data
            plt.figure(figsize=(8, 6))
            plt.plot(valid_data[x_column], valid_data[y_column], marker='o', linestyle='-')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f"{y_column} vs {x_column}")
            plt.grid(True)

            # Map the y_column name to desired label
            mapped_y_column = map_column_name(y_column)
            # print("Starting to process Y-axis columns...")  # Duplicate print statement removed

            # Sanitize the mapped y_column for logging or other purposes (if needed)
            sanitized_y_column_log = re.sub(r'[^\w\-_. ]', '_', mapped_y_column)

            # Get relevant analysis keys for the current y_column
            relevant_analysis_keys = column_to_analysis_keys.get(mapped_y_column, [])

            if debug:
                logger.debug(f"Relevant analysis keys for '{mapped_y_column}': {relevant_analysis_keys}")

            # To avoid duplicate legends, use a set to track added labels
            added_legends = set()

            # Highlight relevant analysis points
            for analysis_key in relevant_analysis_keys:
                if analysis_key in analysis_results:
                    analysis_value = str(analysis_results[analysis_key]).strip()

                    if not analysis_value or analysis_value.lower() == "nan":
                        logger.warning(f"Skipping invalid metadata value for '{analysis_key}': {analysis_value}")
                        continue

                    if debug:
                        logger.info(f"Plotter: Highlighting '{analysis_key}' in column '{y_column}'")

                    # Process based on the type of analysis key
                    if "Onset" in analysis_key:
                        try:
                            onset_x = float(analysis_value)
                            onset_y_index = (x_data - onset_x).abs().argsort()[0]
                            onset_y = y_data.iloc[onset_y_index]

                            plt.scatter(onset_x, onset_y, color='red', label=f"Onset: X={onset_x:.2f}")
                            y_min, y_max = plt.ylim()
                            plt.annotate(
                                f"Onset\n({onset_x:.2f}, {onset_y:.2f})",
                                xy=(onset_x, onset_y),
                                xytext=(onset_x, onset_y + 0.05 * (y_max - y_min)),
                                arrowprops=dict(facecolor='red', arrowstyle="->"),
                                fontsize=8,
                                ha='center',
                            )
                            plt.axvline(x=onset_x, color='red', linestyle='--', label=f"Onset Line: X={onset_x:.2f}")
                            # Dynamically adjust the y-axis limits to accommodate annotations
                            y_min, y_max = plt.ylim()  # Get current y-axis limits
                            new_y_max = y_max + 0.1 * (y_max - y_min)  # Add 10% extra space to the top
                            plt.ylim(y_min, new_y_max)  # Set the adjusted y-axis limits
                        except ValueError as e:
                            logger.error(f"Plotter: Failed to highlight Onset '{analysis_value}' - {e}")


                    if "Peak" in analysis_key:
                        try:
                            peak_x_matches = re.findall(r"X:\s*([\d\.]+)", analysis_value)
                            if not peak_x_matches:
                                peak_x_matches = re.findall(r"([\d\.]+)", analysis_value)

                            for peak_x_str in peak_x_matches:
                                peak_x = float(peak_x_str)
                                peak_y_index = (x_data - peak_x).abs().argsort()[0]
                                peak_y = y_data.iloc[peak_y_index]

                                plt.scatter(peak_x, peak_y, color='green', label=f"{analysis_key}: X={peak_x:.2f}")
                                y_min, y_max = plt.ylim()
                                plt.annotate(
                                    f"Peak\n({peak_x:.2f}, {peak_y:.5f})",
                                    xy=(peak_x, peak_y),
                                    xytext=(peak_x, peak_y + 0.05 * (y_max - y_min)),
                                    arrowprops=dict(facecolor='green', arrowstyle="->"),
                                    fontsize=8,
                                    ha='center',
                                )
                                plt.axvline(x=peak_x, color='green', linestyle='--', label=f"Peak Line: X={peak_x:.2f}")
                                # Dynamically adjust the y-axis limits to accommodate annotations
                                y_min, y_max = plt.ylim()  # Get current y-axis limits
                                new_y_max = y_max + 0.1 * (y_max - y_min)  # Add 10% extra space to the top
                                plt.ylim(y_min, new_y_max)  # Set the adjusted y-axis limits
                        except ValueError as e:
                            logger.error(f"Plotter: Failed to highlight Peak '{analysis_value}' - {e}")

            # Add legend for highlighted points
            plt.legend()

            # Sanitize the mapped y_column for filenames
            sanitized_y_column = re.sub(r'[^\w\-_. ]', '_', mapped_y_column)  # Remove problematic characters

            # Save the plot
            output_file = f"{os.path.splitext(csv_file)[0]}_{sanitized_y_column}.png"
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()

            if debug:
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
        if len(sys.argv) < 2:
            print("Plotter: No CSV file provided.")
            print("Usage: python plotter.py <path_to_csv_file>")
            sys.exit(1)

        csv_file = sys.argv[1]
        debug = True  # Enable debug mode for additional logging
        logger.info(f"Plotter: Processing '{csv_file}'")
        print("Starting to process Y-axis columns...")
        # Call the plotting function
        plot_temperature_sweep(csv_file, debug=debug)

if __name__ == "__main__":
        main()