import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_column_name(column_name):
    """
    Sanitize column names by replacing problematic characters.
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

    # Fill remaining NaN values with placeholders
    df = df.fillna("")

    # Convert numeric columns where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            if debug:
                logger.debug(f"Could not convert column '{col}' to numeric. Keeping it as-is.")

    if debug:
        logger.debug("Sanitized DataFrame:")
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
        # Extract analysis results from the metadata
        analysis_results = {}
        if "Metadata Property" in df.columns:
            metadata_property_index = df.columns.get_loc("Metadata Property")  # Find the column index
            value_column_index = metadata_property_index + 1  # The next column index

            if value_column_index < len(df.columns):  # Ensure we don't go out of bounds
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
            logger.info(f"Plotter: Extracted analysis results: {analysis_results}")

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

            # Define mappings between metadata keys and column names
            metadata_to_column = {
                "Onset of Eprime": [col for col in df.columns if re.search(r"Eprime|E'", col, re.IGNORECASE)],
                "Peak of Edoubleprime": [col for col in df.columns if re.search(r"Edoubleprime|E''", col, re.IGNORECASE)],
                "Peak of tand": [col for col in df.columns if re.search(r"tand|tan ?δ", col, re.IGNORECASE)],
            }

            # Highlight relevant analysis points
            # Highlight relevant analysis points
            for analysis_key, columns in metadata_to_column.items():
                if analysis_key in analysis_results:
                    analysis_value = str(analysis_results[analysis_key]).strip()  # Convert and clean the value

                    if not analysis_value or analysis_value.lower() == "nan":
                        logger.warning(f"Skipping invalid metadata value for {analysis_key}: {analysis_value}")
                        continue
                    
                    if not analysis_value or analysis_value.lower() == "nan":
                        logger.warning(f"Skipping invalid metadata value for {analysis_key}: {analysis_value}")
                        continue

                    for column in columns:
                        if debug:
                            logger.info(f"Plotter: Highlighting '{analysis_key}' in column '{column}'")

                        # Ensure valid numeric data
                        x_data = pd.to_numeric(df[x_column], errors='coerce')
                        y_data = pd.to_numeric(df[column], errors='coerce').dropna()

                        if "Onset" in analysis_key:
                            try:
                                # Extract onset value
                                onset_x = float(analysis_value)
                                onset_y = y_data.iloc[(x_data - onset_x).abs().argsort()[0]]  # Find nearest Y
                                plt.scatter(onset_x, onset_y, color='red', label=f"Onset: X={onset_x:.2f}")
                                plt.annotate(f"Onset\n({onset_x:.2f}, {onset_y:.2f})", 
                                            xy=(onset_x, onset_y), xytext=(onset_x, onset_y + 0.05 * y_data.max()),
                                            arrowprops=dict(facecolor='red', arrowstyle="->"))
                            except ValueError as e:
                                logger.error(f"Plotter: Failed to highlight Onset '{analysis_value}' - {e}")

                        if "Peak" in analysis_key:
                            try:
                                # Extract X and Y values for peaks
                                peak_match = re.search(r"X:\s*([\d\.]+),\s*Y:\s*([\d\.]+)", analysis_value)
                                if peak_match:
                                    peak_x = float(peak_match.group(1))
                                    peak_y = float(peak_match.group(2))
                                    plt.scatter(peak_x, peak_y, color='green', label=f"Peak: X={peak_x:.2f}, Y={peak_y:.2f}")
                                    plt.annotate(f"Peak\n({peak_x:.2f}, {peak_y:.2f})", 
                                                xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.05 * y_data.max()),
                                                arrowprops=dict(facecolor='green', arrowstyle="->"))
                                else:
                                    logger.warning(f"Plotter: Invalid peak format for '{analysis_key}': {analysis_value}")
                            except ValueError as e:
                                logger.error(f"Plotter: Failed to highlight Peak '{analysis_value}' - {e}")


            # Add legend for highlighted points
            plt.legend()

            # Save the plot
            output_file = f"{os.path.splitext(csv_file)[0]}_{y_column}.png"
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

    # Call the plotting function
    plot_temperature_sweep(csv_file, debug=debug)

if __name__ == "__main__":
    main()
