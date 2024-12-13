import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import os

def sanitize_column_name(column_name):
    """
    Sanitize column names by replacing problematic characters.
    """
    if not isinstance(column_name, str):
        return column_name

    sanitized_name = (
        column_name.replace(' ', ' ')
        .replace('E"', 'E5678')
        .replace("E5678","E''")
        #.replace("E'", "E1234")
        .replace('"', '')
        .replace("/", " ")
        .replace("°", "º")
        #.replace("(", "")
        #.replace(")", "")
        .replace("�", "°")  # Handle encoding errors
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

    # Optionally convert numeric columns back to float/int if needed
    # Convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])  # Convert to numeric where possible
        except ValueError:
            if debug:
                print(f"Could not convert column '{col}' to numeric. Keeping it as-is.")

    if debug:
        print("Sanitized DataFrame:")
        print(df)

    return df

def plot_temperature_sweep(csv_file, debug=False):
    """
    Plot data from a CSV file where the X-axis columns are identified by 'Temp' in their headers,
    and each subsequent column is used as the Y-axis for plotting.

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
            print(f"Plotter: Sanitized data columns: {list(df.columns)}")

        # Sanitize the DataFrame
        df = sanitize_dataframe(df, debug=debug)

        # Identify 'Temp' columns
        temp_columns = [
            idx for idx, col in enumerate(df.columns)
            if "temp" in col.lower() and not pd.isna(col)
        ]
        if debug:
            print(f"Plotter: Identified 'Temp' columns at indices: {temp_columns}")

        # Iterate over identified Temp columns and pair with the next column for plotting
        for temp_idx in temp_columns:
            x_column = df.columns[temp_idx]
            y_idx = temp_idx + 1  # Pair with the next column
            if y_idx < len(df.columns):
                y_column = df.columns[y_idx]

                if debug:
                    print(f"Plotter: Preparing to plot X-axis '{x_column}' with Y-axis '{y_column}'")

                # Ensure valid numeric data
                x_data = pd.to_numeric(df[x_column], errors='coerce')
                y_data = pd.to_numeric(df.iloc[:, y_idx], errors='coerce')  # Explicitly select by index

                # Drop rows with NaN values
                valid_data = pd.DataFrame({x_column: x_data, y_column: y_data}).dropna()

                # Debug valid data
                if valid_data.empty:
                    if debug:
                        print(f"Plotter: No valid data for X-axis '{x_column}' and Y-axis '{y_column}'. Skipping.")
                    continue

                if debug:
                    print(f"Plotter: Valid data for X-axis '{x_column}' and Y-axis '{y_column}':\n{valid_data.head()}")

                # Plot the data
                plt.figure()
                plt.plot(valid_data[x_column], valid_data[y_column], marker='o')
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title(f"{y_column} vs {x_column}")
                plt.grid(True)

                # Save the plot
                output_file = os.path.splitext(csv_file)[0] + f"_{y_column.replace('/', '_')}.png"
                plt.savefig(output_file)
                plt.close()
                if debug:
                    print(f"Plotter: Plot saved to '{output_file}'")
            else:
                if debug:
                    print(f"Plotter: No Y-axis found for X-axis column '{x_column}'. Skipping.")

    except Exception as e:
        print(f"Plotter: An error occurred while processing '{csv_file}': {e}")

def main():
    """
    Main function to handle standalone execution of plotter.py.
    """
    if len(sys.argv) < 2:
        print("Plotter: No CSV file provided.")
        sys.exit(1)

    csv_file = sys.argv[1]
    debug = True  # Enable debug mode for additional logging
    print(f"Plotter: Processing {csv_file}")

    # Call the plotting function
    plot_temperature_sweep(csv_file, debug=debug)

if __name__ == "__main__":
    main()
