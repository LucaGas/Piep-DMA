# plotter.py

import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    """
    Plot data from the CSV file.
    """
    if len(sys.argv) < 2:
        print("Plotter: No CSV file provided.")
        sys.exit(1)

    csv_file = sys.argv[1]
    print(f"Plotter: Processing {csv_file}")

    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
        print(f"Plotter: Loaded data from '{csv_file}' with {len(df)} records.")

        # Extract metadata columns
        metadata_df = df.iloc[:, -2:].dropna(how="all")  # Last two columns
        metadata = {row["Metadata Property"]: row["Metadata Value"] for _, row in metadata_df.iterrows()}

        # Drop the metadata rows and empty column
        data_df = df.iloc[:, :-2].dropna()

        # Identify the common header and Y-axis columns
        common_header = data_df.columns[0]
        y_columns = data_df.columns[1:]  # All remaining columns after the first

        if len(y_columns) == 0:
            print("Plotter: No Y-axis columns found. Skipping.")
            return

        print(f"Plotter: Common header identified as '{common_header}'.")

        # Plot each Y column against the common header
        for y_col in y_columns:
            plt.figure()
            plt.plot(data_df[common_header], data_df[y_col], marker="o")
            plt.xlabel(common_header)
            plt.ylabel(y_col)
            plt.title(f"{y_col} vs {common_header}")
            plt.grid(True)

            # Save the plot
            output_file = csv_file.replace(".csv", f"_{y_col.replace('/', '_')}.png")
            plt.savefig(output_file)
            plt.close()
            print(f"Plotter: Plot saved to '{output_file}'")

    except Exception as e:
        print(f"Plotter: An error occurred while processing {csv_file}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
