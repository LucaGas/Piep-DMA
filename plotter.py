# plotter.py

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def identify_common_header(columns, debug=False):
    """
    Identify the common header as the one containing 'Temp' or 'Time'.
    Priority: 'Temp' > 'Time' > first column.
    """
    temp_headers = [col for col in columns if 'temp' in col.lower()]
    if temp_headers:
        if debug:
            print(f"Plotter: Identified common header based on 'Temp': {temp_headers[0]}")
        return temp_headers[0]
    
    time_headers = [col for col in columns if 'time' in col.lower()]
    if time_headers:
        if debug:
            print(f"Plotter: Identified common header based on 'Time': {time_headers[0]}")
        return time_headers[0]
    
    # Default to first column if neither 'Temp' nor 'Time' is found
    if columns:
        if debug:
            print(f"Plotter: No 'Temp' or 'Time' found. Defaulting to first header: {columns[0]}")
        return columns[0]
    else:
        if debug:
            print("Plotter: No columns found to identify common header.")
        return None

def plot_data(csv_path, debug=False):
    """
    Plot the data from the CSV file using the common header as X-axis and other data as Y-axis.
    Save the plot in the same directory as the CSV file.
    """
    try:
        # Load the CSV file with UTF-8 encoding
        data = pd.read_csv(csv_path, encoding='utf-8')
        if debug:
            print(f"Plotter: Loaded data from '{csv_path}' with {len(data)} records.")
        
        # Identify the common header
        common_header = identify_common_header(data.columns, debug=debug)
        if not common_header:
            print(f"Plotter Error: Unable to identify common header in '{csv_path}'. Skipping plot.")
            return
        
        # Determine if the common header is related to temperature
        is_temp = 'temp' in common_header.lower()
        
        # Set the X-axis label
        if is_temp:
            x_label = "Temp (°C)"
        else:
            x_label = common_header  # Use the exact header name
        
        # Identify Y-axis columns (exclude the common header and metadata)
        metadata_keys = ['ASSAY', 'FILE']
        y_columns = [col for col in data.columns if col not in [common_header] + metadata_keys]
        
        if debug:
            print(f"Plotter: Common header identified as '{common_header}'. Y-axis columns: {y_columns}")
            print(f"Plotter: X-axis label set to '{x_label}'")
        
        if not y_columns:
            print(f"Plotter Warning: No Y-axis data columns found in '{csv_path}'. Skipping plot.")
            return
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        for y_col in y_columns:
            # Convert Y-axis data to numeric, coercing errors
            y_data = pd.to_numeric(data[y_col], errors='coerce')
            plt.plot(data[common_header], y_data, marker='o')  # Removed label parameter to exclude legend
            if debug:
                print(f"Plotter: Plotted '{y_col}' against '{common_header}'")
        
        plt.xlabel(x_label)
        
        # Set Y-axis label
        if len(y_columns) == 1:
            plt.ylabel(y_columns[0])
            if debug:
                print(f"Plotter: Y-axis label set to '{y_columns[0]}'")
        else:
            plt.ylabel('Value')  # Generic label if multiple Y columns exist
            if debug:
                print("Plotter: Y-axis label set to 'Value'")
        
        plt.title(f"Plot for {Path(csv_path).stem}")
        # Removed plt.legend() to exclude the legend from the plot
        plt.grid(True)
        
        # Define the plot file path
        csv_file = Path(csv_path)
        plot_path = csv_file.parent / f"{csv_file.stem}.png"
        
        # Save the plot with UTF-8 support
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plotter: Plot saved to '{plot_path}'.")
        if debug:
            print(f"Plotter: Successfully created plot for '{csv_path}'.")
    
    except Exception as e:
        print(f"Plotter Error: An unexpected error occurred - {e}")

def main():
    """
    Main function to handle command-line arguments and initiate plotting.
    """
    if len(sys.argv) != 2:
        print("Usage: python plotter.py <path_to_csv_file>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    plot_data(csv_path, debug=True)  # Set debug=True for detailed logs

if __name__ == "__main__":
    main()
