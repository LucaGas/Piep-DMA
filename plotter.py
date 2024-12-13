# plotter.py

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import chardet  # Ensure chardet is installed: pip install chardet

def detect_file_encoding(file_path, debug=False):
    """
    Detect the encoding of the given file using chardet.
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    confidence = result['confidence']
    if debug:
        print(f"Plotter: Detected encoding '{encoding}' with confidence {confidence} for file '{file_path}'")
    return encoding if encoding else 'utf-8'  # Default to 'utf-8' if detection fails

def plot_data(csv_path, debug=False):
    """
    Plot the data from the CSV file using the first column as X-axis and the second column as Y-axis.
    Save the plot in the same directory as the CSV file.
    """
    try:
        # Detect encoding
        encoding = detect_file_encoding(csv_path, debug=debug)
        
        # Load the CSV file with detected encoding
        data = pd.read_csv(csv_path, encoding=encoding)
        if debug:
            print(f"Plotter: Loaded data from '{csv_path}' with {len(data)} records.")
            print("\nPlotter: Data Types of All Columns:")
            print(data.dtypes)
        
        # Ensure there are at least two columns
        if data.shape[1] < 2:
            print(f"Plotter Error: The file '{csv_path}' does not contain enough columns for plotting.")
            return
        
        # Assign the first column to X-axis and the second column to Y-axis
        x_column = data.columns[0]
        y_column = data.columns[1]
        
        if debug:
            print(f"Plotter: X-axis column identified as '{x_column}'")
            print(f"Plotter: Y-axis column identified as '{y_column}'")
        
        # Handle encoding issues in column names
        if '�' in x_column or '�' in y_column:
            corrected_x = x_column.replace('�', '°')
            corrected_y = y_column.replace('�', '°')
            if debug:
                print(f"Plotter: Corrected X-axis column from '{x_column}' to '{corrected_x}'")
                print(f"Plotter: Corrected Y-axis column from '{y_column}' to '{corrected_y}'")
            x_column = corrected_x
            y_column = corrected_y
        
        # Extract X and Y data
        x_data = data[x_column]
        y_data = pd.to_numeric(data[y_column], errors='coerce')  # Convert Y data to numeric, coercing errors
        
        # Check if Y data is valid
        if y_data.isnull().all():
            print(f"Plotter Warning: All Y-axis data in '{y_column}' are non-numeric. Skipping plot.")
            return
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, marker='o', linestyle='-')  # Simple line plot with markers
        
        # Set axis labels
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        if debug:
            print(f"Plotter: X-axis label set to '{x_column}'")
            print(f"Plotter: Y-axis label set to '{y_column}'")
        
        # Set plot title
        plt.title(f"Plot for {Path(csv_path).stem}")
        
        # Add grid for better readability
        plt.grid(True)
        
        # Define the plot file path
        csv_file = Path(csv_path)
        plot_path = csv_file.parent / f"{csv_file.stem}.png"
        
        # Save the plot with high resolution and tight layout
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plotter: Plot saved to '{plot_path}'.")
        if debug:
            print(f"Plotter: Successfully created plot for '{csv_path}'.")
    
    except FileNotFoundError:
        print(f"Plotter Error: The file '{csv_path}' does not exist.")
    except pd.errors.EmptyDataError:
        print(f"Plotter Error: The file '{csv_path}' is empty.")
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
