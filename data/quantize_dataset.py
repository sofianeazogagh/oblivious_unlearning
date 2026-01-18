#!/usr/bin/env python3
"""
Dataset quantization script for machine learning applications.

This script quantizes numerical features of a dataset using a specified number
of bits, allowing to reduce data precision for cryptographic or homomorphic
computing applications.

Usage:
    python quantize_dataset.py input.csv output.csv --bits 11 --class_column class
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path


def quantize_to_bits(data, b):
    """
    Quantize the dataset to integers using b bits.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset to quantize
    b : int
        The number of bits to use for quantization

    Returns:
    --------
    pd.DataFrame
        The quantized dataset
    """
    # Calculate the number of levels
    num_levels = 2 ** b

    # Initialize a DataFrame to store the quantized data
    quantized_data = pd.DataFrame()

    for column in data.columns:
        if data[column].dtype in [np.float64, np.int64, np.float32, np.int32]:
            # Find the min and max of the column
            col_min = data[column].min()
            col_max = data[column].max()

            # Avoid division by zero
            if col_max == col_min:
                quantized_data[column] = data[column].astype(int)
            else:
                # Quantize the column
                quantized_data[column] = (
                    (data[column] - col_min) / (col_max - col_min) * (num_levels - 1)
                ).round().astype(int)
        else:
            # If the column is not numeric, copy it as is
            quantized_data[column] = data[column]

    return quantized_data


def convert_class_to_numeric(y, class_column_name="class"):
    """
    Convert class column to numeric values if it's not already numeric.
    Creates a mapping from unique class values to integers (0, 1, 2, ...).

    Parameters:
    -----------
    y : pd.Series
        The class column to convert
    class_column_name : str
        Name of the class column (for display purposes)

    Returns:
    --------
    pd.Series
        The converted class column with numeric values
    dict
        Mapping from original class values to numeric values
    """
    # Check if already numeric
    if pd.api.types.is_numeric_dtype(y):
        unique_values = sorted(y.unique())
        # Check if values are consecutive integers starting from 0
        is_consecutive_from_zero = (
            len(unique_values) > 0 and
            unique_values[0] == 0 and
            unique_values == list(range(len(unique_values)))
        )
        
        if is_consecutive_from_zero:
            print(f"Class column '{class_column_name}' is already numeric (range: 0-{len(unique_values)-1})")
            return y, {}
        else:
            # Convert to integer starting from 0
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            y_numeric = y.map(mapping)
            print(f"Class column '{class_column_name}' converted from numeric to consecutive integers (0-{len(unique_values)-1}):")
            for original_val, numeric_val in mapping.items():
                if original_val != numeric_val:
                    print(f"  {original_val} -> {numeric_val}")
            return y_numeric, mapping
    
    # Not numeric - create mapping
    unique_values = sorted(y.unique())
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    y_numeric = y.map(mapping)
    
    print(f"Class column '{class_column_name}' converted to numeric:")
    for original_val, numeric_val in mapping.items():
        print(f"  {original_val} -> {numeric_val}")
    
    return y_numeric, mapping


def display_characteristics(df_quantized, df_original, class_column):
    """
    Display dataset characteristics.

    Parameters:
    -----------
    df_quantized : pd.DataFrame
        The quantized dataset
    df_original : pd.DataFrame
        The original dataset
    class_column : str
        The name of the class column
    """
    num_features = df_quantized.shape[1] - 1  # Exclude the class column
    num_instances = df_quantized.shape[0]
    
    if class_column in df_original.columns:
        num_classes = df_original[class_column].nunique()
    else:
        num_classes = "N/A"

    print(f"Number of features: {num_features}")
    print(f"Number of instances: {num_instances}")
    print(f"Number of classes: {num_classes}")


def process_dataset(input_file, output_file, bits=11, class_column=None, 
                   delimiter=None, header=None):
    """
    Process a dataset: load, quantize and export.

    Parameters:
    -----------
    input_file : str
        Path to the input CSV or .data file
    output_file : str
        Path to the output CSV file
    bits : int
        Number of bits for quantization (default: 11)
    class_column : str or int
        Name or index of the class column (required)
    delimiter : str, optional
        Delimiter to use when reading the file (default: auto-detect)
    header : int or None, optional
        Row to use as column names (default: auto-detect, None for no header)
    """
    if class_column is None:
        print("Error: --class_column is required. Please specify which column contains the class labels.", file=sys.stderr)
        sys.exit(1)
    # Load the dataset
    print(f"Loading dataset from {input_file}...")
    
    # Auto-detect if file is .data (typically no header) or CSV
    file_ext = Path(input_file).suffix.lower()
    if file_ext == '.data':
        # For .data files, assume no header
        has_header = None
        print("Detected .data file, assuming no header")
    else:
        # For CSV files, use provided header or auto-detect
        has_header = header
    
    # Auto-detect delimiter if not specified
    if delimiter is None:
        # Try common delimiters
        with open(input_file, 'r') as f:
            first_line = f.readline()
            if ',' in first_line:
                delimiter = ','
            elif '\t' in first_line:
                delimiter = '\t'
            elif ';' in first_line:
                delimiter = ';'
            else:
                delimiter = ','  # Default to comma
        print(f"Auto-detected delimiter: '{delimiter}'")
    
    try:
        df = pd.read_csv(input_file, delimiter=delimiter, header=has_header)
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        print(f"Trying with header=None...", file=sys.stderr)
        try:
            # Try without header as fallback
            df = pd.read_csv(input_file, delimiter=delimiter, header=None)
        except Exception as e2:
            print(f"Error loading file: {e2}", file=sys.stderr)
            sys.exit(1)

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check if columns are numeric (no header) or named (with header)
    has_numeric_columns = df.columns.dtype == 'int64' or all(isinstance(col, int) for col in df.columns)
    
    # Handle -1 as last column index
    original_class_column = class_column
    if class_column == "-1" or class_column == -1:
        if has_numeric_columns:
            class_column = df.columns[-1]
            print(f"Using last column (index {class_column}) as class column")
        else:
            class_column = df.columns[-1]
            print(f"Using last column ('{class_column}') as class column")
    elif has_numeric_columns:
        print("No header detected, using numeric column indices")
        # Try to convert class_column to int if it's a string representation of a number
        try:
            class_column = int(class_column)
        except (ValueError, TypeError):
            pass  # Keep as string if conversion fails
        
        if class_column not in df.columns:
            print(f"Error: Column '{original_class_column}' does not exist in the dataset.", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"Class column specified: column {class_column}")
    else:
        if class_column not in df.columns:
            print(f"Error: Column '{class_column}' does not exist in the dataset.", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"Class column specified: {class_column}")

    # Separate features and class - exclude only the specified class column
    feature_columns = [col for col in df.columns if col != class_column]
    print(f"Class column '{class_column}' excluded from quantization")

    X = df[feature_columns]
    y = df[class_column]

    # Convert class column to numeric if needed
    print(f"\nProcessing class column '{class_column}'...")
    y_numeric, class_mapping = convert_class_to_numeric(y, class_column)
    
    # Ensure the numeric class column has the same name as the original
    y_numeric.name = class_column

    # Quantize the features
    print(f"\nQuantizing with {bits} bits...")
    X_quantized = quantize_to_bits(X, bits)

    # Recombine with the numeric class column
    dataset_processed = pd.concat([X_quantized, y_numeric], axis=1)

    # Display characteristics
    print("\nQuantized dataset characteristics:")
    display_characteristics(dataset_processed, df, class_column)

    # Export
    print(f"\nExporting to {output_file}...")
    dataset_processed.to_csv(output_file, index=False, header=False)
    print("Export completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a dataset using a specified number of bits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize wine.data with 11 bits (default), class column is first column (index 0)
  python quantize_dataset.py wine.data wine.csv --class_column 0

  # Use last column as class (use -1)
  python quantize_dataset.py iris.data iris.csv --class_column -1

  # Quantize with 8 bits, specify the class column by name
  python quantize_dataset.py input.csv output.csv --bits 8 --class_column label

  # For .data files (UCI format, no header), use column index
  python quantize_dataset.py iris.data iris.csv --class_column 4

Note: 
  - The class column must be specified with --class_column
  - Use -1 to specify the last column
  - For .data files (UCI format), use numeric column indices (0, 1, 2, ... or -1 for last)
  - The class column is automatically converted to numeric values (0, 1, 2, ...)
  - Output is always without header
        """
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input CSV or .data file"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output CSV file"
    )
    parser.add_argument(
        "--class_column",
        type=str,
        required=True,
        help="Name or index of the class column (required). Use -1 for last column. For .data files, use numeric index (0, 1, 2, ... or -1)"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=11,
        help="Number of bits for quantization (default: 11)"
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=None,
        help="Delimiter to use when reading the file (default: auto-detect)"
    )
    parser.add_argument(
        "--header",
        type=int,
        default=None,
        help="Row to use as column names (default: auto-detect, 0 for first row, None for no header). For .data files, header=None is used automatically."
    )

    args = parser.parse_args()

    # Check that the input file exists
    if not Path(args.input_file).exists():
        print(f"Error: File '{args.input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Create output directory if necessary
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process the dataset
    process_dataset(
        args.input_file,
        args.output_file,
        bits=args.bits,
        class_column=args.class_column,
        delimiter=args.delimiter,
        header=args.header
    )


if __name__ == "__main__":
    main()

