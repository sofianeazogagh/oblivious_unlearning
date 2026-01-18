# Dataset Quantization Script Usage Guide

The `quantize_dataset.py` script allows you to quantize numerical features of a dataset using a specified number of bits. This process is useful for reducing data precision for cryptographic or homomorphic computing applications.

## Installation

No special installation is required, only standard Python libraries:
- `pandas`
- `numpy`

```bash
pip install pandas numpy
```

## Basic Usage

### Simple Example

```bash
python quantize_dataset.py input.data output.csv --class_column 0
```

This command:
- Loads `input.data` (or `input.csv`)
- For `.data` files, automatically detects no header (typical for UCI datasets)
- Quantizes all numerical columns with 11 bits (default)
- Uses column 0 as the class column (specified with `--class_column`)
- **Excludes the class column from quantization**
- **Converts the class column to numeric values (0, 1, 2, ...)** if needed
- Exports the result to `output.csv` (without header)

### Important Notes

- **`--class_column` is required**: You must specify which column contains the class labels
- Use `-1` to specify the last column
- For `.data` files (no header), use numeric column indices: `0`, `1`, `2`, etc., or `-1` for last column
- For CSV files (with header), use the column name: `label`, `class`, etc.
- The class column is automatically converted to consecutive integers starting from 0
- Output is always without header

### Examples

#### Wine dataset (class in first column, values 1,2,3 → converted to 0,1,2)

```bash
python quantize_dataset.py wine.data wine.csv --class_column 0
```

#### Iris dataset (class in last column)

```bash
python quantize_dataset.py iris.data iris.csv --class_column -1
```


#### Specify the number of bits

```bash
python quantize_dataset.py input.data output.csv --class_column 0 --bits 8
```

#### CSV file with header

```bash
python quantize_dataset.py input.csv output.csv --class_column label
```

## Input File Format

The input file can be:
- A **CSV file** (`.csv`) with or without header
- A **`.data` file** (UCI format, typically no header) - automatically detected

The file should contain:
- A class/label column (specified with `--class_column`)
- Numerical columns (float or int) that will be quantized

**Important:** 
- **`--class_column` is required** - you must specify which column contains the class labels
- For `.data` files, the script automatically assumes no header (typical for UCI datasets)
- **Class column conversion**: The class column is automatically converted to numeric values (0, 1, 2, ...) if it's not already consecutive integers starting from 0. For example:
  - "Iris-setosa" → 0, "Iris-versicolor" → 1, "Iris-virginica" → 2
  - "M" → 0, "B" → 1
  - 1, 2, 3 → 0, 1, 2 (wine dataset)
  - The mapping is displayed during processing

## Output File Format

The output file is a CSV containing:
- The quantized features (integer values between 0 and 2^bits - 1)
- The class column (converted to numeric values 0, 1, 2, ...)
- **No header** (always exported without column names)

## Examples for Reusing Existing Datasets

### Wine dataset

```bash
cd data/wine-uci
python ../quantize_dataset.py wine.data wine.csv --class_column 0
```

The first column (index 0) contains the class values 1, 2, 3, which will be converted to 0, 1, 2.

### Cancer dataset

```bash
cd data/cancer-uci
python ../quantize_dataset.py wdbc.data cancer.csv --class_column 1
```

The second column (index 1) contains the Diagnosis class ("M" or "B"), which will be converted to 0 and 1.

### Iris dataset

```bash
cd data/iris-uci
python ../quantize_dataset.py iris.data iris.csv --class_column -1
```

Or using the column index:

```bash
cd data/iris-uci
python ../quantize_dataset.py iris.data iris.csv --class_column 4
```

The last column contains the class labels (Iris-setosa, Iris-versicolor, Iris-virginica), which will be converted to 0, 1, 2.

## How Quantization Works

Quantization transforms each numerical feature into an integer between 0 and 2^bits - 1 using the formula:

```
quantized_value = round((value - min) / (max - min) * (2^bits - 1))
```

This ensures that:
- The minimum value becomes 0
- The maximum value becomes 2^bits - 1
- Intermediate values are uniformly distributed

## Complete Help

To see all available options:

```bash
python quantize_dataset.py --help
```
