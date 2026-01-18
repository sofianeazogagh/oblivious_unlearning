# Oblivious (Un)Learning of Extremely Randomized Trees using TFHE


## Dependencies 

You need to install Rust and Cargo to use tfhe-rs and RevoLUT.

First, install the needed Rust toolchain:
```bash
rustup toolchain install nightly
```

Then, you can either:

1. Manually specify the toolchain to use in each of the cargo commands:
For example:
```bash
cargo +nightly build
cargo +nightly run
```
2. Or override the toolchain to use for the current project:
```bash
rustup override set nightly
```

Cargo will use the `nightly` toolchain.
```
cargo build
```

## Usage

The program supports three execution modes: **standard**, **hybrid** and **oblivious**.
The three execution modes supported by this program are as follows:

- **Standard Mode**:  
  In this mode, the forest of extremely randomized trees (ERTs) is fully trained on unencrypted (clear) data. After training, the best-performing model is selected (optionally using multiple trials), exported (without count data), and is retrained on encrypted data. This mode is useful for benchmarking and verifying correctness.

- **Hybrid Mode**:  
  The dataset is split into two subsets, D_0 and D_1. The ERTs are first trained on D_0 in the clear, using the Gini-index criterion, and counts for classes at each leaf are recorded (exported with the model). Then, additional training (or updating) can be done using encrypted data from D_1.

- **Oblivious Mode (Oblivious Training/Unlearning)**:  
  This mode allows you to perform *oblivious* training or unlearning of the forest on encrypted data from a provided CSV file. The same function is used for both operations; labels are encoded differently to indicate whether you are training (add counts) or unlearning (remove counts).


### Standard Mode

The standard mode trains a forest in clear, selects the best model, exports it without counts, then retrains with encrypted data to demonstrate that training with encrypted data does not degrade accuracy.

**Example:**
```bash
cargo run --release -- --mode standard --dataset iris --num-trees 8 --depth 4  --split 0.8 --best-model-trials 10 --verbose
```

**Options:**
- `--dataset` or `-d`: Dataset name (iris, wine, cancer). Default: `iris`
- `--num-trees` or `-n`: Number of trees (comma-separated values, e.g., "8,16"). Default: `8,16`
- `--depth`: Tree depth. Default: `4`
- `--split`: Train/test split percentage. Default: `0.8`
- `--trials` or `-t`: Number of repetitions/trials. Default: `10`
- `--best-model-trials`: Number of trials to find the best model. Default: `1`
- `--output`: Output directory. Default: `./export/`
- `--verbose` or `-v`: Verbose mode
- `--mode`: Execution mode (`standard` or `hybrid`). Default: `standard`

### Hybrid Mode

The hybrid mode, is more "practical", it splits the dataset into D_0 and D_1, trains ERTs in clear on D_0 using Gini-index, exports the forest with counts, then continues training on encrypted D_1.

**Example:**
```bash
cargo run --release -- --mode hybrid --dataset iris --num-trees 8 --depth 4  --split 0.3 --verbose
```

**Options:**
- Same as standard mode, with the following differences:
- `--split`: Used for D_0/D_1 split (percentage for D_0, remainder is D_1). The train/test split within D_0 is fixed at 0.8
- `--mode`: Must be set to `hybrid`

### Oblivious Training/Unlearning Mode

The oblivious mode allows you to train or unlearn on a pre-trained forest using encrypted data from a CSV file. The same function is used for both operations; the only difference is in the label encoding:
- **Training**: multiply the one-hot encoded label by 1
- **Unlearning**: multiply the one-hot encoded label by 2

**Example (Training):**
```bash
cargo run --release -- --mode oblivious --forest-path ./export/iris/8/best_iris_8_4_0.95_exported.json --csv-path ./data/iris-uci/iris-train.csv --operation train --verbose
```

**Example (Unlearning):**
```bash
cargo run --release -- --mode oblivious --forest-path ./export/iris/8/best_iris_8_4_0.95_exported.json --csv-path ./data/iris-uci/iris-unlearn.csv --operation unlearn --verbose
```

**Options:**
- `--mode`: Must be set to `oblivious`
- `--forest-path`: Path to the pre-trained forest JSON file (required)
- `--csv-path`: Path to the CSV file containing data to train/unlearn (required)
- `--operation`: Operation type (`train` or `unlearn`). Default: `train`
- `--output`: Output directory for the updated forest. Default: `./export/`
- `--verbose` or `-v`: Verbose mode

**CSV Format:**
The CSV file should have the same format as the training data:
- Features as the first columns
- Class label as the last column
- No header row

### Common Examples

**Run standard mode with default settings:**
```bash
cargo run --release -- --mode standard
```

**Run hybrid mode with custom dataset and parameters:**
```bash
cargo run --release -- --dataset cancer --num-trees 16 --depth 5 --mode hybrid --split 0.3 --verbose
```

**Run standard mode with multiple forest configurations:**
```bash
cargo run --release -- --dataset wine --num-trees 8,16,32 --depth 4 --mode standard --best-model-trials 5 --verbose
```

**Run oblivious training on a pre-trained forest:**
```bash
cargo run --release -- --mode oblivious --forest-path ./export/iris/8/best_iris_8_4_0.95_exported.json --csv-path ./data/iris-uci/iris-sample.csv --operation train --verbose
```

**Run oblivious unlearning on a pre-trained forest:**
```bash
cargo run --release -- --mode oblivious --forest-path ./export/iris/8/best_iris_8_4_0.95_exported.json --csv-path ./data/iris-uci/iris_to_unlearn.csv --operation unlearn --verbose
```

### Output

The program generates:
- Exported forest files (JSON format) in the `export` directory
- Performance metrics and timing statistics
- Accuracy comparisons between clear and encrypted training

The output structure is:
```
export/
  {dataset}/
    {num_trees}/
      standard_{dataset}_{num_trees}_{depth}_{accuracy}.json  (standard mode)
      hybrid_{dataset}_{num_trees}_{depth}_{accuracy}.json  (hybrid mode)
      perf.csv  (performance metrics)
  forest_{operation}_{forest_filename}_{csv_filename}.json  (oblivious mode)
```
