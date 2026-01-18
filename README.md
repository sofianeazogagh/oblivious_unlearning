# Oblivious (Un)Learning of Extremely Randomized Trees using TFHE


## Dependencies 

You need to install Rust and Cargo to use tfhe-rs.

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

The program supports two execution modes: **standard** and **hybrid**. Both modes allow you to train random forests with encrypted data.

### Standard Mode

The standard mode trains a forest in clear, selects the best model, exports it without counts, then retrains with encrypted data to demonstrate that training with encrypted data does not degrade accuracy.

**Example:**
```bash
cargo run --release -- --dataset iris --num-trees 8 --depth 4 --mode standard --split 0.8 --best-model-trials 10 --verbose
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
cargo run --release -- --dataset iris --num-trees 8 --depth 4 --mode hybrid --split 0.3 --verbose
```

**Options:**
- Same as standard mode, with the following differences:
- `--split`: Used for D_0/D_1 split (percentage for D_0, remainder is D_1). The train/test split within D_0 is fixed at 0.8
- `--mode`: Must be set to `hybrid`

### Common Examples

**Run standard mode with default settings:**
```bash
cargo run --release -- --mode standard
```

**Run hybrid mode with custom dataset and parameters:**
```bash
cargo run --release -- --dataset cancer --num-trees 16 --depth 5 --mode hybrid --split 0.3 --verbose
```

**Run standard mode with multiple tree configurations:**
```bash
cargo run --release -- --dataset wine --num-trees 8,16,32 --depth 4 --mode standard --best-model-trials 5 --verbose
```

### Output

The program generates:
- Exported forest files (JSON format) in the output directory
- Performance metrics and timing statistics
- Accuracy comparisons between clear and encrypted training

The output structure is:
```
export/
  {dataset}/
    {num_trees}/
      best_{dataset}_{num_trees}_{depth}_{accuracy}_exported.json  (standard mode)
      hybrid_{dataset}_{num_trees}_{depth}_{accuracy}_with_counts.json  (hybrid mode)
      perf.csv  (performance metrics)
```
