# Defaults parameters (iris, 8,16 arbres, 10 trials)
```bash
cargo run --release
```

# With personalized option
```bash
cargo run --release -- --dataset wine --num-trees "16,32" --depth 5 --trials 5 --verbose
```
# Help
```bash
cargo run --release -- --help
```

## Directory Structure

```
oblivious_unlearning/
├── README.md
├── tree.rs
├── ctree.rs
├── forest.rs
├── utils_maj.rs
├── utils_serial.rs
└── dataset.rs
```

### File Explanations

- **tree.rs**: Defines the `Tree` structure, representing a decision tree with nodes and leaves. Includes methods for creating, generating, and printing trees, as well as tests for tree operations.
- **ctree.rs**: Defines the `CTree` structure, a compiled version of a decision tree. Includes methods for creating a `CTree` from a `Tree` and an `EncryptedSample`, evaluating, and printing the `CTree`.
- **forest.rs**: Defines the `Forest` structure, a collection of `Tree` objects, representing a random forest.
- **mod.rs**: Serves as a module index, re-exporting modules and importing necessary components from external libraries.
- **dataset.rs**: Defines structures and methods for handling datasets, including unencrypted and encrypted data, with methods for loading, splitting, and encrypting datasets.


