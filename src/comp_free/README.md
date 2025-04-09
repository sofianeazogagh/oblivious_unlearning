If you run `cargo test --release -- --nocapture test_compile_tree`, it is a test that will generate a random tree and a sample, then compile the tree with the sample into a comparison tree and return the selected leaf.

## Directory Structure

```
comp_free/
├── README.md
├── tree.rs
├── ctree.rs
├── forest.rs
├── mod.rs
└── dataset.rs
```

### File Explanations

- **tree.rs**: Defines the `Tree` structure, representing a decision tree with nodes and leaves. Includes methods for creating, generating, and printing trees, as well as tests for tree operations.
- **ctree.rs**: Defines the `CTree` structure, a compiled version of a decision tree. Includes methods for creating a `CTree` from a `Tree` and an `EncryptedSample`, evaluating, and printing the `CTree`.
- **forest.rs**: Defines the `Forest` structure, a collection of `Tree` objects, representing a random forest.
- **mod.rs**: Serves as a module index, re-exporting modules and importing necessary components from external libraries.
- **dataset.rs**: Defines structures and methods for handling datasets, including unencrypted and encrypted data, with methods for loading, splitting, and encrypting datasets.
