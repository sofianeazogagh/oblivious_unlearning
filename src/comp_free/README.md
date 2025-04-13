## Some tests that can be run

If you run `cargo test --release -- --nocapture test_compile_tree`, it is a test that will generate a random tree and a sample, then compile the tree with the sample into a comparison tree and return the selected leaf.

If you run `cargo test --release -- --nocapture test_leaves_update_multiple_samples`, it is a test that will generate a random tree and multiple samples, then update the leaves of the tree with the classes of the samples.

If you run `cargo test --release -- --nocapture test_train_forest`, it is a test that will generate a random forest and train it on the iris dataset. It is not parallelized so you can see live the training of the forest. But it can be parallelized by changing the number of threads in the `ThreadPoolBuilder`.

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


# Campaign 1

To run the campaign, you can use the following command:

```
nohup cargo test --release -- --nocapture test_bench_best > campaign_1.log &
```

