# Oblivious (Un)Learning of Extreremly Randomized Trees using TFHE


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
