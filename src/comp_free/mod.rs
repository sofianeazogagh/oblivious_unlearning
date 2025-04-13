pub mod clear;
pub mod ctree;
pub mod dataset;
pub mod forest;
pub mod serial;
pub mod tree;

// REVOLUT
pub use revolut::{radix::ByteLWE, LUT, LWE, Context, PrivateKey, PublicKey};

// TFHE
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::parameters::*;

// type LWE = LweCiphertext<Vec<u64>>;
type RLWE = GlweCiphertext<Vec<u64>>;

// - Forest:
// 	- trees : \[Tree]
// 	-
