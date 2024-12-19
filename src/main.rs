use std::io::Write;
use std::time::Instant;

mod probonite;
use probonite::*;

mod probolut;
use probolut::*;

use bincode::{config::AllowTrailing, de};

mod model;
use model::*;

mod xt_probolut;
use xt_probolut::*;

mod dataset;
use dataset::*;

mod clear_model;
use clear_model::*;

mod xt_clear;
use xt_clear::*;

mod xt_probonite;
use xt_probonite::*;

use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    vec,
};
use revolut::{key, Context, PrivateKey, LUT};

use tfhe::{core_crypto::prelude::LweCiphertext, shortint::parameters::*};

type LWE = LweCiphertext<Vec<u64>>;

const GENERATE_TREE: bool = true;
const EXPORT_FOREST: bool = true;

fn main() {
    let clear_dataset = ClearDataset::from_file("data/iris_2bits.csv".to_string());

    let tree = generate_clear_random_tree(
        3,
        clear_dataset.n_classes,
        clear_dataset.features_domain,
        clear_dataset.f,
    );

    tree.print();
}
