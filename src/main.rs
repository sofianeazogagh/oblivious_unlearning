#![allow(warnings)]

use std::io::Write;
use std::time::Instant;

mod probonite;
use probonite::*;

use bincode::{config::AllowTrailing, de};

mod model;
use model::*;

mod xt_probolut;
use xt_probolut::*;

mod dataset;
use dataset::*;

mod clear_model;
use clear_model::*;

mod xt_probonite;
use xt_probonite::*;

mod helpers;
use helpers::*;

use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    ThreadPoolBuilder,
};
use revolut::*;

use tfhe::shortint::parameters::*;

#[allow(dead_code)]
const GENERATE_TREE: bool = true;
#[allow(dead_code)]
const EXPORT_FOREST: bool = true;
const NUM_THREADS: usize = 10;
const VERBOSE: bool = true;

fn main() {
    ThreadPoolBuilder::new()
        .num_threads(NUM_THREADS)
        .build_global()
        .unwrap();

    xt_probolut::example_xt_training_probolut_vs_clear();
}
