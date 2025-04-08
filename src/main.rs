#![allow(warnings)]

use std::io::Write;
use std::time::Instant;

mod probonite;
use probonite::*;

use bincode::{config::AllowTrailing, de};

mod model;
use model::*;

// mod xt_probolut;
// use xt_probolut::*;

mod dataset;
use dataset::*;

mod clear_model;
use clear_model::*;

mod xt_probonite;
use xt_probonite::*;

mod helpers;
use helpers::*;

mod clear_training;
use clear_training::*;

use clap::Parser;
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    ThreadPoolBuilder,
};
use revolut::*;

use tfhe::shortint::parameters::*;

use std::env;

#[allow(dead_code)]
const GENERATE_TREE: bool = true;
#[allow(dead_code)]
const EXPORT_FOREST: bool = true;
const NUM_THREADS: usize = 10;
const VERBOSE: bool = true;

// mod xt_probolut_radix;
// use xt_probolut_radix::*;

fn main() {
    ThreadPoolBuilder::new()
        .num_threads(NUM_THREADS)
        .build_global()
        .unwrap();

    let args = Args::parse();

    // xt_probolut::example_xt_training_probolut_vs_clear(args);

    // xt_probolut_radix::train_test_probolut_vs_clear(args);
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    use revolut::*;
    use tfhe::shortint::parameters::PARAM_MESSAGE_4_CARRY_0;

    #[test]
    fn test_cost_probolut() {
        let params = PARAM_MESSAGE_4_CARRY_0;
        let mut ctx = Context::from(params);
        let private_key = key(params);
        let public_key = &private_key.public_key;

        let d = 4;
        let a = private_key.allocate_and_encrypt_lwe(0, &mut ctx);
        let b = private_key.allocate_and_encrypt_lwe(1, &mut ctx);
        let lut = LUT::from_vec_trivially(&vec![1; ctx.full_message_modulus()], &ctx);

        let start = Instant::now();
        for i in 0..d {
            // 1 BR for getting the feature

            if i == 0 {
                let feature = public_key.lut_extract(&lut, 1, &ctx);
            } else {
                let feature = public_key.blind_array_access(&a, &lut, &ctx);
            }
            // 1 BMA MV for comparison
            let c = public_key.blind_lt_bma_mv(&a, &b, &ctx);

            // 2 BR + 2 SE (1 for index and 1 for threshold)
            let _ = public_key.blind_array_access(&c, &lut, &ctx);
            let _ = public_key.blind_array_access(&c, &lut, &ctx);
        }

        let end = Instant::now();
        println!("Time taken cost probolut: {:?}", end.duration_since(start));
    }

    #[test]
    fn test_cost_comp_free() {
        let params = PARAM_MESSAGE_4_CARRY_0;
        let mut ctx = Context::from(params);
        let private_key = key(params);
        let public_key = &private_key.public_key;

        // 2^{d+1} - 1 + d SE + d BR + KS

        let d = 4;
        let a = private_key.allocate_and_encrypt_lwe(0, &mut ctx);
        let b = private_key.allocate_and_encrypt_lwe(1, &mut ctx);
        let mut lut = LUT::from_lwe(&b, public_key, &ctx);

        let vec = vec![0; 2_u64.pow(d) as usize];
        let many_lwe: Vec<LWE> = vec
            .into_iter()
            .map(|x| private_key.allocate_and_encrypt_lwe(x, &mut ctx))
            .collect();

        let start = Instant::now();

        for i in 0..d {
            // i SE for getting the comparison result for each nodes
            for j in 0..2_u64.pow(i) {
                let _ = public_key.glwe_extract(&lut.0, 1, &ctx);
            }

            // Compiled tree
            // 1 KS
            let stage = LUT::from_vec_of_lwe(&many_lwe[0..2_u64.pow(i) as usize], public_key, &ctx);

            // 1 BR for evaluating the stage
            let _ = public_key.blind_array_access(&a, &stage, &ctx);
        }

        let end = Instant::now();
        println!("Time taken cost comp free: {:?}", end.duration_since(start));
    }
}
