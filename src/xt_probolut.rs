use std::time::Instant;

use crate::dataset::*;
use crate::model::*;
use crate::probolut::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use revolut::{key, Context, PrivateKey, LUT};

use tfhe::{core_crypto::prelude::LweCiphertext, shortint::parameters::*};

pub fn example_xt_training_probolut() {
    const TREE_DEPTH: u64 = 3;
    const N_CLASSES: u64 = 3;
    const PRECISION_BITS: u64 = 4;
    const DATASET_NAME: &str = "iris_2bits";
    // const DATASET_NAME: &str = "wine_4bits";
    const M: u64 = 10;
    const NUM_EXPERIMENTS: u64 = 1;

    let mut ctx = Context::from(PARAM_MESSAGE_3_CARRY_0);
    let private_key = key(ctx.parameters());
    let public_key = &private_key.public_key;

    let dataset = EncryptedDatasetLut::from_file(
        "data/".to_string() + DATASET_NAME + ".csv",
        &private_key,
        &mut ctx,
        N_CLASSES,
    );

    let (train_dataset, test_dataset) = dataset.split(0.8);
    // let train_size: u64 = train_dataset.n;

    let train_size: u64 = train_dataset.records.len() as u64;
    let test_size: u64 = test_dataset.records.len() as u64;

    // let mut accuracies = Vec::new();
    // let mut times = Vec::new();
    // let mut inference_times = Vec::new();

    for experiment in 0..NUM_EXPERIMENTS {
        // Server trains the forest
        println!("\n --------- Training the forest ---------");
        let start = Instant::now();

        let forest: Vec<(TreeLUT, Vec<Vec<LUT>>)> = (0..M)
            .into_par_iter()
            .map(|i| {
                // let tree = TreeLUT::generate_random_tree(TREE_DEPTH, N_CLASSES, dataset.f, &ctx);
                let classical_tree =
                    Tree::generate_random_tree(TREE_DEPTH, N_CLASSES, train_dataset.f, &ctx);

                let tree = TreeLUT::from_tree(&classical_tree, TREE_DEPTH, N_CLASSES, &ctx);
                // tree.print_tree(private_key, &ctx);

                let luts_samples = (0..train_size)
                    .into_par_iter()
                    .map(|j| {
                        println!("Training Tree[{}] --- sample [{}]", i, j);
                        let query = &train_dataset.records[j as usize];

                        probolut_training(&tree, query, &public_key, &ctx)
                    })
                    .collect::<Vec<_>>();

                (tree, luts_samples)
            })
            .collect();

        let mut result = Vec::new();
        forest.iter().for_each(|(tree, luts_samples)| {
            let mut res = Vec::new();
            luts_samples.iter().for_each(|luts| {
                let mut res_luts = Vec::new();
                luts.iter().for_each(|lut| {
                    res_luts.push(lut.to_array(private_key, &ctx));
                });
                res.push(res_luts);
            });
            result.push(res);
        });

        // println!("{:?}", result);

        let mut summed_counts = Vec::new();

        for i in 0..M {
            let mut tree_cout = Vec::new();
            for j in 0..N_CLASSES {
                let mut class_count = Vec::new();
                for k in 0..2u32.pow(TREE_DEPTH as u32) {
                    let mut count = 0;
                    for l in 0..train_size {
                        count += result[i as usize][l as usize][j as usize][k as usize];
                    }

                    class_count.push(count);
                }
                tree_cout.push(class_count);
            }
            summed_counts.push(tree_cout);
        }

        //     // Server tests the forest
        println!("\n --------- Testing the forest ---------");
        let mut correct = 0;
        let mut total = 0;
        let results: Vec<Vec<LweCiphertext<Vec<u64>>>> = (0..M)
            .into_par_iter()
            .map(|i| {
                let tree = &forest[i as usize].0;
                let mut accs = Vec::new();
                let mut time = 0.0;
                for j in 0..test_size {
                    println!("Testing Tree[{}] --- sample [{}]", i, j);
                    let query = &test_dataset.records[j as usize];
                    let selector = probolut_inference(tree, &query.features, &public_key, &ctx);
                    accs.push(selector);
                }
                accs
            })
            .collect::<Vec<_>>();

        let selected_leaves = results
            .iter()
            .map(|tree_results| {
                tree_results
                    .iter()
                    .map(|result| private_key.decrypt_lwe(result, &ctx))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // println!("{:?}", selected_leaves);
        // println!("Summed counts: {:?}", summed_counts);

        for sample_idx in 0..test_size {
            let mut votes = vec![0; N_CLASSES as usize];

            for tree_idx in 0..M {
                let selected_leaf = selected_leaves[tree_idx as usize][sample_idx as usize];
                for class_idx in 0..N_CLASSES {
                    votes[class_idx as usize] += summed_counts[tree_idx as usize]
                        [class_idx as usize][selected_leaf as usize];
                }
            }

            let (predicted_label, _) = votes.iter().enumerate().max_by_key(|x| x.1).unwrap();
            let mut true_label = 0;
            for c in (0..N_CLASSES) {
                let one_label = private_key.decrypt_lwe(
                    &test_dataset.records[sample_idx as usize].class[c as usize],
                    &ctx,
                );
                if one_label == 1 {
                    true_label = c;
                    break;
                }
            }

            if predicted_label == true_label as usize {
                correct += 1;
            }

            total += 1;
        }

        println!("\n-------- Accuracy -------- ");
        println!("Correct: {}, Total: {}", correct, total);
        println!("\n Accuracy: {} ", correct as f64 / total as f64);
    }
}
