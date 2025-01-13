use std::time::Instant;
// DATASET
use crate::dataset::*;

// MODEL
use crate::model::*;
use crate::ClearDataset;
use crate::ClearTree;
// REVOLUT
use revolut::{key, Context, PrivateKey, PublicKey, LUT};
type LWE = LweCiphertext<Vec<u64>>;

// TFHE
use tfhe::{core_crypto::prelude::LweCiphertext, shortint::parameters::*};

// RAYON
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use std::env::args;
// TIME
// HELPERS
use crate::helpers::*;
use crate::{create_progress_bar, finish_progress, inc_progress, make_pb};

// CONSTANTS
use crate::VERBOSE;

pub fn benchmark_clear_training(args: Args) {
    // const TREE_DEPTH: u64 = 4;
    // const DATASET: &str = "iris";
    const N_CLASSES: u64 = 3;
    // const NUM_EXPERIMENTS: u64 = 10;

    // let precisions = vec![2, 3, 4, 5];
    // let num_forests = vec![10, 40, 60];
    // let quantizations = vec!["", "non_uni", "fd"];

    let precisions = args.precisions;
    let num_forests = args.num_forests;
    let quantizations = args.quantizations;
    let DATASET = args.dataset_name;
    let NUM_EXPERIMENTS = args.number_of_experiments;
    let depths = args.depths;

    for q in quantizations {
        for b in &precisions {
            for m in &num_forests {
                for d in depths.clone() {
                    let TREE_DEPTH = d;

                    let mut dataset_name: String;

                    if q == "original" {
                        dataset_name = DATASET.clone();
                    } else {
                        dataset_name = format!("{}_{}_{}bits", DATASET, q, b);
                    }

                    let M = *m;

                    let mut ctx = if *b <= 4 {
                        Context::from(PARAM_MESSAGE_4_CARRY_0)
                    } else {
                        Context::from(PARAM_MESSAGE_5_CARRY_0)
                    };
                    let private_key = key(ctx.parameters());
                    let public_key = &private_key.public_key;

                    for _ in 0..NUM_EXPERIMENTS {
                        let start = Instant::now();
                        println!("{}", dataset_name);
                        let clear_dataset =
                            ClearDataset::from_file("data/".to_string() + &dataset_name + ".csv");
                        let (train_dataset, test_dataset) = clear_dataset.split(0.8);

                        if VERBOSE {
                            println!("\n --------- Training the clear forest ---------");
                        }
                        let mp = MultiProgress::new();
                        let mut clear_forest_trained = (0..M)
                            .into_par_iter()
                            .map(|i| {
                                let mut clear_tree = ClearTree::generate_clear_random_tree(
                                    TREE_DEPTH,
                                    clear_dataset.n_classes,
                                    clear_dataset.features_domain,
                                    clear_dataset.f,
                                );

                                // clear_tree.print();
                                let pb =
                                    make_pb(&mp, train_dataset.records.len() as u64, i.to_string());

                                train_dataset.records.iter().for_each(|sample| {
                                    clear_tree.update_statistic(sample);
                                    inc_progress!(&pb);
                                });

                                finish_progress!(pb);
                                clear_tree
                            })
                            .collect::<Vec<_>>();

                        let training_time = Instant::now() - start;
                        let start = Instant::now();

                        if VERBOSE {
                            println!("\n --------- Testing the clear forest ---------");
                        }
                        let mp = MultiProgress::new();
                        let pb = make_pb(&mp, test_dataset.records.len() as u64, "_");

                        let mut correct = 0;
                        let mut total = 0;
                        test_dataset.records.iter().for_each(|sample| {
                            let mut votes = vec![0; N_CLASSES as usize];
                            clear_forest_trained.iter_mut().for_each(|tree| {
                                let leaf = tree.infer(sample);
                                for c in 0..N_CLASSES {
                                    votes[c as usize] += leaf.counts[c as usize];
                                }
                            });

                            let (predicted_label, _) =
                                votes.iter().enumerate().max_by_key(|x| x.1).unwrap();
                            let true_label = sample.class;
                            if predicted_label == true_label as usize {
                                correct += 1;
                            }
                            total += 1;
                            inc_progress!(&pb);
                        });

                        let testing_time = Instant::now() - start;
                        finish_progress!(pb);

                        println!("\n-------- Clear Forest Accuracy -------- ");
                        println!("Correct: {}, Total: {}", correct, total);
                        let accuracy = correct as f64 / total as f64;
                        println!("\n Accuracy: {} ", accuracy);

                        log(
                            &format!("logs/{}_{}d_{}m_clear.csv", dataset_name, TREE_DEPTH, M),
                            &format!(
                                "{},{},{}",
                                accuracy,
                                training_time.as_millis(),
                                testing_time.as_millis()
                            ),
                        );
                    }
                }
            }
        }
    }
}
