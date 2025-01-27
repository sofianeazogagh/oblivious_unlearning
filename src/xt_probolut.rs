// DATASET
use crate::dataset::*;

// MODEL
use crate::model::*;
use crate::ClearDataset;
use crate::ClearTree;
// REVOLUT
use revolut::{key, Context, PrivateKey, PublicKey, LUT};
type LWE = LweCiphertext<Vec<u64>>;

use tfhe::boolean::prelude::Ciphertext;
use tfhe::integer::ciphertext::BaseRadixCiphertext;
use tfhe::integer::RadixCiphertext;
use tfhe::shortint::backward_compatibility::public_key;
// TFHE
use tfhe::{core_crypto::prelude::LweCiphertext, shortint::parameters::*};

// RAYON
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use std::env::args;
// TIME
#[allow(unused_imports)]
use std::time::Instant;

// HELPERS
use crate::helpers::*;
use crate::{create_progress_bar, finish_progress, inc_progress, make_pb};

// CONSTANTS
use crate::VERBOSE;

/// Train a single tree with a single sample.
///
/// Returns the n_classes LUTs of size 2^depth with a 1 in the slot of the selected leaf for the sample class LUT
///
/// # Example
/// depth = 2 and n_classes = 3
/// If the selected leaf is 3 and the sample class is 2, the returned LUTs will be :
///
/// [[0,0,0,],[0,0,0,1],[0,0,0,0]]
pub fn probolut_for_training(
    tree: &TreeLUT,
    sample: &EncryptedSample,
    public_key: &PublicKey,
    ctx: &Context,
) -> Vec<LUT> {
    let leaf = probolut_inference(tree, &sample.features, public_key, ctx);

    let mut counts = tree.leaves.clone();
    for c in 0..tree.n_classes {
        public_key.blind_array_increment(
            &mut counts[c as usize],
            &leaf,
            &sample.class[c as usize],
            ctx,
        );
    }

    counts
}

/// Inference of a single sample with a single tree
/// Returns the selected leaf index for the sample
pub fn probolut_inference(
    tree: &TreeLUT,
    query: &LUT,
    public_key: &PublicKey,
    ctx: &Context,
) -> LWE {
    // First stage
    let index = tree.root.feature_index;
    let threshold = tree.root.threshold;
    let feature = public_key.lut_extract(&query, index as usize, ctx);
    let b = public_key.lt_scalar(&feature, threshold, ctx);
    // private_key.debug_lwe("b", &b, ctx);

    // Internal Stages
    let mut selector = b.clone();
    // private_key.debug_lwe("selector", &selector, ctx);
    for i in 0..tree.stages.len() {
        let (lut_index, lut_threshold) = &tree.stages[i];
        let feature_index = public_key.blind_array_access(&selector, &lut_index, ctx);
        let threshold = public_key.blind_array_access(&selector, &lut_threshold, ctx);
        let feature = public_key.blind_array_access(&feature_index, &query, ctx);
        let b = public_key.blind_lt_bma_mv(&feature, &threshold, ctx);
        selector = public_key.lwe_mul_add(&b, &selector, 2);
        // private_key.debug_lwe("b", &b, ctx);
        // private_key.debug_lwe("selector_updated", &selector, ctx);
    }
    selector
}

/// Train a single tree with all the samples in the train dataset.
/// Returns a tuple containing:
/// - TreeLUT: The random decision tree generated
/// - Vec<Vec<LUT>>: For each training sample, the n_classes LUTs
///
/// # Example
///
/// For depth = 2, n_classes = 3, train_size = 2.
///
/// If the samples end up in leaves (which are slots) 2 and 1 respectively, and have classes (which are LUTs) 2 and 0 respectively.
///
/// The returned vector will be:
///
/// [
///   [[0,0,0,0], [0,0,0,0], [0,0,1,0]],
///   [[0,1,0,0], [0,0,0,0], [0,0,0,0]]
/// ]
///
fn generate_and_train_single_tree(
    train_dataset: &EncryptedDatasetLut,
    tree_depth: u64,
    tree_idx: u64,
    public_key: &revolut::PublicKey,
    ctx: &revolut::Context,
    mp: &MultiProgress,
) -> (TreeLUT, Vec<Vec<LUT>>) {
    let classical_tree =
        Tree::generate_random_tree(tree_depth, train_dataset.n_classes, train_dataset.f, ctx);
    let tree = TreeLUT::from_tree(&classical_tree, tree_depth, train_dataset.n_classes, ctx);

    let pb = make_pb(mp, train_dataset.records.len() as u64, tree_idx.to_string());

    let luts_samples = (0..train_dataset.records.len() as u64)
        .into_par_iter()
        .map(|j| {
            let result =
                probolut_for_training(&tree, &train_dataset.records[j as usize], public_key, ctx);
            inc_progress!(&pb);
            result
        })
        .collect();

    finish_progress!(pb);
    (tree, luts_samples)
}

/// Train a single tree with all the samples in the train dataset.
/// Returns a tuple containing:
/// - TreeLUT: The random decision tree generated
/// - Vec<Vec<LUT>>: For each training sample, the n_classes LUTs
///
/// # Example
///
/// For depth = 2, n_classes = 3, train_size = 2.
///
/// If the samples end up in leaves (which are slots) 2 and 1 respectively, and have classes (which are LUTs) 2 and 0 respectively.
///
/// The returned vector will be:
///
/// [
///   [[0,0,0,0], [0,0,0,0], [0,0,1,0]],
///   [[0,1,0,0], [0,0,0,0], [0,0,0,0]]
/// ]
///
fn train_single_tree(
    train_dataset: &EncryptedDatasetLut,
    tree: &TreeLUT,
    tree_idx: u64,
    public_key: &revolut::PublicKey,
    ctx: &revolut::Context,
    mp: &MultiProgress,
) -> Vec<Vec<LUT>> {
    let tree_depth = tree.depth;
    let pb = make_pb(mp, train_dataset.records.len() as u64, tree_idx.to_string());

    let luts_samples = (0..train_dataset.records.len() as u64)
        .into_par_iter()
        .map(|j| {
            let result =
                probolut_for_training(&tree, &train_dataset.records[j as usize], public_key, ctx);
            inc_progress!(&pb);
            result
        })
        .collect::<Vec<_>>();

    finish_progress!(pb);
    luts_samples
}

/// Aggregate the counts for each tree
fn aggregate_tree_counts(
    luts_samples: &Vec<Vec<LUT>>,
    tree_depth: u64,
    private_key: &revolut::PrivateKey,
    ctx: &revolut::Context,
) -> Vec<Vec<u64>> {
    let mut tree_counts = Vec::new();
    let n_samples = luts_samples.len() as u64;
    let n_classes = luts_samples[0].len() as u64;
    for j in 0..n_classes {
        let mut class_count = Vec::new();
        for k in 0..2u32.pow(tree_depth as u32) {
            let mut count = 0;
            for l in 0..n_samples {
                count +=
                    luts_samples[l as usize][j as usize].to_array(private_key, ctx)[k as usize];
            }
            class_count.push(count);
        }
        tree_counts.push(class_count);
    }
    tree_counts
}

fn aggregate_tree_counts_radix(
    luts_samples: &Vec<Vec<LUT>>,
    tree_depth: u64,
    public_key: &revolut::PublicKey,
    ctx: &revolut::Context,
) -> Vec<Vec<RadixCiphertext>> {
    let n_samples = luts_samples.len() as u64;
    let n_classes = luts_samples[0].len() as u64;
    let mut output = Vec::new();
    for i in 0..n_samples {
        let mut sample_results = Vec::new();
        for j in 0..n_classes {
            let mut class_results =
                luts_samples[i as usize][j as usize].to_many_lwe(public_key, ctx);
            sample_results.push(class_results);
        }
        output.push(sample_results);
    }

    let mut tree_counts = Vec::new();
    for j in 0..n_classes {
        let mut class_count = Vec::new();
        for k in 0..2u32.pow(tree_depth as u32) {
            let mut counts_to_sum = Vec::new();
            for l in 0..n_samples {
                counts_to_sum.push(output[l as usize][j as usize][k as usize].clone());
            }

            let sum = public_key.lwe_add_into_radix_ciphertext(counts_to_sum, 1, ctx);
            class_count.push(sum);
        }
        tree_counts.push(class_count);
    }
    tree_counts
}

// Test a single tree and return the selected leaf for each sample
fn test_single_tree(
    tree: &TreeLUT,
    test_dataset: &EncryptedDatasetLut,
    tree_idx: u64,
    public_key: &revolut::PublicKey,
    ctx: &revolut::Context,
    mp: &MultiProgress,
) -> Vec<LweCiphertext<Vec<u64>>> {
    let pb = make_pb(mp, test_dataset.records.len() as u64, tree_idx.to_string());

    let results = (0..test_dataset.records.len() as u64)
        .map(|j| {
            let result = probolut_inference(
                tree,
                &test_dataset.records[j as usize].features,
                public_key,
                ctx,
            );
            inc_progress!(&pb);
            result
        })
        .collect();

    finish_progress!(pb);
    results
}

fn get_true_label(
    sample: &EncryptedSample,
    n_classes: u64,
    private_key: &revolut::PrivateKey,
    ctx: &revolut::Context,
) -> u64 {
    let mut one_hot_label = Vec::new();
    for c in 0..n_classes {
        one_hot_label.push(private_key.decrypt_lwe(&sample.class[c as usize], ctx));
    }
    // Get the label from the one-hot encoded label
    let label = one_hot_label.iter().position(|&x| x == 1).unwrap();
    label as u64
}

// Evaluate the forest
fn evaluate_forest(
    selected_leaves: &Vec<Vec<u64>>,
    summed_counts: &Vec<Vec<Vec<u64>>>,
    test_dataset: &EncryptedDatasetLut,
    m: u64,
    n_classes: u64,
    private_key: &PrivateKey,
    ctx: &Context,
) -> (u64, u64) {
    let mut correct = 0;
    let mut total = 0;

    for sample_idx in 0..test_dataset.records.len() as u64 {
        let mut votes = vec![0; n_classes as usize];

        for tree_idx in 0..m {
            let selected_leaf = selected_leaves[tree_idx as usize][sample_idx as usize];
            for class_idx in 0..n_classes {
                votes[class_idx as usize] +=
                    summed_counts[tree_idx as usize][class_idx as usize][selected_leaf as usize];
            }
        }

        let (predicted_label, _) = votes.iter().enumerate().max_by_key(|x| x.1).unwrap();
        let true_label = get_true_label(
            &test_dataset.records[sample_idx as usize],
            n_classes,
            private_key,
            ctx,
        );

        if predicted_label == true_label as usize {
            correct += 1;
        }
        total += 1;
    }

    (correct, total)
}

pub fn example_xt_training_probolut() {
    const TREE_DEPTH: u64 = 3;
    const N_CLASSES: u64 = 3;
    const DATASET_NAME: &str = "iris_2bits";
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
    // let train_size = train_dataset.records.len() as u64;
    // let test_size = test_dataset.records.len() as u64;

    for _ in 0..NUM_EXPERIMENTS {
        if VERBOSE {
            println!("\n --------- Training the forest ---------");
        }
        // Train forest
        let mp = MultiProgress::new();
        let forest: Vec<(TreeLUT, Vec<Vec<LUT>>)> = (0..M)
            .into_par_iter()
            .map(|i| {
                generate_and_train_single_tree(&train_dataset, TREE_DEPTH, i, public_key, &ctx, &mp)
            })
            .collect();

        // Aggregate counts
        let summed_counts: Vec<Vec<Vec<u64>>> = forest
            .iter()
            .map(|(_, luts_samples)| {
                aggregate_tree_counts(luts_samples, TREE_DEPTH, &private_key, &ctx)
            })
            .collect();

        if VERBOSE {
            println!("\n --------- Testing the forest ---------");
        }
        // Test forest
        let results: Vec<Vec<LweCiphertext<Vec<u64>>>> = (0..M)
            .into_par_iter()
            .map(|i| {
                test_single_tree(
                    &forest[i as usize].0,
                    &test_dataset,
                    i,
                    public_key,
                    &ctx,
                    &mp,
                )
            })
            .collect();

        let selected_leaves = results
            .iter()
            .map(|tree_results| {
                tree_results
                    .iter()
                    .map(|result| private_key.decrypt_lwe(result, &ctx))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let (correct, total) = evaluate_forest(
            &selected_leaves,
            &summed_counts,
            &test_dataset,
            M,
            N_CLASSES,
            &private_key,
            &ctx,
        );

        println!("\n-------- Accuracy -------- ");
        println!("Correct: {}, Total: {}", correct, total);
        println!("\n Accuracy: {} ", correct as f64 / total as f64);
    }
}

/// Train a forest on clear dataset, and the same forest on the same encrypted dataset.
/// Log the accuracy and time taken for training and testing.
pub fn example_xt_training_probolut_vs_clear(args: Args) {
    // const TREE_DEPTH: u64 = 4;
    // const DATASET: &str = "iris";
    const N_CLASSES: u64 = 3;
    // const NUM_EXPERIMENTS: u64 = 10;

    // let precisions = vec![2, 3, 4, 5];
    // let num_forests = vec![10, 40, 60];
    // let quantizations = vec!["uni", "non_uni", "fd"];

    let precisions = args.precisions;
    let num_forests = args.num_forests;
    let quantizations = args.quantizations;
    let DATASET = args.dataset_name;
    let depths = args.depths;
    let NUM_EXPERIMENTS = args.number_of_experiments;

    for q in quantizations {
        for b in &precisions {
            for m in &num_forests {
                for d in depths.clone() {
                    let TREE_DEPTH = d;
                    let dataset_name = if q == "" {
                        format!("{}_{}bits", DATASET, b)
                    } else {
                        format!("{}_{}_{}bits", DATASET, q, b)
                    };
                    let M = *m;

                    println!("\n SUMMARY: {}, {} trees ", dataset_name, *m);
                    println!("----------------------------------------");

                    let mut ctx = if *b <= 4 {
                        Context::from(PARAM_MESSAGE_4_CARRY_0)
                    } else {
                        Context::from(PARAM_MESSAGE_5_CARRY_0)
                    };
                    let private_key = key(ctx.parameters());
                    let public_key = &private_key.public_key;

                    for _ in 0..NUM_EXPERIMENTS {
                        let start = Instant::now();
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
                                    clear_dataset.max_features,
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

                        println!("\n-------- Forest on clear data - Accuracy -------- ");
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

                        if VERBOSE {
                            println!("\n --------- Training the forest on private data ---------");
                        }

                        let enc_train_dataset = EncryptedDatasetLut::from_clear_dataset(
                            &train_dataset,
                            &private_key,
                            &mut ctx,
                        );
                        let enc_test_dataset = EncryptedDatasetLut::from_clear_dataset(
                            &test_dataset,
                            &private_key,
                            &mut ctx,
                        );
                        // Train forest

                        let start = Instant::now();
                        let mp = MultiProgress::new();
                        let forest: Vec<(TreeLUT, Vec<Vec<LUT>>)> = (0..M)
                            .into_par_iter()
                            .map(|i| {
                                let treelut = TreeLUT::from_clear_tree(
                                    &clear_forest_trained[i as usize],
                                    &ctx,
                                );

                                treelut.print_tree(private_key, &ctx);

                                let luts_samples = train_single_tree(
                                    &enc_train_dataset,
                                    &treelut,
                                    i,
                                    public_key,
                                    &ctx,
                                    &mp,
                                );

                                (treelut, luts_samples)
                            })
                            .collect();
                        log(
                            &format!(
                                "logs_1st_campaign/{}_{}d_{}m_clear.csv",
                                dataset_name, TREE_DEPTH, M
                            ),
                            &format!(
                                "{},{},{}",
                                accuracy,
                                training_time.as_millis(),
                                testing_time.as_millis()
                            ),
                        );

                        let training_time = Instant::now() - start;

                        // Aggregate counts
                        let summed_counts: Vec<Vec<Vec<u64>>> = forest
                            .iter()
                            .map(|(_, luts_samples)| {
                                aggregate_tree_counts(luts_samples, TREE_DEPTH, &private_key, &ctx)
                            })
                            .collect();

                        // let summed_counts_radix: Vec<Vec<Vec<RadixCiphertext>>> = forest
                        //     .iter()
                        //     .map(|(_, luts_samples)| {
                        //         aggregate_tree_counts_radix(
                        //             luts_samples,
                        //             TREE_DEPTH,
                        //             public_key,
                        //             &ctx,
                        //         )
                        //     })
                        //     .collect();

                        // println!("Clear version : {:?}", summed_counts);
                        // for i in 0..summed_counts_radix.len() {
                        //     for j in 0..summed_counts_radix[i].len() {
                        //         for k in 0..summed_counts_radix[i][j].len() {
                        //             println!(
                        //                 "{:?}",
                        //                 private_key.decrypt_radix_ciphertext(
                        //                     &summed_counts_radix[i][j][k]
                        //                 )
                        //             );
                        //         }
                        //     }
                        // }

                        if VERBOSE {
                            println!("\n --------- Testing the forest on private data ---------");
                        }
                        // Test forest
                        let start = Instant::now();

                        let mp = MultiProgress::new();
                        let results: Vec<Vec<LweCiphertext<Vec<u64>>>> = (0..M)
                            .into_par_iter()
                            .map(|i| {
                                test_single_tree(
                                    &forest[i as usize].0,
                                    &enc_test_dataset,
                                    i,
                                    public_key,
                                    &ctx,
                                    &mp,
                                )
                            })
                            .collect();

                        let selected_leaves = results
                            .iter()
                            .map(|tree_results| {
                                tree_results
                                    .iter()
                                    .map(|result| private_key.decrypt_lwe(result, &ctx))
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();

                        let (correct, total) = evaluate_forest(
                            &selected_leaves,
                            &summed_counts,
                            &enc_test_dataset,
                            M,
                            N_CLASSES,
                            &private_key,
                            &ctx,
                        );

                        let testing_time = Instant::now() - start;

                        println!("\n-------- Forest on encrypted data - Accuracy -------- ");
                        println!("Correct: {}, Total: {}", correct, total);
                        let accuracy = correct as f64 / total as f64;
                        println!("\n Accuracy: {} ", accuracy);

                        log(
                            &format!(
                                "logs_1st_campaign/{}_{}d_{}m_probolut.csv",
                                dataset_name, TREE_DEPTH, M
                            ),
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

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use tfhe::shortint::parameters::{PARAM_MESSAGE_4_CARRY_0, PARAM_MESSAGE_5_CARRY_0};

    #[test]
    fn test_probolut() {
        // Initialize context
        let mut ctx = Context::from(PARAM_MESSAGE_5_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        const TREE_DEPTH: u64 = 5;
        const N_CLASSES: u64 = 3;
        let f = ctx.full_message_modulus() as u64;

        let classical_tree = Tree::generate_random_tree(TREE_DEPTH, N_CLASSES, f, &ctx);

        let tree = TreeLUT::from_tree(&classical_tree, TREE_DEPTH, N_CLASSES, &ctx);

        tree.print_tree(&private_key, &ctx);

        // Create a query
        let feature_vector = vec![31; ctx.full_message_modulus() as usize];
        let class = 1;
        let sample = EncryptedSample::make_encrypted_sample(
            &feature_vector,
            &class,
            N_CLASSES,
            &private_key,
            &mut ctx,
        );

        // Run probolut
        let start = Instant::now();
        let counts = probolut_for_training(&tree, &sample, &public_key, &ctx);
        let end = Instant::now();
        println!("Time taken: {:?}", end.duration_since(start));

        (0..tree.n_classes).for_each(|i| {
            println!(
                "class[{i}]: {:?}",
                counts[i as usize].to_array(&private_key, &ctx)
            );
        });

        // Assert that class 1 has the highest count since we encrypted class=1
        let class_counts: Vec<Vec<u64>> = (0..tree.n_classes)
            .map(|i| counts[i as usize].to_array(&private_key, &ctx))
            .collect();

        let classes = (0..tree.n_classes)
            .map(|i| {
                let mut c_i = vec![0; ctx.full_message_modulus() as usize];
                if i == class {
                    c_i[ctx.full_message_modulus() as usize - 1] = 1;
                }
                c_i
            })
            .collect::<Vec<Vec<u64>>>();

        (0..tree.n_classes).for_each(|i| {
            assert_eq!(
                class_counts[i as usize], classes[i as usize],
                "Class counts don't match expected values for class {}",
                i
            );
        });

        // tree.print_tree(&private_key, &ctx);
    }
}
