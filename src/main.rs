use std::io::Write;
use std::time::Instant;

mod probonite;
use bincode::{config::AllowTrailing, de};
use probonite::*;

mod model;
use model::*;

mod dataset;
use dataset::*;

mod clear_model;
use clear_model::*;

use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    vec,
};
use revolut::{key, Context, PrivateKey, LUT};

use tfhe::{core_crypto::prelude::LweCiphertext, shortint::parameters::*};

type LWE = LweCiphertext<Vec<u64>>;

const GENERATE_TREE: bool = true;
const EXPORT_FOREST: bool = true;

fn test_probonite() {
    const TREE_DEPTH: u64 = 3;
    const N_CLASSES: u64 = 3;

    let mut ctx = Context::from(PARAM_MESSAGE_3_CARRY_0);
    let private_key = key(ctx.parameters());
    let public_key = &private_key.public_key;

    let mut tree: Tree;

    let f = ctx.full_message_modulus() as u64;

    if GENERATE_TREE {
        tree = Tree::generate_random_tree(TREE_DEPTH, N_CLASSES, f, &ctx);
        tree.save_to_file(
            &format!("random_trees/random_tree_{}_{}.json", TREE_DEPTH, N_CLASSES),
            &ctx,
        );
    } else {
        tree = Tree::load_from_file(
            &format!("random_trees/random_tree_{}_{}.json", TREE_DEPTH, N_CLASSES),
            &ctx,
        )
        .unwrap();
    }

    let feature_vector = vec![0, 0, 0, 0, 0, 0, 0, 0];
    let class = 1;

    let query = Query::make_query(&feature_vector, &class, &private_key, &mut ctx);

    let start = Instant::now();
    let luts = probonite(&mut tree, &query, &public_key, &ctx);
    let end = Instant::now();
    println!("Time taken: {:?}", end.duration_since(start));

    let results = luts
        .iter()
        .map(|lut| lut.to_array(&private_key, &ctx)[..N_CLASSES as usize].to_vec())
        .collect::<Vec<_>>();
    let expected = [[0; N_CLASSES as usize]; 2u32.pow(TREE_DEPTH as u32) as usize];
    println!("{:?}", results);
}

fn example_private_training() {
    const TREE_DEPTH: u64 = 4;
    const N_CLASSES: u64 = 3;
    const PRECISION_BITS: u64 = 4;
    const DATASET_NAME: &str = "iris_4bits";
    // const DATASET_NAME: &str = "wine_4bits";
    const M: u64 = 1;
    const NUM_EXPERIMENTS: u64 = 1;

    let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
    let private_key = key(ctx.parameters());
    let public_key = &private_key.public_key;

    let dataset = EncryptedDataset::from_file(
        "data/".to_string() + DATASET_NAME + ".csv",
        &private_key,
        &mut ctx,
    );

    let (train_dataset, test_dataset) = dataset.split(0.8);
    // let train_size: u64 = train_dataset.n;
    let train_size: u64 = 10;
    let test_size: u64 = test_dataset.n;

    let num_experiments = 1;
    let mut accuracies = Vec::new();
    let mut times = Vec::new();
    let mut inference_times = Vec::new();

    for experiment in 0..NUM_EXPERIMENTS {
        // Server trains the forest
        println!("\n --------- Training the forest ---------");
        let start = Instant::now();

        let mut forest: Vec<(Tree, Vec<Vec<LUT>>)> = (0..M)
            .into_par_iter()
            .map(|i| {
                let mut tree = Tree::generate_random_tree(TREE_DEPTH, N_CLASSES, dataset.f, &ctx);
                let start_one_tree = Instant::now();
                let luts_samples: Vec<Vec<LUT>> = (0..train_size)
                    .into_par_iter()
                    .map(|j| {
                        println!("Training Tree[{}] : Record[{}]", i, j);
                        // println!("Sample features : {:?}", &train_dataset.records[j as usize].features.to_array(&private_key, &ctx));
                        // println!("Sample class : {:?}", private_key.decrypt_lwe(&train_dataset.records[j as usize].class, &ctx));
                        let query = &train_dataset.records[j as usize];
                        probonite(&tree, query, &public_key, &ctx)
                    })
                    .collect();

                // tree.sum_samples_luts_counts(&luts_samples, &public_key);
                let elapsed_one_tree = Instant::now() - start_one_tree;
                println!("Time taken for Tree[{i}] : {:?}", elapsed_one_tree);
                (tree, luts_samples)
            })
            .collect();

        // Client decrypts the counts
        let decrypted_counts = forest
            .iter()
            .map(|(tree, luts_samples)| {
                decrypt_counts(
                    luts_samples.clone(),
                    N_CLASSES,
                    TREE_DEPTH,
                    &private_key,
                    &ctx,
                )
            })
            .collect::<Vec<_>>();

        // // Sum the LUTs counts for each tree
        // forest.iter_mut().for_each(|(tree, luts_samples)| {
        //     tree.sum_samples_luts_counts(luts_samples, &public_key);
        // });

        // Client assign labels to the leaves. assigned_labels[i][j] is the label assigned to the j-th leaf of the i-th tree
        let mut assigned_labels = Vec::new();
        for i in 0..M {
            let mut labels = Vec::new();
            for j in 0..2u32.pow(TREE_DEPTH as u32) {
                let mut label = 0; // default label. Might have an impact on the accuracy
                for k in 1..N_CLASSES {
                    if decrypted_counts[i as usize][j as usize][k as usize]
                        > decrypted_counts[i as usize][j as usize][label as usize]
                    {
                        label = k;
                    }
                }
                labels.push(label);
            }
            assigned_labels.push(labels);
        }

        let end = Instant::now() - start;

        println!(
            "[PARAMETERS] :
        Precision bits: {}, \
        Dataset size: {}, \
        Dataset dimension: {}, \
        Dataset classes: {}, \
        Tree depth: {}, \
        Number of trees: {}",
            PRECISION_BITS, train_dataset.n, train_dataset.f, N_CLASSES, TREE_DEPTH, M
        );
        println!("[SUMMARY] : Forest built in {:?}", end);

        times.push(end.as_secs_f64());

        if EXPORT_FOREST {
            for i in 0..M {
                forest[i as usize].0.save_to_file(
                    &format!(
                        "{DATASET_NAME}_forest/{TREE_DEPTH}_depth/experiment_{}/tree_{}.json",
                        experiment, i
                    ),
                    &ctx,
                );
            }
        }

        // Server tests the forest
        println!("\n --------- Testing the forest ---------");
        let mut correct = 0;
        let mut total = 0;
        let results: Vec<Vec<(Vec<LWE>, f64)>> = (0..M)
            .into_par_iter()
            .map(|i| {
                (0..test_size)
                    .into_par_iter()
                    .map(|j| {
                        let query = &test_dataset.records[j as usize];
                        println!("Testing Tree[{}] : Record[{}]", i, j);
                        probonite_inference(&forest[i as usize].0, query, &public_key, &ctx)
                    })
                    .collect()
            })
            .collect::<Vec<_>>();

        let accumulators: Vec<Vec<Vec<LWE>>> = results
            .iter()
            .map(|tree_results| tree_results.iter().map(|(accs, _)| accs.clone()).collect())
            .collect();

        let avg_inference_time_per_sample_per_tree: f64 = results
            .iter()
            .flat_map(|tree_results| tree_results.iter().map(|(_, time)| time))
            .sum::<f64>()
            / (M * test_size) as f64;
        inference_times.push(avg_inference_time_per_sample_per_tree);

        let decrypted_accumulators: Vec<Vec<Vec<u64>>> = accumulators
            .iter()
            .map(|accs| {
                accs.iter()
                    .map(|acc| {
                        acc.iter()
                            .map(|lwe| private_key.decrypt_lwe(lwe, &ctx))
                            .collect()
                    })
                    .collect()
            })
            .collect::<Vec<_>>();

        println!("{:?}", decrypted_accumulators);

        for sample_idx in 0..test_size {
            let mut votes = vec![0; N_CLASSES as usize];

            for tree_idx in 0..M {
                let sample_leaves = &decrypted_accumulators[tree_idx as usize][sample_idx as usize];
                let mut selected_leaf = 0;
                let mut found = false;
                for i in 0..2u32.pow(TREE_DEPTH as u32) {
                    if sample_leaves[i as usize] == 1 {
                        selected_leaf = i;
                        found = true;
                        break;
                    }
                }

                let predicted_label = assigned_labels[tree_idx as usize][selected_leaf as usize];
                votes[predicted_label as usize] += 1;
            }

            let (predicted_label, _) = votes.iter().enumerate().max_by_key(|x| x.1).unwrap();
            let true_label =
                private_key.decrypt_lwe(&test_dataset.records[sample_idx as usize].class, &ctx);
            if predicted_label == true_label as usize {
                correct += 1;
            }
            total += 1;
        }

        println!("\n-------- Accuracy -------- ");
        println!("Correct: {}, Total: {}", correct, total);
        let accuracy = correct as f64 / total as f64;
        println!("\n Accuracy: {} ", accuracy);
        accuracies.push(accuracy);

        log(
            &format!("logs/{}_{}_{}.csv", DATASET_NAME, TREE_DEPTH, M),
            &format!("{},{}", correct as f64 / total as f64, end.as_secs_f64()),
        );
    }

    let sum_acc: f64 = accuracies.iter().sum();
    let avg_acc = sum_acc / num_experiments as f64;
    let sum_time: f64 = times.iter().sum();
    let avg_time = sum_time / num_experiments as f64;
    let sum_inference_time: f64 = inference_times.iter().sum();
    let avg_inference_time = sum_inference_time / num_experiments as f64;
    println!("\n-------- Average Accuracy: {} -------- ", avg_acc);
    println!("\n-------- Average Time: {} -------- ", avg_time);
    println!(
        "\n-------- Average Inference Time: {} -------- ",
        avg_inference_time
    );
}

pub fn log(filepath: &str, message: &str) {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(filepath)
        .unwrap();
    writeln!(file, "{}", message).unwrap();
}

// Decrypt the counts yielded by one tree for multiple samples
pub fn decrypt_counts(
    lut_samples: Vec<Vec<LUT>>,
    n_classes: u64,
    tree_depth: u64,
    private_key: &PrivateKey,
    ctx: &Context,
) -> Vec<Vec<u64>> {
    let mut results = Vec::new();
    for lut_vec in lut_samples.iter() {
        let mut rec_vec = Vec::new();
        for lut in lut_vec.iter() {
            rec_vec.push(lut.to_array(private_key, &ctx)[..n_classes as usize].to_vec());
        }
        results.push(rec_vec);
    }

    let mut result_counts: Vec<Vec<u64>> = Vec::new();
    for i in 0..results[0].len() {
        result_counts.push(Vec::new());
    }

    for i in 0..2u32.pow(tree_depth as u32) {
        let mut vec = vec![0; n_classes as usize];
        for j in 0..results.len() {
            for k in 0..3 {
                vec[k as usize] += results[j][i as usize][k as usize];
            }
        }
        result_counts[i as usize] = vec.to_vec();
    }

    result_counts
}

fn example_clear_training() {
    let mut clear_dataset = ClearDataset::from_file("data/iris_2bits.csv".to_string());

    // get the train and test datasets
    let (train_dataset, test_dataset) = clear_dataset.split(0.8);
    let column_domains = clear_dataset.column_domains.clone();
    let n_classes = column_domains[column_domains.len() - 1].1 + 1;
    let n_trees = 10;

    let mut forest: Vec<ClearTree> = Vec::new();

    // Training the forest
    for i in 0..n_trees {
        let mut clear_tree =
            generate_clear_random_tree(4, n_classes, column_domains.clone(), train_dataset.f);
        train_dataset.records.iter().for_each(|record| {
            clear_tree.update_statistic(record.to_vec());
        });

        clear_tree.assign_label_to_leafs();
        clear_tree.print_tree();
        forest.push(clear_tree);
    }

    // Testing the forest
    let mut correct = 0;
    let mut total = 0;
    for record in test_dataset.records.iter() {
        let mut votes = vec![0; n_classes as usize];
        for tree in forest.iter() {
            let label = tree.infer_label(record.to_vec());
            votes[label as usize] += 1;
        }
        let predicted_label = votes.iter().enumerate().max_by_key(|x| x.1).unwrap().0 as u64;
        let true_label = record[record.len() - 1];
        if predicted_label == true_label {
            correct += 1;
        }
        total += 1;
    }
    print!("\nNumber of trees : {}\n", n_trees);
    print!("Size of training dataset : {}\n", train_dataset.n);
    print!("Size of testing dataset : {}\n", test_dataset.n);
    println!("Accuracy: {}", correct as f64 / total as f64);
}

fn main() {
    // example_clear_training();
    example_private_training();
    // test_probonite();
}
