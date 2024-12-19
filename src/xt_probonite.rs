use crate::dataset::*;
use crate::model::*;
use crate::probonite::*;
use rayon::prelude::*;
use revolut::*;
use std::io::Write;
use std::time::Instant;
use tfhe::core_crypto::prelude::LweCiphertext;
use tfhe::shortint::parameters::*;

type LWE = LweCiphertext<Vec<u64>>;

const EXPORT_FOREST: bool = true;

fn example_private_training() {
    const TREE_DEPTH: u64 = 4;
    const N_CLASSES: u64 = 3;
    const PRECISION_BITS: u64 = 4;
    const DATASET_NAME: &str = "iris_2bits";
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
