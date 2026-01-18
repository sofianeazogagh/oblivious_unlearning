use std::fs::File;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::time::Instant;

use rand::Rng;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use serde_json::json;
use serde_json::Value;

use crate::dataset::EncryptedDataset;
use crate::timing::TimingCollector;
use crate::tree::Tree;
use crate::utils_maj::Majority;
use crate::{Context, LWE, PublicKey, PrivateKey, RLWE};
use crate::{FOLDER, NUM_THREADS, PRE_SEEDED, SEED};

// Import the time_operation macro
#[allow(unused_imports)]
use crate::time_operation;
pub struct Forest {
    pub trees: Vec<Tree>,
}



impl Forest {
    pub fn new(
        n_trees: u64,
        depth: u64,
        n_classes: u64,
        f: u64,
        public_key: &PublicKey,
        ctx: &Context,
    ) -> Self {
        let mut trees = Vec::new();
        for _ in 0..n_trees {
            if PRE_SEEDED {
                let tree: Tree =
                    Tree::new_random_tree_with_seed(depth, n_classes, f, public_key, ctx, SEED);
                trees.push(tree);
            } else {
                let seed = rand::thread_rng().gen_range(0..u64::MAX);
                let mut file = OpenOptions::new()
                    .write(true)
                    .append(true)
                    .create(true)
                    .open(format!("{}/seed.csv", FOLDER))
                    .unwrap();
                writeln!(file, "{}", seed).unwrap();
                let tree: Tree =
                    Tree::new_random_tree_with_seed(depth, n_classes, f, public_key, ctx, seed);
                trees.push(tree);
            }
        }
        Self { trees }
    }

    pub fn train(&mut self, dataset: &EncryptedDataset, public_key: &PublicKey, ctx: &Context, timing: &TimingCollector) {
        println!("Dataset size: {}", dataset.records.len());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(NUM_THREADS)
            .build()
            .unwrap();

        time_operation!(timing, "Total trees training", {
            pool.install(|| {
                self.trees.par_iter_mut().for_each(|tree| {
                    tree.train(dataset, public_key, ctx, timing);
                });
            });
        });

        time_operation!(timing, "Total forest majority", {
            pool.install(|| {
                self.trees.par_iter_mut().for_each(|tree| {
                    tree.leaves_majority(public_key, ctx);
                });
            });
        });
    }

    pub fn train_fully_oblivious(
        &mut self,
        dataset: &EncryptedDataset,
        public_key: &PublicKey,
        ctx: &Context,
        timing: &TimingCollector,
    ) -> Vec<LWE> {
        println!("Dataset size: {}", dataset.records.len());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(NUM_THREADS)
            .build()
            .unwrap();
        let mut predictions = Vec::new();
        let mut res = Vec::new();
        for sample in dataset.records.iter() {
            for tree in self.trees.iter_mut() {
                let r = tree.train_fully_oblivious(sample, public_key, ctx, timing);
                res.push(r);
            }
            // Perform a majority vote on the results
            let result = time_operation!(timing, "Forest majority", {
                public_key.blind_majority_extra(&res, ctx, self.trees[0].n_classes, timing)
            });
            predictions.push(result);
        }
        predictions
    }

    pub fn test(&self, sample_features: &Vec<RLWE>, public_key: &PublicKey, ctx: &Context, timing: &TimingCollector) -> LWE {
        let mut res = Vec::new();
        time_operation!(timing, "Total forest evaluation", {
            for tree in self.trees.iter() {
                res.push(tree.test(sample_features, public_key, ctx));
            }
        });

        let result = time_operation!(timing, "Majority vote", {
            public_key.blind_majority_extra(&res, ctx, self.trees[0].n_classes, timing)
        });
        result
    }

    pub fn test_index(
        &self,
        sample_features: &Vec<RLWE>,
        public_key: &PublicKey,
        ctx: &Context,
        i: u64,
        timing: &TimingCollector,
    ) -> LWE {
        let mut res = Vec::new();
        time_operation!(timing, "Total forest evaluation", {
            for tree in self.trees.iter() {
                res.push(tree.test(sample_features, public_key, ctx));
            }
        });

        let result = time_operation!(timing, "Majority vote", {
            public_key.blind_majority_extra_index(&res, ctx, self.trees[0].n_classes, i)
        });
        result
    }
    pub fn print(&self, private_key: &PrivateKey, ctx: &Context) {
        for tree in self.trees.iter() {
            tree.print_tree(private_key, ctx);
            // print the final leaves
            println!("Final leaves: [");
            for (i, leaf) in tree.final_leaves.iter().enumerate() {
                // More fancy print
                if i == tree.final_leaves.len() - 1 {
                    print!("{}", private_key.decrypt_lwe(leaf, ctx));
                } else {
                    print!("{},", private_key.decrypt_lwe(leaf, ctx));
                }
            }
            println!("]");
        }
    }
}

mod tests {
    use std::time::{Duration, Instant};

    use tfhe::shortint::parameters::PARAM_MESSAGE_4_CARRY_0;
    use revolut::{key, Context, PrivateKey, PublicKey};
    use crate::clear::ClearForest;
    use crate::dataset::{ClearDataset, EncryptedDataset};
    use crate::forest::Forest;
    use crate::timing::TimingCollector;

    use crate::FOLDER;


    #[test]
    fn test_train_forest() {
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        for dataset_name in ["cancer", "wine", "iris"] {
            println!("Dataset: {}", dataset_name);
            let mut n_classes = 3;
            let mut f = 4;
            if dataset_name == "iris" {
                n_classes = 3;
                f = 4;
            }
            if dataset_name == "wine" {
                n_classes = 3;
                f = 13;
            }
            if dataset_name == "cancer" {
                n_classes = 2;
                f = 30;
            }
            let dataset_path = format!("data/{}-uci/{}-sample.csv", dataset_name, dataset_name);
            let dataset = EncryptedDataset::from_file(
                dataset_path.to_string(),
                &private_key,
                &mut ctx,
                n_classes,
            );

            let timing = TimingCollector::new();
            let start = Instant::now();
            let mut forest = Forest::new(1, 4, dataset.n_classes, dataset.f, &public_key, &ctx);
            forest.train(&dataset, &public_key, &ctx, &timing);
            let duration = start.elapsed();
            println!("Time taken: {:?}", duration);
            forest.print(&private_key, &ctx);
        }
    }

    #[test]
    fn test_forest_inference_function() {
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        let dataset = EncryptedDataset::from_file(
            "data/iris-uci/iris-sample.csv".to_string(),
            &private_key,
            &mut ctx,
            3,
        );

        let timing = TimingCollector::new();
        let n_trees = 1;
        let depth = 3;
        let mut forest = Forest::new(
            n_trees,
            depth,
            dataset.n_classes,
            dataset.f,
            &public_key,
            &ctx,
        );
        forest.train(&dataset, &public_key, &ctx, &timing);
        let filepath = format!(
            "./src/comp_free/test_iris_forest_{}m_{}d.json",
            n_trees, depth
        );
        forest.save_to_file(&filepath, &private_key, &ctx);

        forest.print(&private_key, &ctx);

        let sample_features = dataset.records[0].features.clone();
        let sample_class = dataset.records[0].class.clone();

        let class_one_hot = private_key.decrypt_lwe_vector(&sample_class, &ctx);

        let ground_truth = class_one_hot.iter().position(|&x| x == 1).unwrap() as u64;

        let result = forest.test(&sample_features, &public_key, &ctx, &timing);

        println!("Sample class: {:?}", ground_truth);
        println!("Test result: {:?}", private_key.decrypt_lwe(&result, &ctx));
    }

    #[test]
    fn test_forest_train_and_inference() {
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        // DATASET
        let dataset = EncryptedDataset::from_file(
            "data/iris-uci/iris.csv".to_string(),
            &private_key,
            &mut ctx,
            3,
        );

        let (train_dataset, test_dataset) = dataset.split(0.8);

        // TRAIN FOREST
        let n_trees = 64;
        let depth = 4;
        // let mut forest = Forest::new(
        //     n_trees,
        //     depth,
        //     dataset.n_classes,
        //     dataset.f,
        //     &public_key,
        //     &ctx,
        // );
        let timing = TimingCollector::new();
        let forest_path = "./src/comp_free/best_iris_64_4_0.97.json";
        let mut forest = Forest::new_from_file(forest_path, &public_key, &ctx);

        let start_train = Instant::now();
        forest.train(&train_dataset, &public_key, &ctx, &timing);
        let duration_train = start_train.elapsed();

        let filepath = format!("{}/forest_{}m_{}d.json", FOLDER, n_trees, depth);
        forest.save_to_file(&filepath, &private_key, &ctx);

        forest.print(&private_key, &ctx);

        // TEST FOREST
        let mut correct = 0;
        let mut duration_test_total = Duration::new(0, 0);
        for sample in test_dataset.records.iter() {
            let sample_features = sample.features.clone();
            let sample_class = sample.class.clone();
            let class_one_hot = private_key.decrypt_lwe_vector(&sample_class, &ctx);
            let ground_truth = class_one_hot.iter().position(|&x| x == 1).unwrap() as u64;
            // INFERENCE
            let start_test = Instant::now();
            let result = forest.test(&sample_features, &public_key, &ctx, &timing);
            let duration_test = start_test.elapsed();
            duration_test_total += duration_test;
            if ground_truth == private_key.decrypt_lwe(&result, &ctx) {
                correct += 1;
            }
        }

        let average_duration_test = Duration::from_secs_f64(
            duration_test_total.as_secs_f64() / test_dataset.records.len() as f64,
        );

        let accuracy = correct as f64 / test_dataset.records.len() as f64;
        println!("Accuracy: {:?}", accuracy);

        // Write data to perf.csv
        forest.save_perf_to_file(
            &format!("{}/perf.csv", FOLDER),
            duration_train,
            average_duration_test,
            "iris",
            64,
            4,
            accuracy,
        );
    }

    #[test]
    fn test_train_and_inference() {
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        let num_trials = 1;
        for i in 0..num_trials {
            // DATASET
            let dataset = EncryptedDataset::from_file(
                "data/iris-uci/iris.csv".to_string(),
                &private_key,
                &mut ctx,
                3,
            );

            let (train_dataset, test_dataset) = dataset.split(0.8);

            // TRAIN FOREST
            let n_trees = 1;
            let depth = 4;
            let timing = TimingCollector::new();
            let mut forest = Forest::new(
                n_trees,
                depth,
                dataset.n_classes,
                dataset.f,
                &public_key,
                &ctx,
            );
            println!("Training forest...");
            let start_train = Instant::now();
            forest.train(&train_dataset, &public_key, &ctx, &timing);
            let duration_train = start_train.elapsed();

            println!("Training time: {:?}", duration_train);

            // let filepath = format!("{}/forests/forest_{}.json", FOLDER, i);
            // forest.save_to_file(&filepath, &private_key, &ctx);

            // forest.print(&private_key, &ctx);

            // TEST FOREST
            let mut correct = 0;
            let mut duration_test_total = Duration::new(0, 0);
            for sample in test_dataset.records.iter() {
                let class_one_hot = private_key.decrypt_lwe_vector(&sample.class, &ctx);
                let ground_truth = class_one_hot.iter().position(|&x| x == 1).unwrap() as u64;
                // INFERENCE
                let start_test = Instant::now();
                let result = forest.test(&sample.features, &public_key, &ctx, &timing);
                let duration_test = start_test.elapsed();
                duration_test_total += duration_test;
                println!("Ground truth: {:?}", ground_truth);
                println!("Result: {:?}", private_key.decrypt_lwe(&result, &ctx));
                if ground_truth == private_key.decrypt_lwe(&result, &ctx) {
                    correct += 1;
                }
            }

            let average_duration_test = Duration::from_secs_f64(
                duration_test_total.as_secs_f64() / test_dataset.records.len() as f64,
            );

            let accuracy = correct as f64 / test_dataset.records.len() as f64;
            println!("Accuracy: {:?}", accuracy);

            // Write data to perf.csv
            // forest.save_perf_to_file(
            //     &format!("{}/perf.csv", FOLDER),
            //     duration_train,
            //     average_duration_test,
            //     "iris",
            //     64,
            //     4,
            //     accuracy,
            // );
        }
    }

    #[test]
    fn test_bench_best() {
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        if !std::path::Path::new(FOLDER).exists() {
            std::fs::create_dir_all(FOLDER).unwrap();
        }
        for num_trees in [8, 16] {
            let num_trials = 10;
            for i in 0..num_trials {
                println!("--------------------------------");
                println!("------------- Trial: {}", i);
                println!("--------------------------------");

                // let dataset_name = "cancer";
                // let dataset_name = "adult";
                // let dataset_name = "wine";
                // let dataset_name = "cancer";
                let dataset_name = "iris";
                let dataset_path = format!("data/{}-uci/{}.csv", dataset_name, dataset_name);

                // Dataset
                let dataset = ClearDataset::from_file(dataset_path.to_string());

                // Base case is for iris
                let mut n_classes = 3;
                let mut f = 4;
                let max_features = dataset.max_features;
                let mut split_percentage = 0.8;
                if dataset_name == "adult" {
                    n_classes = 2;
                    f = 105;
                }

                if dataset_name == "wine" {
                    n_classes = 3;
                    f = 13;
                }

                if dataset_name == "cancer" {
                    n_classes = 2;
                    f = 30;
                }

                let (train_dataset_clear, test_dataset_clear) = dataset.split(split_percentage);

                // let num_trees = 32;
                let depth = 4;

                // FIND BEST MODEL
                let num_trials_best_model = 1;
                let mut best_accuracy = 0.0;
                let mut best_model =
                    ClearForest::new_random_forest(num_trees, depth, n_classes, max_features, f);

                for _ in 0..num_trials_best_model {
                    let mut forest = ClearForest::new_random_forest(
                        num_trees,
                        depth,
                        n_classes,
                        max_features,
                        f,
                    );
                    forest.train(&train_dataset_clear, 0);
                    let accuracy = forest.evaluate(&test_dataset_clear);
                    if accuracy > best_accuracy {
                        best_accuracy = accuracy;
                        best_model = forest;
                    }
                }

                println!("Best accuracy: {:?}", best_accuracy);

                let dir_path = format!("{}/{}/{}", FOLDER, dataset_name, num_trees);

                if !std::path::Path::new(&dir_path).exists() {
                    std::fs::create_dir_all(&dir_path).expect("Failed to create directory");
                }
                let filepath_clear_forest = format!(
                    "{}/best_{}_{}_{}_{:.2}_{}.json",
                    dir_path, dataset_name, num_trees, depth, best_accuracy, i
                );

                best_model.save_to_file(&filepath_clear_forest);

                let train_dataset_encrypted = EncryptedDataset::from_clear_dataset(
                    &train_dataset_clear,
                    &private_key,
                    &mut ctx,
                );

                let test_dataset_encrypted = EncryptedDataset::from_clear_dataset(
                    &test_dataset_clear,
                    &private_key,
                    &mut ctx,
                );

                // TRAIN FOREST
                let timing = TimingCollector::new();
                let mut forest =
                    Forest::new_from_file(filepath_clear_forest.as_str(), &public_key, &ctx);

                let start_train = Instant::now();
                forest.train(&train_dataset_encrypted, &public_key, &ctx, &timing);
                let duration_train = start_train.elapsed();

                // Create the directory "forests" if it does not exist
                let forests_dir = format!("{}/forests", dir_path);
                if !std::path::Path::new(&forests_dir).exists() {
                    std::fs::create_dir_all(&forests_dir)
                        .expect("Failed to create forests directory");
                }
                let filepath_forest = format!("{}/forest_{}.json", forests_dir, i);
                forest.save_to_file(&filepath_forest, &private_key, &ctx);

                // TEST FOREST
                let mut correct = 0;
                let mut abstention = 0;
                let mut duration_test_total = Duration::new(0, 0);
                for (i, sample) in test_dataset_encrypted.records.iter().enumerate() {
                    let class_one_hot = private_key.decrypt_lwe_vector(&sample.class, &ctx);
                    let ground_truth = class_one_hot.iter().position(|&x| x == 1).unwrap() as u64;

                    // INFERENCE
                    let start_test = Instant::now();
                    let result = forest.test_index(&sample.features, &public_key, &ctx, i as u64, &timing);
                    let duration_test = start_test.elapsed();
                    duration_test_total += duration_test;
                    let result_clear = private_key.decrypt_lwe(&result, &ctx);
                    println!("Ground truth: {:?}", ground_truth);
                    println!("Result: {:?}", result_clear);

                    // If the result is dataset.n_classes we increase the abstention
                    if result_clear == dataset.n_classes {
                        abstention += 1;
                    }

                    // If the result is good we increase the accuracy
                    if ground_truth == result_clear {
                        correct += 1;
                    }
                }

                let average_duration_test = Duration::from_secs_f64(
                    duration_test_total.as_secs_f64() / test_dataset_encrypted.records.len() as f64,
                );
                let accuracy = correct as f64 / test_dataset_encrypted.records.len() as f64;
                let real_accuracy = (correct as f64 + abstention as f64)
                    / test_dataset_encrypted.records.len() as f64;

                println!("Accuracy: {:?}", accuracy);
                println!("Real accuracy: {:?}", real_accuracy);
                println!("Abstention: {:?}", abstention);
                // Write data to perf.csv
                forest.save_perf_for_bench(
                    &format!("{}/perf.csv", dir_path),
                    duration_train,
                    average_duration_test,
                    dataset_name,
                    num_trees,
                    depth,
                    accuracy,
                    real_accuracy,
                    best_accuracy,
                );
            }
        }
    }

    #[test]
    fn test_bench_fully_oblivious() {
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        if !std::path::Path::new(FOLDER).exists() {
            std::fs::create_dir_all(FOLDER).unwrap();
        }
        for num_trees in [8, 16] {
            println!("num_trees: {}", num_trees);
            let num_trials = 10;
            for i in 0..num_trials {
                println!("--------------------------------");
                println!("------------- Trial: {}", i);
                println!("--------------------------------");

                // let dataset_name = "cancer";
                // let dataset_name = "adult";
                // let dataset_name = "wine";
                let dataset_name = "cancer";
                // let dataset_name = "iris";
                let dataset_path = format!("data/{}-uci/{}-sample.csv", dataset_name, dataset_name);

                // Dataset
                let dataset = ClearDataset::from_file(dataset_path.to_string());

                // Base case is for iris
                let mut n_classes = 3;
                let mut f = 4;
                let max_features = dataset.max_features;
                let mut split_percentage = 0.8;
                if dataset_name == "adult" {
                    n_classes = 2;
                    f = 105;
                }

                if dataset_name == "wine" {
                    n_classes = 3;
                    f = 13;
                }

                if dataset_name == "cancer" {
                    n_classes = 2;
                    f = 30;
                }

                let (train_dataset_clear, test_dataset_clear) = dataset.split(split_percentage);

                // let num_trees = 32;
                let depth = 4;

                // FIND BEST MODEL
                let num_trials_best_model = 1;
                let mut best_accuracy = 0.0;
                let mut best_model =
                    ClearForest::new_random_forest(num_trees, depth, n_classes, max_features, f);

                for _ in 0..num_trials_best_model {
                    let mut forest = ClearForest::new_random_forest(
                        num_trees,
                        depth,
                        n_classes,
                        max_features,
                        f,
                    );
                    forest.train(&train_dataset_clear, 0);
                    let accuracy = forest.evaluate(&test_dataset_clear);
                    if accuracy > best_accuracy {
                        best_accuracy = accuracy;
                        best_model = forest;
                    }
                }

                println!("Best accuracy: {:?}", best_accuracy);

                let dir_path = format!("{}/{}/{}", FOLDER, dataset_name, num_trees);

                if !std::path::Path::new(&dir_path).exists() {
                    std::fs::create_dir_all(&dir_path).expect("Failed to create directory");
                }
                let filepath_clear_forest = format!(
                    "{}/best_{}_{}_{}_{:.2}_{}.json",
                    dir_path, dataset_name, num_trees, depth, best_accuracy, i
                );

                best_model.save_to_file(&filepath_clear_forest);

                let train_dataset_encrypted = EncryptedDataset::from_clear_dataset(
                    &train_dataset_clear,
                    &private_key,
                    &mut ctx,
                );

                let test_dataset_encrypted = EncryptedDataset::from_clear_dataset(
                    &test_dataset_clear,
                    &private_key,
                    &mut ctx,
                );

                // TRAIN FOREST
                let timing = TimingCollector::new();
                let mut forest =
                    Forest::new_from_file(filepath_clear_forest.as_str(), &public_key, &ctx);

                let start_train = Instant::now();
                let _predictions =
                    forest.train_fully_oblivious(&train_dataset_encrypted, &public_key, &ctx, &timing);
                let duration_train = start_train.elapsed();

                // Create the directory "forests" if it does not exist
                let forests_dir = format!("{}/forests", dir_path);
                if !std::path::Path::new(&forests_dir).exists() {
                    std::fs::create_dir_all(&forests_dir)
                        .expect("Failed to create forests directory");
                }
                let filepath_forest = format!("{}/forest_{}.json", forests_dir, i);
                forest.save_to_file(&filepath_forest, &private_key, &ctx);

                // TEST FOREST
                let mut correct = 0;
                let mut abstention = 0;
                let mut duration_test_total = Duration::new(0, 0);
                for (i, sample) in test_dataset_encrypted.records.iter().enumerate() {
                    let class_one_hot = private_key.decrypt_lwe_vector(&sample.class, &ctx);
                    let ground_truth = class_one_hot.iter().position(|&x| x == 1).unwrap() as u64;

                    // INFERENCE
                    let start_test = Instant::now();
                    let result = forest.test_index(&sample.features, &public_key, &ctx, i as u64, &timing);
                    let duration_test = start_test.elapsed();
                    duration_test_total += duration_test;
                    let result_clear = private_key.decrypt_lwe(&result, &ctx);
                    println!("Ground truth: {:?}", ground_truth);
                    println!("Result: {:?}", result_clear);

                    // If the result is dataset.n_classes we increase the abstention
                    if result_clear == dataset.n_classes {
                        abstention += 1;
                    }

                    // If the result is good we increase the accuracy
                    if ground_truth == result_clear {
                        correct += 1;
                    }
                }

                let average_duration_test = Duration::from_secs_f64(
                    duration_test_total.as_secs_f64() / test_dataset_encrypted.records.len() as f64,
                );
                let accuracy = correct as f64 / test_dataset_encrypted.records.len() as f64;
                let real_accuracy = (correct as f64 + abstention as f64)
                    / test_dataset_encrypted.records.len() as f64;

                println!("Accuracy: {:?}", accuracy);
                println!("Real accuracy: {:?}", real_accuracy);
                println!("Abstention: {:?}", abstention);
                // Write data to perf.csv
                forest.save_perf_for_bench(
                    &format!("{}/perf.csv", dir_path),
                    duration_train,
                    average_duration_test,
                    dataset_name,
                    num_trees,
                    depth,
                    accuracy,
                    real_accuracy,
                    best_accuracy,
                );
            }
        }
    }
}
