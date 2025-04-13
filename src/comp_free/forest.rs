use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;

use rand::Rng;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use serde_json::json;
use serde_json::Value;

use crate::*;

use super::dataset::*;
use super::tree::*;
use super::Majority;
use super::RLWE;
pub struct Forest {
    pub trees: Vec<Tree>,
}

const SEED: u64 = 1;
const PRE_SEEDED: bool = false;
const EXPORT: bool = true;
const NUM_THREADS: usize = 4;

const FOLDER: &str = "./src/comp_free/campaign_1";

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

    pub fn train(&mut self, dataset: &EncryptedDataset, public_key: &PublicKey, ctx: &Context) {
        println!("Dataset size: {}", dataset.records.len());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(NUM_THREADS)
            .build()
            .unwrap();

        let start = Instant::now();
        pool.install(|| {
            self.trees.par_iter_mut().for_each(|tree| {
                tree.train(dataset, public_key, ctx);
            });
        });
        let duration = start.elapsed();
        println!("[TIME] Total trees training: {:?}", duration);

        let start = Instant::now();
        pool.install(|| {
            self.trees.par_iter_mut().for_each(|tree| {
                tree.leaves_majority(public_key, ctx);
            });
        });
        let duration = start.elapsed();
        println!("[TIME] Total forest majority: {:?}", duration);
    }

    pub fn test(&self, sample_features: &Vec<RLWE>, public_key: &PublicKey, ctx: &Context) -> LWE {
        let mut res = Vec::new();
        let start = Instant::now();
        for tree in self.trees.iter() {
            res.push(tree.test(sample_features, public_key, ctx));
        }
        let duration = start.elapsed();
        println!("[TIME] Total forest evaluation: {:?}", duration);

        let start = Instant::now();
        let result = public_key.blind_majority_extra(&res, ctx);
        let duration = start.elapsed();
        println!("[TIME] Majority vote: {:?}", duration);
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
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_train_forest() {
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        let dataset = EncryptedDataset::from_file(
            "data/iris-uci/iris.csv".to_string(),
            &private_key,
            &mut ctx,
            3,
        );

        let start = Instant::now();
        let mut forest = Forest::new(64, 4, dataset.n_classes, dataset.f, &public_key, &ctx);
        forest.train(&dataset, &public_key, &ctx);
        let duration = start.elapsed();
        println!("Time taken: {:?}", duration);
        if EXPORT {
            forest.save_perf_to_file(
                "perf.csv",
                duration,
                Duration::new(0, 0),
                "iris",
                64,
                4,
                -1.0,
            );
            let filepath = "./src/comp_free/iris_forest_test_lut.json";
            forest.save_to_file(filepath, &private_key, &ctx);
        }

        forest.print(&private_key, &ctx);
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
        forest.train(&dataset, &public_key, &ctx);
        let filepath = format!(
            "./src/comp_free/test_iris_forest_{}m_{}d.json",
            n_trees, depth
        );
        forest.save_to_file(&filepath, &private_key, &ctx);

        // let forest = Forest::load_from_file(
        //     format!(
        //         "./src/comp_free/test_iris_forest_{}m_{}d.json",
        //         n_trees, depth
        //     )
        //     .as_str(),
        //     &ctx,
        //     &public_key,
        // );

        forest.print(&private_key, &ctx);

        let sample_features = dataset.records[0].features.clone();
        let sample_class = dataset.records[0].class.clone();

        let class_one_hot = private_key.decrypt_lwe_vector(&sample_class, &ctx);

        let ground_truth = class_one_hot.iter().position(|&x| x == 1).unwrap() as u64;

        let result = forest.test(&sample_features, &public_key, &ctx);

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
        let mut forest = Forest::new(
            n_trees,
            depth,
            dataset.n_classes,
            dataset.f,
            &public_key,
            &ctx,
        );
        let start_train = Instant::now();
        forest.train(&train_dataset, &public_key, &ctx);
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
            let result = forest.test(&sample_features, &public_key, &ctx);
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
    fn test_bench() {
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        for _ in 0..10 {
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
            let mut forest = Forest::new(
                n_trees,
                depth,
                dataset.n_classes,
                dataset.f,
                &public_key,
                &ctx,
            );
            let start_train = Instant::now();
            forest.train(&train_dataset, &public_key, &ctx);
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
                let result = forest.test(&sample_features, &public_key, &ctx);
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
    }
}
