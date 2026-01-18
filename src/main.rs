#![allow(warnings)]

use std::{io::Write, time::Duration};
use std::time::Instant;

// ============================================================================
// Declarations of modules
// ============================================================================
mod api;            // API for running standard and hybrid modes
mod clear;
mod ctree;
mod dataset;
mod forest;
mod timing;          // Timing collector for microbenchmarking
mod tree;
mod utils_maj;      // Extension trait Majority for RevoLUT's PublicKey
mod utils_serial;   // Serialization methods for Tree and ClearForest


use api::{run_hybrid_mode, run_standard_mode};
use clear::ClearForest;
use dataset::{ClearDataset, EncryptedDataset};
use forest::Forest;
use timing::TimingCollector;

use clap::Parser;
use revolut::*;
use tfhe::shortint::parameters::*;

use revolut::{Context, PrivateKey, PublicKey, key, LUT, LWE, radix::ByteLWE, radix::NyblByteLUT};

type RLWE = crate::GLWE;

pub const FOLDER: &str = "./export/";
const DEFAULT_NUM_TRIALS: usize = 10;
const DEFAULT_DEPTH: u64 = 4;
const DEFAULT_SPLIT_PERCENTAGE: f64 = 0.8;
const DEFAULT_NUM_TRIALS_BEST_MODEL: usize = 1;


pub const NUM_THREADS: usize = 10;
pub const VERBOSE: bool = true;
pub const DEBUG: bool = false;
pub const OBLIVIOUS: bool = true; // oblivious train/unlearn
pub const SEED: u64 = 1;
pub const PRE_SEEDED: bool = true;

// ============================================================================
// Configuration structures
// ============================================================================
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Dataset name (iris, adult, wine, cancer)
    #[arg(short, long, default_value = "iris")]
    dataset: String,

    /// Number of trees (comma-separated values, e.g., "8,16")
    #[arg(short, long, default_value = "8,16")]
    num_trees: String,

    /// Tree depth
    #[arg(short, long, default_value_t = DEFAULT_DEPTH)]
    depth: u64,

    /// Number of repetitions/trials
    #[arg(short, long, default_value_t = DEFAULT_NUM_TRIALS)]
    trials: usize,

    /// Train/test split percentage
    #[arg(long, default_value_t = DEFAULT_SPLIT_PERCENTAGE)]
    split: f64,

    /// Number of trials to find the best model
    #[arg(long, default_value_t = DEFAULT_NUM_TRIALS_BEST_MODEL)]
    best_model_trials: usize,

    /// Output directory
    #[arg(long, default_value = FOLDER)]
    output: String,

    /// Verbose mode
    #[arg(short, long)]
    verbose: bool,

    /// Execution mode: "standard" or "hybrid"
    #[arg(long, default_value = "standard")]
    mode: String,
}

#[derive(Debug, Clone)]
struct DatasetConfig {
    n_classes: u64,
    f: u64,
}

// Utility functions
fn get_dataset_config(dataset_name: &str) -> DatasetConfig {
    match dataset_name {
        "iris" => DatasetConfig { n_classes: 3, f: 4 },
        "wine" => DatasetConfig { n_classes: 3, f: 13 },
        "cancer" => DatasetConfig { n_classes: 2, f: 30 },
        _ => {
            eprintln!("Dataset unknown: {}. Using default values (iris).", dataset_name);
            DatasetConfig { n_classes: 3, f: 4 }
        }
    }
}

fn parse_num_trees(num_trees_str: &str) -> Vec<u64> {
    num_trees_str
        .split(',')
        .map(|s| s.trim().parse().expect("Invalid value for num_trees"))
        .collect()
}


fn find_best_model(
    train_dataset: &ClearDataset,
    test_dataset: &ClearDataset,
    num_trees: u64,
    depth: u64,
    n_classes: u64,
    max_features: u64,
    f: u64,
    num_trials: usize,
) -> (ClearForest, f64) {
    let mut best_accuracy = 0.0;
    let mut best_model = ClearForest::new_random_forest(num_trees, depth, n_classes, max_features, f);

    for _ in 0..num_trials {
        let mut forest = ClearForest::new_random_forest(
            num_trees,
            depth,
            n_classes,
            max_features,
            f,
        );
        forest.train(train_dataset, 0);
        let accuracy = forest.evaluate(test_dataset);
        if accuracy > best_accuracy {
            best_accuracy = accuracy;
            best_model = forest;
        }
    }

    (best_model, best_accuracy)
}

fn train_and_test_forest(
    filepath_clear_forest: &str,
    train_dataset_encrypted: &EncryptedDataset,
    test_dataset_encrypted: &EncryptedDataset,
    dataset: &ClearDataset,
    public_key: &PublicKey,
    private_key: &PrivateKey,
    ctx: &mut Context,
    trial_id: usize,
    dir_path: &str,
    dataset_name: &str,
    num_trees: u64,
    depth: u64,
    best_accuracy: f64,
    timing: &TimingCollector,
) {
    // TRAIN FOREST
    let mut forest = Forest::new_from_file(filepath_clear_forest, public_key, ctx);

    time_operation!(timing, "Train forest", {
        forest.train(train_dataset_encrypted, public_key, ctx, timing);
    });
    
    // Get duration_train from timing stats
    let stats = timing.get_statistics();
    let duration_train = stats.get("Train forest")
        .map(|s| s.total)
        .unwrap_or(Duration::ZERO);

    // Create the directory "forests" if it does not exist
    let forests_dir = format!("{}/forests", dir_path);
    if !std::path::Path::new(&forests_dir).exists() {
        std::fs::create_dir_all(&forests_dir)
            .expect("Failed to create forests directory");
    }
    let filepath_forest = format!("{}/forest_{}.json", forests_dir, trial_id);
    forest.save_to_file(&filepath_forest, private_key, ctx);

    // TEST FOREST
    let mut correct = 0;
    let mut abstention = 0;
    
    for (i, sample) in test_dataset_encrypted.records.iter().enumerate() {
        let class_one_hot = private_key.decrypt_lwe_vector(&sample.class, ctx);
        let ground_truth = class_one_hot.iter().position(|&x| x == 1).unwrap() as u64;

        // INFERENCE - timing is recorded internally
        let result = forest.test_index(&sample.features, public_key, ctx, i as u64, timing);
        
        let result_clear = private_key.decrypt_lwe(&result, ctx);
        
        if result_clear == dataset.n_classes {
            abstention += 1;
        }

        if ground_truth == result_clear {
            correct += 1;
        }
    }

    // Calculate average test duration from timing stats
    let stats = timing.get_statistics();
    let eval_total = stats.get("Total forest evaluation")
        .map(|s| s.total)
        .unwrap_or(Duration::ZERO);
    let majority_total = stats.get("Majority vote")
        .map(|s| s.total)
        .unwrap_or(Duration::ZERO);
    let duration_test_total = eval_total + majority_total;
    let average_duration_test = Duration::from_secs_f64(
        duration_test_total.as_secs_f64() / test_dataset_encrypted.records.len() as f64,
    );
    let accuracy = correct as f64 / test_dataset_encrypted.records.len() as f64;
    let real_accuracy = (correct as f64 + abstention as f64)
        / test_dataset_encrypted.records.len() as f64;

    println!("[Trial {}] Accuracy: {:.4}, Real accuracy: {:.4}, Abstention: {}", 
             trial_id, accuracy, real_accuracy, abstention);
    
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

fn run_benchmark(
    args: &Args,
    ctx: &mut Context,
    private_key: &PrivateKey,
    public_key: &PublicKey,
) {
    let timing = TimingCollector::new();
    
    let dataset_config = get_dataset_config(&args.dataset);
    let dataset_path = format!("data/{}-uci/{}.csv", args.dataset, args.dataset);
    let dataset = ClearDataset::from_file(dataset_path);
    let max_features = dataset.max_features;

    let (train_dataset_clear, test_dataset_clear) = dataset.split(args.split);

    let num_trees_list = parse_num_trees(&args.num_trees);

    for num_trees in num_trees_list {
        if args.verbose {
            println!("============= Dataset: {}, Num trees: {}, Depth: {} =============",
                     args.dataset, num_trees, args.depth);
        }

        // FIND BEST MODEL
        let (best_model, best_accuracy) = find_best_model(
            &train_dataset_clear,
            &test_dataset_clear,
            num_trees,
            args.depth,
            dataset_config.n_classes,
            max_features,
            dataset_config.f,
            args.best_model_trials,
        );

        if args.verbose {
            println!("Best accuracy: {:.4}", best_accuracy);
        }

        let dir_path = format!("{}/{}/{}", args.output, args.dataset, num_trees);
        if !std::path::Path::new(&dir_path).exists() {
            std::fs::create_dir_all(&dir_path).expect("Failed to create directory");
        }

        // Encrypt datasets
        let train_dataset_encrypted = EncryptedDataset::from_clear_dataset(
            &train_dataset_clear,
            private_key,
            ctx,
        );

        let test_dataset_encrypted = EncryptedDataset::from_clear_dataset(
            &test_dataset_clear,
            private_key,
            ctx,
        );

        // Run trials
        for trial_id in 0..args.trials {
            if args.verbose {
                println!("-------------------------------");
                println!("---------- Trial: {} ----------", trial_id);
                println!("-------------------------------");
            }

            let filepath_clear_forest = format!(
                "{}/best_{}_{}_{}_{:.2}_{}.json",
                dir_path, args.dataset, num_trees, args.depth, best_accuracy, trial_id
            );

            best_model.save_to_file(&filepath_clear_forest);

            train_and_test_forest(
                &filepath_clear_forest,
                &train_dataset_encrypted,
                &test_dataset_encrypted,
                &dataset,
                public_key,
                private_key,
                ctx,
                trial_id,
                &dir_path,
                &args.dataset,
                num_trees,
                args.depth,
                best_accuracy,
                &timing,
            );
        }
        
        // Print timing statistics after all trials for this configuration
        timing.print_summary();
        timing.clear();
    }
}

fn main() {
    let args = Args::parse();

    // Initialize context and keys
    let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
    let private_key = key(ctx.parameters());
    let public_key = &private_key.public_key;

    // Create output directory if it doesn't exist
    if !std::path::Path::new(&args.output).exists() {
        std::fs::create_dir_all(&args.output).unwrap();
    }

    let dataset_config = get_dataset_config(&args.dataset);
    let dataset_path = format!("data/{}-uci/{}.csv", args.dataset, args.dataset);
    let dataset = ClearDataset::from_file(dataset_path);
    let max_features = dataset.max_features;

    let timing = TimingCollector::new();

    // Choisir entre mode standard et hybrid
    match args.mode.as_str() {
        "standard" => {
            if args.verbose {
                println!("Configuration:");
                println!("  Mode: standard");
                println!("  Dataset: {}", args.dataset);
                println!("  Num trees: {}", args.num_trees);
                println!("  Depth: {}", args.depth);
                println!("  Split: {:.2}", args.split);
                println!("  Output: {}", args.output);
            }

            // For standard mode, take the first number of trees
            let num_trees_list = parse_num_trees(&args.num_trees);
            let num_trees = num_trees_list[0];

            run_standard_mode(
                &args.dataset,
                num_trees,
                args.depth,
                dataset_config.n_classes,
                max_features,
                dataset_config.f,
                args.best_model_trials,
                args.split,
                &args.output,
                &mut ctx,
                &private_key,
                public_key,
                &timing,
                args.verbose,
            );
        }
        "hybrid" => {
            if args.verbose {
                println!("Configuration:");
                println!("  Mode: hybrid");
                println!("  Dataset: {}", args.dataset);
                println!("  Num trees: {}", args.num_trees);
                println!("  Depth: {}", args.depth);
                println!("  Split D_0/D_1: {:.2}", args.split);
                println!("  Output: {}", args.output);
            }

            // For hybrid mode, take the first number of trees
            let num_trees_list = parse_num_trees(&args.num_trees);
            let num_trees = num_trees_list[0];

            // k = number of candidates for Gini split (default 3)
            let k = 3;
            // Split D_0/D_1: use args.split for D_0, remainder is D_1
            // Train/test split in D_0: 0.8 by default
            let split_train_test = 0.8;

            run_hybrid_mode(
                &args.dataset,
                num_trees,
                args.depth,
                dataset_config.n_classes,
                max_features,
                dataset_config.f,
                k,
                args.split, // pourcentage pour D_0
                split_train_test,
                &args.output,
                &mut ctx,
                &private_key,
                public_key,
                &timing,
                args.verbose,
            );
        }
        _ => {
            eprintln!("Unknown mode: {}. Use 'standard' or 'hybrid'.", args.mode);
            std::process::exit(1);
        }
    }

    // Display timing statistics
    timing.print_summary();
}
