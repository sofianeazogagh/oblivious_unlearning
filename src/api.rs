use std::time::Duration;

use crate::clear::ClearForest;
use crate::dataset::{ClearDataset, EncryptedDataset};
use crate::forest::Forest;
use crate::timing::TimingCollector;
use crate::time_operation;
use revolut::{Context, PrivateKey, PublicKey};

/// Standard option: trains in clear, selects the best model, exports without counts,
/// then retrains with encrypted data
pub fn run_standard_mode(
    dataset_name: &str,
    num_trees: u64,
    depth: u64,
    n_classes: u64,
    max_features: u64,
    f: u64,
    num_trials_best_model: usize,
    split_percentage: f64,
    output_dir: &str,
    ctx: &mut Context,
    private_key: &PrivateKey,
    public_key: &PublicKey,
    timing: &TimingCollector,
    verbose: bool,
) {
    if verbose {
        println!("========== Standard Mode ==========");
        println!("Dataset: {}", dataset_name);
        println!("Num trees: {}, Depth: {}", num_trees, depth);
    }

    // Load the dataset
    let dataset_path = format!("data/{}-uci/{}.csv", dataset_name, dataset_name);
    let dataset = ClearDataset::from_file(dataset_path);
    let (train_dataset_clear, test_dataset_clear) = dataset.split(split_percentage);

    if verbose {
        println!("Train size: {}, Test size: {}", 
                 train_dataset_clear.records.len(), 
                 test_dataset_clear.records.len());
    }

    // 1. Find the best model in clear
    if verbose {
        println!("\n[1/4] Searching for the best model in clear...");
    }
    let (mut best_model, best_accuracy) = crate::find_best_model(
        &train_dataset_clear,
        &test_dataset_clear,
        num_trees,
        depth,
        n_classes,
        max_features,
        f,
        num_trials_best_model,
    );

    if verbose {
        println!("Best accuracy: {:.4}", best_accuracy);
    }

    // 2. Export without counts
    if verbose {
        println!("\n[2/4] Exporting forest without counts...");
    }
    let exported_forest = best_model.export_without_counts();
    let dir_path = format!("{}/{}/{}", output_dir, dataset_name, num_trees);
    if !std::path::Path::new(&dir_path).exists() {
        std::fs::create_dir_all(&dir_path).expect("Failed to create directory");
    }
    let export_path = format!(
        "{}standard_{}_{}_{}_{:.2}.json",
        dir_path, dataset_name, num_trees, depth, best_accuracy
    );
    exported_forest.save_to_file(&export_path);
    if verbose {
        println!("Forest exported to: {}", export_path);
    }

    // 3. Encrypt the datasets
    if verbose {
        println!("\n[3/4] Encrypting datasets...");
    }
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

    // 4. Retrain with encrypted data
    if verbose {
        println!("\n[4/4] Retraining with encrypted data...");
    }
    let mut forest = Forest::new_from_file(&export_path, public_key, ctx);
    
    time_operation!(timing, "Train forest encrypted", {
        forest.train(&train_dataset_encrypted, public_key, ctx, timing);
    });

    // 5. Test the retrained forest
    if verbose {
        println!("\nTesting the retrained forest...");
    }
    let mut correct = 0;
    let mut abstention = 0;
    
    for (i, sample) in test_dataset_encrypted.records.iter().enumerate() {
        let class_one_hot = private_key.decrypt_lwe_vector(&sample.class, ctx);
        let ground_truth = class_one_hot.iter().position(|&x| x == 1).unwrap() as u64;

        let result = forest.test_index(&sample.features, public_key, ctx, i as u64, timing);
        let result_clear = private_key.decrypt_lwe(&result, ctx);
        
        if result_clear == n_classes {
            abstention += 1;
        }
        if ground_truth == result_clear {
            correct += 1;
        }
    }

    let accuracy_encrypted = correct as f64 / test_dataset_encrypted.records.len() as f64;
    let real_accuracy_encrypted = (correct as f64 + abstention as f64)
        / test_dataset_encrypted.records.len() as f64;

    println!("\n========== Results ==========");
    println!("Accuracy in clear: {:.4}", best_accuracy);
    println!("Accuracy with encrypted data: {:.4}", accuracy_encrypted);
    println!("Real accuracy with encrypted data: {:.4}", real_accuracy_encrypted);
    println!("Abstention: {}", abstention);
    println!("Degradation: {:.4}", best_accuracy - accuracy_encrypted);
}

/// Hybrid option: split D_0/D_1, trains in clear on D_0 with Gini, exports with counts,
/// then continues training on encrypted D_1
pub fn run_hybrid_mode(
    dataset_name: &str,
    num_trees: u64,
    depth: u64,
    n_classes: u64,
    max_features: u64,
    f: u64,
    k: usize, // number of candidates for Gini split
    split_d0_d1: f64, // percentage for D_0 (remainder = D_1)
    split_train_test: f64, // train/test percentage in D_0
    output_dir: &str,
    ctx: &mut Context,
    private_key: &PrivateKey,
    public_key: &PublicKey,
    timing: &TimingCollector,
    verbose: bool,
) {
    if verbose {
        println!("========== Hybrid Mode ==========");
        println!("Dataset: {}", dataset_name);
        println!("Num trees: {}, Depth: {}, k: {}", num_trees, depth, k);
    }

    // Load the dataset
    let dataset_path = format!("data/{}-uci/{}.csv", dataset_name, dataset_name);
    let dataset = ClearDataset::from_file(dataset_path);

    // 1. Split D_0 and D_1
    if verbose {
        println!("\n[1/5] Splitting dataset into D_0 and D_1...");
    }
    let (d_0, d_1) = dataset.split(split_d0_d1);
    let (train_dataset_d0, test_dataset_d0) = d_0.split(split_train_test);

    if verbose {
        println!("D_0 size: {} (train: {}, test: {})", 
                 d_0.records.len(),
                 train_dataset_d0.records.len(),
                 test_dataset_d0.records.len());
        println!("D_1 size: {}", d_1.records.len());
    }

    // 2. Train in clear on D_0 with Gini (ERT)
    if verbose {
        println!("\n[2/5] Training in clear on D_0 with Gini (ERT)...");
    }
    let mut forest_clear = ClearForest::new_ert_forest(
        num_trees,
        depth,
        n_classes,
        max_features,
        f,
        k,
        &train_dataset_d0,
    );
    forest_clear.train(&train_dataset_d0, 1);
    
    let accuracy_d0 = forest_clear.evaluate(&test_dataset_d0);
    if verbose {
        println!("Accuracy on D_0: {:.4}", accuracy_d0);
    }

    // 3. Export with counts
    if verbose {
        println!("\n[3/5] Exporting forest with counts...");
    }
    let dir_path = format!("{}/{}/{}", output_dir, dataset_name, num_trees);
    if !std::path::Path::new(&dir_path).exists() {
        std::fs::create_dir_all(&dir_path).expect("Failed to create directory");
    }
    let export_path = format!(
        "{}hybrid_{}_{}_{}_{:.2}.json",
        dir_path, dataset_name, num_trees, depth, accuracy_d0
    );
    forest_clear.save_to_file(&export_path);
    if verbose {
        println!("Forest exported to: {}", export_path);
    }

    // 4. Encrypt D_1
    if verbose {
        println!("\n[4/5] Encrypting D_1...");
    }
    let d_1_encrypted = EncryptedDataset::from_clear_dataset(
        &d_1,
        private_key,
        ctx,
    );

    // 5. Continue training on encrypted D_1
    if verbose {
        println!("\n[5/5] Continuing training on encrypted D_1...");
    }
    let mut forest = Forest::new_from_file(&export_path, public_key, ctx);
    
    time_operation!(timing, "Train forest encrypted D1", {
        forest.train(&d_1_encrypted, public_key, ctx, timing);
    });

    // 6. Test on the test set of D_0
    if verbose {
        println!("\nTesting the forest after training on D_1...");
    }
    let test_dataset_d0_encrypted = EncryptedDataset::from_clear_dataset(
        &test_dataset_d0,
        private_key,
        ctx,
    );

    let mut correct = 0;
    let mut abstention = 0;
    
    for (i, sample) in test_dataset_d0_encrypted.records.iter().enumerate() {
        let class_one_hot = private_key.decrypt_lwe_vector(&sample.class, ctx);
        let ground_truth = class_one_hot.iter().position(|&x| x == 1).unwrap() as u64;

        let result = forest.test_index(&sample.features, public_key, ctx, i as u64, timing);
        let result_clear = private_key.decrypt_lwe(&result, ctx);
        
        if result_clear == n_classes {
            abstention += 1;
        }
        if ground_truth == result_clear {
            correct += 1;
        }
    }

    let accuracy_after_d1 = correct as f64 / test_dataset_d0_encrypted.records.len() as f64;
    let real_accuracy_after_d1 = (correct as f64 + abstention as f64)
        / test_dataset_d0_encrypted.records.len() as f64;

    println!("\n========== Results ==========");
    println!("Accuracy on D_0 (clear): {:.4}", accuracy_d0);
    println!("Accuracy after D_1 (encrypted): {:.4}", accuracy_after_d1);
    println!("Real accuracy after D_1: {:.4}", real_accuracy_after_d1);
    println!("Abstention: {}", abstention);
    println!("Evolution: {:.4}", accuracy_after_d1 - accuracy_d0);
}

/// Oblivious training/unlearning mode: loads a pre-trained forest and a CSV file,
/// then trains or unlearns based on the multiplier (1 for training, 2 for unlearning)
pub fn run_oblivious_mode(
    forest_path: &str,
    csv_path: &str,
    operation: &str, // "train" or "unlearn"
    output_dir: &str,
    ctx: &mut Context,
    private_key: &PrivateKey,
    public_key: &PublicKey,
    timing: &TimingCollector,
    verbose: bool,
) {
    if verbose {
        println!("========== Oblivious {} Mode ==========", operation);
        println!("Forest path: {}", forest_path);
        println!("CSV path: {}", csv_path);
    }

    // Determine multiplier based on operation
    let multiplier = match operation {
        "train" => 1,
        "unlearn" => 2,
        _ => {
            eprintln!("Unknown operation: {}. Use 'train' or 'unlearn'.", operation);
            std::process::exit(1);
        }
    };

    if verbose {
        println!("Multiplier: {} ({} = multiply one-hot label by {})", 
                 multiplier, operation, multiplier);
    }

    // 1. Load the pre-trained forest
    if verbose {
        println!("\n[1/3] Loading pre-trained forest...");
    }
    let mut forest = Forest::new_from_file(forest_path, public_key, ctx);
    
    // Get n_classes from the forest
    let n_classes = forest.trees[0].n_classes;
    if verbose {
        println!("Forest loaded: {} trees, depth {}, {} classes", 
                 forest.trees.len(), 
                 forest.trees[0].depth,
                 n_classes);
    }

    // 2. Load and encrypt the CSV data with the appropriate multiplier
    if verbose {
        println!("\n[2/3] Loading and encrypting CSV data (multiplier: {})...", multiplier);
    }
    let dataset_encrypted = EncryptedDataset::from_file_with_multiplier(
        csv_path.to_string(),
        private_key,
        ctx,
        n_classes,
        multiplier,
    );
    
    if verbose {
        println!("Loaded {} samples from CSV", dataset_encrypted.records.len());
    }

    // 3. Train/unlearn on the encrypted data
    if verbose {
        println!("\n[3/3] {}ing on encrypted data...", operation);
    }
    
    time_operation!(timing, &format!("{} forest", operation), {
        forest.train(&dataset_encrypted, public_key, ctx, timing);
    });

    // 4. Save the updated forest
    if verbose {
        println!("\n[4/4] Saving updated forest...");
    }
    
    // Create output directory if it doesn't exist
    if !std::path::Path::new(output_dir).exists() {
        std::fs::create_dir_all(output_dir).expect("Failed to create output directory");
    }
    
    // Generate a meaningful filename
    let csv_filename = std::path::Path::new(csv_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("data");
    
    let forest_filename = std::path::Path::new(forest_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("forest");
    
    let output_path = format!("{}/forest_{}_{}_{}.json", 
                             output_dir, 
                             operation,
                             forest_filename,
                             csv_filename);
    
    forest.save_to_file(&output_path, private_key, ctx);
    
    if verbose {
        println!("Updated forest saved to: {}", output_path);
    }

    println!("\n========== Results ==========");
    println!("Operation: {}", operation);
    println!("Samples processed: {}", dataset_encrypted.records.len());
    println!("Updated forest saved to: {}", output_path);
}

