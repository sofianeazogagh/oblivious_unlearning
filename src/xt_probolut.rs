// DATASET
use crate::dataset::*;

// MODEL
use crate::model::*;

// REVOLUT
use revolut::{key, Context, PrivateKey, PublicKey, LUT};
type LWE = LweCiphertext<Vec<u64>>;

// TFHE
use tfhe::{core_crypto::prelude::LweCiphertext, shortint::parameters::*};

// RAYON
use rayon::iter::{IntoParallelIterator, ParallelIterator};

// TIME
#[allow(unused_imports)]
use std::time::Instant;

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
        let (lut_threshold, lut_index) = &tree.stages[i];
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
fn train_single_tree(
    train_dataset: &EncryptedDatasetLut,
    tree_depth: u64,
    tree_idx: u64,
    public_key: &revolut::PublicKey,
    ctx: &revolut::Context,
) -> (TreeLUT, Vec<Vec<LUT>>) {
    let classical_tree =
        Tree::generate_random_tree(tree_depth, train_dataset.n_classes, train_dataset.f, ctx);
    let tree = TreeLUT::from_tree(&classical_tree, tree_depth, train_dataset.n_classes, ctx);

    let luts_samples = (0..train_dataset.records.len() as u64)
        .into_par_iter()
        .map(|j| {
            println!("Training Tree[{}] --- sample [{}]", tree_idx, j);
            probolut_for_training(&tree, &train_dataset.records[j as usize], public_key, ctx)
        })
        .collect();

    (tree, luts_samples)
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

// Test a single tree and return the selected leaf for each sample
fn test_single_tree(
    tree: &TreeLUT,
    test_dataset: &EncryptedDatasetLut,
    tree_idx: u64,
    public_key: &revolut::PublicKey,
    ctx: &revolut::Context,
) -> Vec<LweCiphertext<Vec<u64>>> {
    (0..test_dataset.records.len() as u64)
        .map(|j| {
            println!("Testing Tree[{}] --- sample [{}]", tree_idx, j);
            probolut_inference(
                tree,
                &test_dataset.records[j as usize].features,
                public_key,
                ctx,
            )
        })
        .collect()
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
        println!("\n --------- Training the forest ---------");
        // Train forest
        let forest: Vec<(TreeLUT, Vec<Vec<LUT>>)> = (0..M)
            .into_par_iter()
            .map(|i| train_single_tree(&train_dataset, TREE_DEPTH, i, public_key, &ctx))
            .collect();

        // Aggregate counts
        let summed_counts: Vec<Vec<Vec<u64>>> = forest
            .iter()
            .map(|(_, luts_samples)| {
                aggregate_tree_counts(luts_samples, TREE_DEPTH, &private_key, &ctx)
            })
            .collect();

        println!("\n --------- Testing the forest ---------");
        // Test forest
        let results: Vec<Vec<LweCiphertext<Vec<u64>>>> = (0..M)
            .into_par_iter()
            .map(|i| test_single_tree(&forest[i as usize].0, &test_dataset, i, public_key, &ctx))
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
