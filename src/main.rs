use std::time::Instant;

mod probonite;
use probonite::*;

mod model;
use model::*;

mod dataset;
use dataset::*;

mod clear_model;
use clear_model::*;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use revolut::{key, Context};

use tfhe::shortint::parameters::*;

const GENERATE_TREE: bool = false;
const EXPORT_FOREST: bool = true;
fn update() {
    let mut ctx = Context::from(PARAM_MESSAGE_3_CARRY_0);
    let private_key = key(ctx.parameters());
    let public_key = &private_key.public_key;

    let mut tree: Tree;
    let tree_depth = 3;
    let n_classes = 3;
    let f = ctx.full_message_modulus() as u64;

    if GENERATE_TREE {
        tree = Tree::generate_random_tree(tree_depth, n_classes, f, &ctx);
        tree.save_to_file(
            &format!("random_trees/random_tree_{}_{}.json", tree_depth, n_classes),
            &ctx,
        );
    } else {
        tree = Tree::load_from_file(
            &format!("random_trees/random_tree_{}_{}.json", tree_depth, n_classes),
            &ctx,
        )
        .unwrap();
    }

    let feature_vector = vec![1, 1, 1, 1, 1, 1, 1, 1];
    let class = 1;

    let query = Query::make_query(&feature_vector, &class, &private_key, &mut ctx);

    let start = Instant::now();
    probonite(&mut tree, &query, &public_key, &ctx);
    let end = Instant::now();
    println!("Time taken: {:?}", end.duration_since(start));

    tree.print_tree(&private_key, &ctx);
}

fn example_private_training() {
    let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
    let private_key = key(ctx.parameters());
    let public_key = &private_key.public_key;

    // let dataset =
    //     EncryptedDataset::from_file("data/iris_2bits.csv".to_string(), &private_key, &mut ctx);
    let dataset =
        EncryptedDataset::from_file("data/wine_4bits.csv".to_string(), &private_key, &mut ctx);

    let m = 10;
    let d = 4;
    let n_classes = 3;

    let start = Instant::now();

    let forest: Vec<Tree> = (0..m)
        .into_par_iter()
        .map(|i| {
            let mut tree = Tree::generate_random_tree(d, n_classes, dataset.f, &ctx);
            let start_one_tree = Instant::now();
            let luts_samples = (0..dataset.n)
                .into_par_iter()
                .map(|j| {
                    let query = &dataset.records[j as usize];
                    probonite(&tree, query, &public_key, &ctx)
                })
                .collect();
            tree.sum_samples_luts_counts(&luts_samples, &public_key);
            let elapsed_one_tree = Instant::now() - start_one_tree;
            println!("Time taken for Tree[{i}] : {:?}", elapsed_one_tree);
            tree
        })
        .collect();

    let end = Instant::now() - start;
    println!(
        "[PARAMETERS] :
    Precision bits: {}, \
    Dataset size: {}, \
    Dataset dimension: {}, \
    Dataset classes: {}, \
    Tree depth: {}, \
    Number of trees: {}",
        (ctx.full_message_modulus() as u64).ilog2(),
        dataset.n,
        dataset.f,
        n_classes,
        d,
        m
    );
    println!("[SUMMARY] : Forest built in {:?}", end);

    if EXPORT_FOREST {
        for i in 0..m {
            forest[i].save_to_file(&format!("wine_forest/tree_{}.json", i), &ctx);
        }
    }
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
}
