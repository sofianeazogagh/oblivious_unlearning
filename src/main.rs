use std::time::Instant;

mod probonite;
use probonite::*;

mod model;
use model::*;

mod dataset;
use dataset::*;

use rayon::{iter::{IntoParallelIterator, ParallelIterator}, vec};
use revolut::{key, Context};
use tfhe::shortint::parameters::*;

const GENERATE_TREE: bool = true;

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

fn main() {
    let mut ctx = Context::from(PARAM_MESSAGE_2_CARRY_0);
    let private_key = key(ctx.parameters());
    let public_key = &private_key.public_key;

    let dataset = EncryptedDataset::from_file(
        "data/iris_2bits.csv".to_string(),
        &private_key,
        &mut ctx,
    );

    let m = 10;
    let d = 4;
    let n_classes = 3;

    let start = Instant::now();

    let forest:Vec<Tree> = (0..m).into_iter().map(|i|
        {

            let mut tree = Tree::generate_random_tree(d, n_classes, dataset.f, &ctx);
            let start_one_tree = Instant::now();
            for j in 0..dataset.n {
                let query = &dataset.records[j as usize];
                probonite(&mut tree, &query, &public_key, &ctx);
            }
            let elapsed_one_tree = Instant::now() - start_one_tree;
            println!("Time taken for Tree[{i}] : {:?}",elapsed_one_tree);

            tree
            
        }).collect();

    let end = Instant::now() - start;
    println!("Time taken for building the forest : {:?}", end);
    
}
