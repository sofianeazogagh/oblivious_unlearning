use std::time::Instant;

mod probonite;
use probonite::*;

mod model;
use model::*;

use revolut::{key, Context};
use tfhe::shortint::parameters::*;

const GENERATE_TREE: bool = false;

fn main() {
    let mut ctx = Context::from(PARAM_MESSAGE_3_CARRY_0);
    let private_key = key(ctx.parameters());
    let public_key = &private_key.public_key;

    let mut tree: Tree;
    let tree_depth = 3;
    let n_classes = 2;

    if GENERATE_TREE {
        tree = Tree::generate_random_tree(tree_depth, n_classes, &ctx);
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

    let feature_vector = vec![1, 0, 1, 1, 1, 1, 0, 1];
    let class = 1;

    let query = Query::make_query(&feature_vector, &class, &private_key, &mut ctx);

    let start = Instant::now();
    probonite(&mut tree, &query, &public_key, &ctx);
    let end = Instant::now();
    println!("Time taken: {:?}", end.duration_since(start));

    tree.print_tree(&private_key, &ctx);
}
