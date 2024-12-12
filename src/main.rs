mod probonite;
use probonite::*;
use revolut::{key, Context};
use tfhe::shortint::parameters::*;

fn main() {
    let mut ctx = Context::from(PARAM_MESSAGE_2_CARRY_0);
    let private_key = key(ctx.parameters());
    let public_key = &private_key.public_key;

    let n_classes = 3;
    let depth = 4;
    let mut tree = Tree::generate_random_tree(depth, n_classes, &ctx);
    tree.print_tree(&private_key, &ctx, n_classes);

    let feature_vector = vec![1, 2, 3];
    let class = 1;

    let query = Query::new(&feature_vector, &class, &private_key, &mut ctx);
    probonite(&mut tree, &query, &public_key, &ctx);

    tree.print_tree(&private_key, &ctx, n_classes);
}
