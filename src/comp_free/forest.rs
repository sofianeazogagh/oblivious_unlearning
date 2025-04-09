use crate::*;

use super::dataset::*;
use super::tree::*;
pub struct Forest {
    trees: Vec<Tree>,
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
            let tree = Tree::new_random_tree(depth, n_classes, f, public_key, ctx);
            trees.push(tree);
        }
        Self { trees }
    }

    pub fn train(&mut self, dataset: &EncryptedDataset, public_key: &PublicKey, ctx: &Context) {
        for tree in self.trees.iter_mut() {
            tree.train(dataset, public_key, ctx);
        }
    }
    pub fn print(&self, private_key: &PrivateKey, ctx: &Context) {
        for tree in self.trees.iter() {
            tree.print_tree(private_key, ctx);
        }
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_train_forest() {
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        let mut forest = Forest::new(10, 10, 10, 10, &public_key, &ctx);

        let dataset = EncryptedDataset::from_file(
            "/Users/sofianeazogagh/Desktop/PROBONITE/PROBONITE/data/iris/iris.csv".to_string(),
            &private_key,
            &mut ctx,
            3,
        );

        forest.train(&dataset, &public_key, &ctx);

        forest.print(&private_key, &ctx);
    }
}
