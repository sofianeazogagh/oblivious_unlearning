use std::fs::File;
use std::io::Read;

use rayon::iter::IntoParallelRefMutIterator;
use serde_json::json;
use serde_json::Value;

use crate::*;

use super::dataset::*;
use super::tree::*;
pub struct Forest {
    pub trees: Vec<Tree>,
}

const SEED: u64 = 1;

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
            let tree: Tree =
                Tree::new_random_tree_with_seed(depth, n_classes, f, public_key, ctx, SEED);
            trees.push(tree);
        }
        Self { trees }
    }

    pub fn train(&mut self, dataset: &EncryptedDataset, public_key: &PublicKey, ctx: &Context) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(10)
            .build()
            .unwrap();
        pool.install(|| {
            self.trees.par_iter_mut().for_each(|tree| {
                tree.train(dataset, public_key, ctx);
            });
        });
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

        let dataset = EncryptedDataset::from_file(
            "data/iris-uci/iris.csv".to_string(),
            &private_key,
            &mut ctx,
            3,
        );

        let start = Instant::now();
        let mut forest = Forest::new(10, 4, dataset.n_classes, dataset.f, &public_key, &ctx);
        forest.train(&dataset, &public_key, &ctx);
        let duration = start.elapsed();
        println!("Time taken: {:?}", duration);

        forest.print(&private_key, &ctx);
    }
}
