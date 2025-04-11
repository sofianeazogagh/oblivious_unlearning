use std::fs::File;
use std::io::Read;

use rayon::iter::IntoParallelRefMutIterator;
use serde_json::json;
use serde_json::Value;

use crate::*;

use super::dataset::*;
use super::tree::*;
use super::RLWE;
pub struct Forest {
    pub trees: Vec<Tree>,
}

const SEED: u64 = 1;
const EXPORT: bool = true;
const NUM_THREADS: usize = 4;
const SEEDED: bool = false;

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
            if SEEDED {
                let tree: Tree =
                    Tree::new_random_tree_with_seed(depth, n_classes, f, public_key, ctx, SEED);
                trees.push(tree);
            } else {
                let tree: Tree = Tree::new_random_tree(depth, n_classes, f, public_key, ctx);
                trees.push(tree);
            }
        }
        Self { trees }
    }

    pub fn train(&mut self, dataset: &EncryptedDataset, public_key: &PublicKey, ctx: &Context) {
        println!("Dataset size: {}", dataset.records.len());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            self.trees.par_iter_mut().for_each(|tree| {
                tree.train(dataset, public_key, ctx);
            });
        });

        pool.install(|| {
            self.trees.par_iter_mut().for_each(|tree| {
                tree.leaves_majority(public_key, ctx);
            });
        });
    }

    pub fn test(&self, sample_features: &Vec<RLWE>, public_key: &PublicKey, ctx: &Context) -> LWE {
        let mut res = Vec::new();
        for tree in self.trees.iter() {
            res.push(tree.test(sample_features, public_key, ctx));
        }
        public_key.blind_majority(&res, ctx)
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
        let mut forest = Forest::new(64, 4, dataset.n_classes, dataset.f, &public_key, &ctx);
        forest.train(&dataset, &public_key, &ctx);
        let duration = start.elapsed();
        println!("Time taken: {:?}", duration);

        if EXPORT {
            forest.save_perf_to_file(duration, "iris", 64, 4);

            let filepath = "./src/comp_free/iris_forest_test_lut.json";
            forest.save_to_file(filepath, &private_key, &ctx);
        }

        // forest.print(&private_key, &ctx);
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

        let sample_features = dataset.records[0].features.clone();
        let sample_class = dataset.records[0].class.clone();

        let class_one_hot = private_key.decrypt_lwe_vector(&sample_class, &ctx);

        let ground_truth = class_one_hot.iter().position(|&x| x == 1).unwrap() as u64;

        let result = forest.test(&sample_features, &public_key, &ctx);

        println!("Sample class: {:?}", ground_truth);
        println!("Test result: {:?}", private_key.decrypt_lwe(&result, &ctx));
    }
}
