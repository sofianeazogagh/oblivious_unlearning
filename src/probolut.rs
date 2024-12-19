use crate::dataset::*;
use crate::model::{InternalNode, Leaf, Root, Tree};
use revolut::*;
use tfhe::core_crypto::prelude::LweCiphertext;

pub struct TreeLUT {
    pub root: Root,
    pub stages: Vec<(LUT, LUT)>, // stages[i] contains (lut_index, lut_threshold) for the i-th stage
    pub leaves: Vec<LUT>,        // leaves[i] contains the lut for the i-th class
    pub n_classes: u64,
    pub depth: u64,
}

impl TreeLUT {
    pub fn new() -> Self {
        Self {
            root: Root {
                threshold: 0,
                feature_index: 0,
            },
            stages: Vec::new(),
            leaves: Vec::new(),
            n_classes: 0,
            depth: 0,
        }
    }

    pub fn from_tree(tree: &Tree, depth: u64, n_classes: u64, ctx: &Context) -> Self {
        let mut stages = Vec::new();

        // Pour chaque niveau de l'arbre (sauf la racine)
        for level in 0..depth - 1 {
            let mut thresholds = Vec::new();
            let mut feature_indices = Vec::new();

            for internal_node in tree.nodes[level as usize].iter() {
                thresholds.push(internal_node.threshold);
                feature_indices.push(internal_node.feature_index);
            }

            let lut_threshold = LUT::from_vec_trivially(&thresholds, ctx);
            let lut_index = LUT::from_vec_trivially(&feature_indices, ctx);
            stages.push((lut_index, lut_threshold));
        }

        // Initialiser les feuilles
        let leaves = (0..n_classes)
            .map(|_| LUT::from_vec_trivially(&vec![0], ctx))
            .collect();

        Self {
            root: tree.root.clone(),
            stages,
            leaves,
            n_classes,
            depth,
        }
    }

    pub fn generate_random_tree(depth: u64, n_classes: u64, dim: u64, ctx: &Context) -> Self {
        let mut tree = Self::new();
        tree.depth = depth;
        tree.n_classes = n_classes;

        // Generate the root
        tree.root.threshold = rand::random::<u64>() % ctx.full_message_modulus() as u64;
        tree.root.feature_index = rand::random::<u64>() % dim;

        // Generate the nodes
        // Generate internal nodes for each level
        for _ in 1..depth {
            let mut vec_index = Vec::new();
            let mut vec_threshold = Vec::new();

            // Number of nodes at this level is 2^level
            let num_nodes = 2u64.pow(tree.stages.len() as u32 + 1);

            for _ in 0..num_nodes {
                vec_index.push(rand::random::<u64>() % dim);
                vec_threshold.push(rand::random::<u64>() % ctx.full_message_modulus() as u64);
            }
            tree.stages.push((
                LUT::from_vec_trivially(&vec_threshold, ctx),
                LUT::from_vec_trivially(&vec_index, ctx),
            ));
        }

        // Generate the leaves
        let num_leaves = n_classes;
        for _ in 0..num_leaves {
            let counts = LUT::from_vec_trivially(&vec![0u64; n_classes as usize], ctx);
            tree.leaves.push(counts);
        }

        tree
    }

    pub fn print_tree(&self, private_key: &PrivateKey, ctx: &Context) {
        println!(
            "Root: ({}, {})",
            self.root.feature_index, self.root.threshold
        );
        for (i, stage) in self.stages.iter().enumerate() {
            let number_of_nodes = 2u64.pow(i as u32 + 1);

            let thresholds = stage.0.to_array(&private_key, ctx);
            let feature_indices = stage.1.to_array(&private_key, ctx);

            let thresholds = &thresholds[0..number_of_nodes as usize];
            let feature_indices = &feature_indices[0..number_of_nodes as usize];
            for j in 0..number_of_nodes {
                print!(
                    "({:?}, {:?})    ",
                    feature_indices[j as usize], thresholds[j as usize]
                );
            }
            println!();
        }

        println!("Leaves:");
        for (i, leaf) in self.leaves.iter().enumerate() {
            let class_counts = leaf.to_array(&private_key, ctx);
            println!("Leaf {}: {:?}", i, class_counts);
        }
    }
}

pub fn probolut_training(
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

pub fn probolut_inference(
    tree: &TreeLUT,
    query: &LUT,
    public_key: &PublicKey,
    ctx: &Context,
) -> LweCiphertext<Vec<u64>> {
    // First stage
    let private_key = key(ctx.parameters());
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

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use revolut::*;
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
        let feature_vector = vec![1; ctx.full_message_modulus() as usize];
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
        let counts = probolut_training(&tree, &sample, &public_key, &ctx);
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
