use std::time::Instant;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
// TFHE
use tfhe::core_crypto::prelude::*;

// REVOLUT
use revolut::*;

type LWE = LweCiphertext<Vec<u64>>;

use crate::model::*;

const DEBUG: bool = false;

pub struct Query {
    pub class: LWE,
    pub features: LUT,
}

impl Query {
    pub fn make_query(
        feature_vector: &Vec<u64>,
        class: &u64,
        private_key: &PrivateKey,
        ctx: &mut Context,
    ) -> Self {
        let feature_lut = LUT::from_vec(feature_vector, private_key, ctx);
        let class_lwe = private_key.allocate_and_encrypt_lwe(class.clone(), ctx);
        Self {
            class: class_lwe,
            features: feature_lut,
        }
    }

    pub fn clone(&self) -> Self {
        Self {
            class: self.class.clone(),
            features: self.features.clone(),
        }
    }
}

pub fn next_accumulators(
    accumulators: &Vec<LWE>,
    selector_bit: &LWE,
    public_key: &PublicKey,
    ctx: &Context,
) -> Vec<LWE> {
    let not_selector_bit = public_key.not_lwe(selector_bit, ctx);
    let mut nexts_accumulators = Vec::new();
    accumulators.iter().for_each(|lwe| {
        let accumulator_left = public_key.lwe_mul_encrypted_bit(lwe, &not_selector_bit, ctx);
        let accumulator_right = public_key.lwe_mul_encrypted_bit(lwe, selector_bit, ctx);
        nexts_accumulators.push(accumulator_left);
        nexts_accumulators.push(accumulator_right);
    });

    nexts_accumulators
}

pub fn blind_node_selection(
    nodes: &Vec<InternalNode>,
    accumulators: &Vec<LWE>,
    public_key: &PublicKey,
    ctx: &Context,
) -> (LWE, LWE) {
    let mut thresholds = Vec::new();
    nodes.iter().for_each(|node| {
        thresholds.push(node.threshold);
    });
    let selected_threshold = public_key.private_selection(&thresholds, accumulators, ctx);

    let mut feature_indices = Vec::new();
    nodes.iter().for_each(|node| {
        feature_indices.push(node.feature_index);
    });
    let selected_feature_index = public_key.private_selection(&feature_indices, accumulators, ctx);

    (selected_threshold, selected_feature_index)
}

pub fn blind_leaf_increment(
    accumulators: &Vec<LWE>,
    sample_class: &LWE,
    public_key: &PublicKey,
    ctx: &Context,
) -> Vec<LUT> {
    let luts = (0..accumulators.len())
        .into_par_iter()
        .map(|i| {
            let mut lut = LUT::from_lwe(&accumulators[i], public_key, ctx);
            public_key.blind_rotation_assign(
                &public_key.neg_lwe(&sample_class, ctx),
                &mut lut,
                ctx,
            );
            // lut = [0, 0, ..., 0, 1, 0, ..., 0] if acc[i] = 1
            // lut = [0, 0, ..., 0, 0, 0, ..., 0] if acc[i] = 0
            lut
        })
        .collect();

    luts
}

pub fn probonite(tree: &Tree, query: &Query, public_key: &PublicKey, ctx: &Context) -> Vec<LUT> {
    // Normal Probonite inference

    let accumulators = probonite_inference(tree, query, public_key, ctx);

    // Writing the class in the selected leaf
    let start = Instant::now();
    let luts = blind_leaf_increment(&accumulators.0, &query.class, public_key, ctx);
    let end = Instant::now();
    if DEBUG {
        println!("Last stage: {:?}", end.duration_since(start));
    }

    luts
}

pub fn probonite_inference(
    tree: &Tree,
    query: &Query,
    public_key: &PublicKey,
    ctx: &Context,
) -> (Vec<LWE>, f64) {
    let mut inference_time = 0.0;
    let start = Instant::now();
    let index = tree.root.feature_index;
    let threshold = tree.root.threshold;
    let feature = public_key.lut_extract(&query.features, index as usize, ctx);
    let b = public_key.leq_scalar(&feature, threshold, ctx);
    let not_b = public_key.not_lwe(&b, ctx);
    let mut accumulators = vec![b, not_b];
    let end = Instant::now();
    inference_time += end.duration_since(start).as_secs_f64();
    if DEBUG {
        println!("First stage: {:?}", end.duration_since(start));
    }

    // Internal Stages
    for i in 0..tree.nodes.len() {
        let start = Instant::now();
        let (threshold, feature_index) =
            blind_node_selection(&tree.nodes[i], &accumulators, public_key, ctx);
        let feature = public_key.blind_array_access(&feature_index, &query.features, ctx);
        let b = public_key.blind_lt_bma_mv(&threshold, &feature, ctx);

        if DEBUG {
            let private_key = key(ctx.parameters());
            let t_selected = private_key.decrypt_lwe(&threshold, ctx);
            let f_selected = private_key.decrypt_lwe(&feature_index, ctx);
            println!("selected:({t_selected}, {f_selected})");
            let feature_decrypted = private_key.decrypt_lwe(&feature, ctx);
            let threshold_decrypted = private_key.decrypt_lwe(&threshold, ctx);
            println!(
                "expected b: {}",
                (threshold_decrypted < feature_decrypted) as u64
            );
            private_key.debug_lwe("actual b : ", &b, ctx);
        }
        accumulators = next_accumulators(&accumulators, &b, public_key, ctx);
        let end = Instant::now();
        if DEBUG {
            println!("Internal stage {}: {:?}", i, end.duration_since(start));
        }
        inference_time += end.duration_since(start).as_secs_f64();
    }

    if DEBUG {
        println!("Total inference time: {:?}", inference_time);
    }

    (accumulators, inference_time)
}

struct TreeLUT {
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
        let num_leaves = 2u64.pow(depth as u32);
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
                    thresholds[j as usize], feature_indices[j as usize]
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

pub struct QueryLUT {
    pub class: Vec<LWE>, // one hot encoded class
    pub features: LUT,
}

fn probolut(tree: &mut TreeLUT, query: &QueryLUT, public_key: &PublicKey, ctx: &Context) {
    // First stage
    let private_key = key(ctx.parameters());
    let index = tree.root.feature_index;
    let threshold = tree.root.threshold;
    let feature = public_key.lut_extract(&query.features, index as usize, ctx);
    let b = public_key.lt_scalar(&feature, threshold, ctx);
    private_key.debug_lwe("b", &b, ctx);

    // Internal Stages
    let mut selector = b.clone();
    private_key.debug_lwe("selector", &selector, ctx);
    for i in 0..tree.stages.len() {
        let (lut_threshold, lut_index) = &tree.stages[i];
        let feature_index = public_key.blind_array_access(&selector, &lut_index, ctx);
        let threshold = public_key.blind_array_access(&selector, &lut_threshold, ctx);
        let feature = public_key.blind_array_access(&feature_index, &query.features, ctx);
        let b = public_key.blind_lt_bma_mv(&feature, &threshold, ctx);
        selector = public_key.lwe_mul_add(&b, &selector, 2);
        private_key.debug_lwe("b", &b, ctx);
        private_key.debug_lwe("selector_updated", &selector, ctx);
    }

    // Last stage
    private_key.debug_lwe("selector", &selector, ctx);
    private_key.debug_lwe("class[0]", &query.class[0], ctx);
    private_key.debug_lwe("class[1]", &query.class[1], ctx);
    private_key.debug_lwe("class[2]", &query.class[2], ctx);
    for c in 0..tree.n_classes {
        public_key.blind_array_increment(
            &mut tree.leaves[c as usize],
            &selector,
            &query.class[c as usize],
            ctx,
        );
    }
}

#[cfg(test)]
mod tests {
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

        let mut tree = TreeLUT::from_tree(&classical_tree, TREE_DEPTH, N_CLASSES, &ctx);

        tree.print_tree(&private_key, &ctx);

        // Create a query
        let query = QueryLUT {
            class: vec![
                private_key.allocate_and_encrypt_lwe(1, &mut ctx),
                private_key.allocate_and_encrypt_lwe(0, &mut ctx),
                private_key.allocate_and_encrypt_lwe(0, &mut ctx),
            ],
            features: LUT::from_vec(
                &vec![31; ctx.full_message_modulus() as usize],
                &private_key,
                &mut ctx,
            ),
        };

        // Run probolut
        let start = Instant::now();
        probolut(&mut tree, &query, &public_key, &ctx);
        let end = Instant::now();
        println!("Time taken: {:?}", end.duration_since(start));

        tree.print_tree(&private_key, &ctx);
    }

    #[test]
    fn test_probonite() {
        // Initialize context
        let mut ctx = Context::from(PARAM_MESSAGE_5_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        const TREE_DEPTH: u64 = 5;
        const N_CLASSES: u64 = 3;
        let f = ctx.full_message_modulus() as u64;

        // Generate a random tree
        let mut tree = Tree::generate_random_tree(TREE_DEPTH, N_CLASSES, f, &ctx);

        // Create a query with test data
        let feature_vector = vec![1; ctx.full_message_modulus() as usize];
        let features_vector = LUT::from_vec(&feature_vector, &private_key, &mut ctx);
        let class = private_key.allocate_and_encrypt_lwe(0, &mut ctx);

        let query = Query {
            features: features_vector,
            class: class,
        };

        // Run probonite
        let start = Instant::now();
        let _ = probonite(&mut tree, &query, &public_key, &ctx);
        let end = Instant::now();
        println!("Time probonite taken: {:?}", end.duration_since(start));

        tree.print_tree(&private_key, &ctx);
    }
}
